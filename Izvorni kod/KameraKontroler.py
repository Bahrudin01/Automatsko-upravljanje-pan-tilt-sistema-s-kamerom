import os
import time
import threading
import atexit
import socket
import signal
import cv2
from flask import Flask, Response, request, jsonify
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

PINS = [23, 24, 25, 8]
for p in PINS:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, 0)

SEQUENCE = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]
DELAY = 0.002

DEADBAND = 0.07
INVERT_DIR = False

MANUAL_STEPS_DEFAULT = 4        
MANUAL_STEPS_MAX = 400          

AUTO_MIN_STEPS = 1              
AUTO_MAX_STEPS = 30             
AUTO_GAIN = 120.0               

JPEG_QUALITY = 80
STREAM_SLEEP = 0.03

app = Flask(__name__)

_state_lock = threading.Lock()
_auto_mode = True


def motor_off():
    for p in PINS:
        GPIO.output(p, 0)


class MotorThread:
    def __init__(self, pins, sequence, delay):
        self.pins = pins
        self.seq = sequence
        self.delay = delay
        self.idx = 0
        self._lock = threading.Lock()
        self._dir = 0
        self._steps_remaining = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def command_steps(self, direction: int, steps: int):
        if direction == 0:
            return
        if steps <= 0:
            return
        direction = 1 if direction > 0 else -1
        with self._lock:
            self._dir = direction
            self._steps_remaining = int(steps)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        motor_off()

    def _write_step(self, step):
        for pin, val in zip(self.pins, step):
            GPIO.output(pin, val)

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                direction = self._dir
                steps_left = self._steps_remaining
            if direction != 0 and steps_left > 0:
                self.idx = (self.idx + direction) % len(self.seq)
                self._write_step(self.seq[self.idx])

                with self._lock:
                    if self._steps_remaining > 0:
                        self._steps_remaining -= 1
                time.sleep(self.delay)
            else:
                motor_off()
                time.sleep(0.005)


def find_haarcascade():
    filename = "haarcascade_frontalface_default.xml"
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        p = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(p):
            return p

    candidates = [
        f"/usr/share/opencv4/haarcascades/{filename}",
        f"/usr/share/opencv/haarcascades/{filename}",
        f"/usr/local/share/opencv4/haarcascades/{filename}",
        f"/usr/local/share/opencv/haarcascades/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("Could not find Haar cascade XML.")


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class Streamer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.face_cascade = cv2.CascadeClassifier(find_haarcascade())
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")

        self.motor = MotorThread(PINS, SEQUENCE, DELAY)

        self._lock = threading.Lock()
        self._jpeg = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.motor.stop()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def manual_move_steps(self, direction, steps):
        self.motor.command_steps(direction, steps)

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with _state_lock:
                auto_on = _auto_mode

            if auto_on:
                h, w = frame.shape[:2]
                frame_cx = w / 2.0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    x, y, fw, fh = map(int, (x, y, fw, fh))

                    face_cx = x + fw / 2.0
                    offset = (face_cx - frame_cx) / w 

                    direction = 0
                    if offset < -DEADBAND:
                        direction = -1
                    elif offset > DEADBAND:
                        direction = 1

                    if INVERT_DIR:
                        direction *= -1

                    if direction != 0:
                        steps = int(AUTO_GAIN * abs(offset))
                        if steps < AUTO_MIN_STEPS:
                            steps = AUTO_MIN_STEPS
                        if steps > AUTO_MAX_STEPS:
                            steps = AUTO_MAX_STEPS
                        self.motor.command_steps(direction, steps)

                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

                cv2.line(frame, (int(frame_cx), 0), (int(frame_cx), h), (255, 255, 255), 1)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()

            time.sleep(0.001)


streamer = None


@app.route("/")
def index():
    ip = get_lan_ip()
    with _state_lock:
        auto_on = _auto_mode
    checked = "checked" if auto_on else ""
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Camera</title>
  <style>
    body {{ margin:0; background:#0b0b0b; color:#fff; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
    .wrap {{ max-width: 900px; margin: 0 auto; padding: 14px; }}
    .card {{ background:#141414; border:1px solid #2a2a2a; border-radius:12px; padding:12px; }}
    .row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
    .spacer {{ flex:1; }}
    .hint {{ opacity:0.85; font-size: 14px; }}
    .cam {{ width:100%; max-height: 520px; object-fit: contain; background:#000; border-radius:12px; border:1px solid #2a2a2a; }}
    button {{
      padding: 10px 16px; border-radius: 10px; border: 1px solid #3a3a3a;
      background: #1f1f1f; color: #fff; cursor: pointer; font-size: 15px;
      transition: opacity 120ms ease, filter 120ms ease;
    }}
    button:disabled {{ opacity:0.35; filter: grayscale(1); cursor:not-allowed; }}
    label {{ display:flex; align-items:center; gap:10px; }}
    input[type="checkbox"] {{ transform: scale(1.2); }}
    .small {{ font-size: 13px; opacity:0.8; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row">
        <div class="hint">LAN: <b>http://{ip}:5000/</b></div>
        <div class="spacer"></div>
        <label>
          <input id="autoBox" type="checkbox" {checked}/>
          Automatic facial
        </label>
      </div>

      <div class="row" style="margin-top:12px;">
        <button id="leftBtn">Left (CW)</button>
        <button id="rightBtn">Right (CCW)</button>
        <div class="hint small" id="statusText"></div>
      </div>

      <div class="row" style="margin-top:10px;">
        <div class="hint small">
          Manual step size:
          <input id="stepsBox" type="number" min="1" max="{MANUAL_STEPS_MAX}" value="{MANUAL_STEPS_DEFAULT}"
                 style="width:90px; padding:6px; border-radius:8px; border:1px solid #3a3a3a; background:#101010; color:#fff;">
          <span class="small">(try 1–10)</span>
        </div>
      </div>
    </div>

    <div style="height:12px;"></div>

    <img class="cam" src="/video_feed" />
  </div>

<script>
  const autoBox = document.getElementById('autoBox');
  const leftBtn = document.getElementById('leftBtn');
  const rightBtn = document.getElementById('rightBtn');
  const stepsBox = document.getElementById('stepsBox');
  const statusText = document.getElementById('statusText');

  function setButtonsEnabled() {{
    const autoOn = autoBox.checked;
    leftBtn.disabled = autoOn;
    rightBtn.disabled = autoOn;
    stepsBox.disabled = autoOn;
  }}

  async function setMode(autoOn) {{
    try {{
      const r = await fetch(`/api/mode?auto=${{autoOn ? 1 : 0}}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || 'mode failed');
      statusText.textContent = '';
    }} catch(e) {{
      statusText.textContent = String(e);
    }}
  }}

  async function move(dir) {{
    try {{
      const steps = Math.max(1, Math.min(999999, parseInt(stepsBox.value || "1", 10)));
      const r = await fetch(`/api/move?dir=${{encodeURIComponent(dir)}}&steps=${{encodeURIComponent(String(steps))}}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || 'move failed');
      statusText.textContent = '';
    }} catch(e) {{
      statusText.textContent = String(e);
    }}
  }}

  autoBox.addEventListener('change', async () => {{
    setButtonsEnabled();
    await setMode(autoBox.checked);
  }});

  leftBtn.addEventListener('click', async () => {{
    if (leftBtn.disabled) return;
    leftBtn.disabled = true; rightBtn.disabled = true;
    await move('left');
    setTimeout(() => setButtonsEnabled(), 80);
  }});

  rightBtn.addEventListener('click', async () => {{
    if (rightBtn.disabled) return;
    leftBtn.disabled = true; rightBtn.disabled = true;
    await move('right');
    setTimeout(() => setButtonsEnabled(), 80);
  }});

  setButtonsEnabled();
</script>
</body>
</html>
"""


def gen():
    boundary = b"--frame"
    while True:
        if streamer is None:
            time.sleep(0.05)
            continue

        frame = streamer.get_jpeg()
        if frame is None:
            time.sleep(0.01)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
        yield frame + b"\r\n"

        time.sleep(STREAM_SLEEP)


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/mode")
def api_mode():
    global _auto_mode
    v = request.args.get("auto", "").strip()
    if v not in ("0", "1"):
        return jsonify({"error": "auto must be 0 or 1"}), 400
    with _state_lock:
        _auto_mode = (v == "1")
    return jsonify({"ok": True, "auto": _auto_mode})


@app.route("/api/move")
def api_move():
    if streamer is None:
        return jsonify({"error": "streamer not ready"}), 503

    with _state_lock:
        auto_on = _auto_mode

    if auto_on:
        return jsonify({"error": "manual buttons disabled while auto is on"}), 409

    d = request.args.get("dir", "").strip().lower()
    if d not in ("left", "right"):
        return jsonify({"error": "dir must be left or right"}), 400

    # steps parameter (tiny moves!)
    steps_raw = request.args.get("steps", "").strip()
    if steps_raw == "":
        steps = MANUAL_STEPS_DEFAULT
    else:
        try:
            steps = int(steps_raw)
        except Exception:
            return jsonify({"error": "steps must be an integer"}), 400

    if steps < 1:
        steps = 1
    if steps > MANUAL_STEPS_MAX:
        steps = MANUAL_STEPS_MAX

    direction = 1 if d == "left" else -1
    if INVERT_DIR:
        direction *= -1

    streamer.manual_move_steps(direction, steps)
    return jsonify({"ok": True, "dir": d, "steps": steps})


def cleanup(*_):
    global streamer
    if streamer is not None:
        try:
            streamer.stop()
        finally:
            streamer = None


atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == "__main__":
    streamer = Streamer()
    ip = get_lan_ip()
    print(f"Camera stream available at: http://{ip}:5000/")
    print("Also available locally at: http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)
