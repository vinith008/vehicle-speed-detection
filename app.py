# app.py
import cv2
import numpy as np
import time
import threading
from queue import Queue
from collections import defaultdict, deque
from flask import Flask, render_template, Response
from ultralytics import YOLO
from scipy.spatial import distance as dist

app = Flask(__name__)

# ================= CONFIGURATION =================
IP_CAMERA_URL = "http://10.32.222.23:8080/videofeed"

src_points = np.float32([
    [300, 200],
    [980, 200],
    [1100, 680],
    [180, 680]
])

dst_points = np.float32([
    [0, 0],
    [12, 0],
    [12, 35],
    [0, 35]
])

MODEL = "yolov8m.pt"
CONF = 0.45
VEHICLE_CLASSES = {2, 3, 5, 7}  # car=2, motorcycle=3, bus=5, truck=7

EMA_ALPHA = 0.3
MIN_KMH = 5
MAX_KMH = 180
STOP_THRESHOLD_M = 0.15  # meters — consider stopped

FRAME_QUEUE_MAX = 5
# ================================================

model = YOLO(MODEL)
print(f"Loaded {MODEL} on {model.device}")

try:
    H, _ = cv2.findHomography(src_points, dst_points)
except:
    print("Warning: Homography failed → using fallback pixel scaling")
    H = None

track_world_history = defaultdict(lambda: deque(maxlen=12))
smoothed_speed = {}
last_valid_speed = {}

def world_coords(x, y):
    if H is None:
        return x * 0.05, y * 0.05  # fallback
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H)
    return warped[0][0][0], warped[0][0][1]

# Global camera & queue
camera = None
frame_queue = Queue(maxsize=FRAME_QUEUE_MAX)
stop_event = threading.Event()

def connect_camera():
    global camera
    if camera is not None:
        camera.release()
    camera = cv2.VideoCapture(IP_CAMERA_URL, cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    camera.set(cv2.CAP_PROP_FPS, 30)
    print("Camera (re)connected")
    return camera.isOpened()

def capture_worker():
    while not stop_event.is_set():
        if camera is None or not camera.isOpened():
            time.sleep(1.5)
            connect_camera()
            continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.2)
            continue

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame)

# Start capture thread once
threading.Thread(target=capture_worker, daemon=True).start()

def gen_frames():
    fps_timer = time.time()
    fps_count = 0
    fps_val = 0

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.004)
            continue

        frame = frame_queue.get()
        now = time.time()

        # FPS calculation
        fps_count += 1
        if now - fps_timer > 1.0:
            fps_val = fps_count
            fps_count = 0
            fps_timer = now

        # YOLO tracking
        results = model.track(
            frame,
            persist=True,
            conf=CONF,
            tracker="botsort.yaml",
            verbose=False
        )[0]

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps_val}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clss):
                if int(cls) not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                wx, wy = world_coords(cx, cy)
                track_world_history[tid].append((wx, wy, now))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 200), -1)

                if len(track_world_history[tid]) >= 5:
                    pts = list(track_world_history[tid])
                    total_dist = 0.0
                    for i in range(1, len(pts)):
                        total_dist += dist.euclidean(pts[i-1][:2], pts[i][:2])

                    dt = pts[-1][2] - pts[0][2]

                    if dt > 0.12:
                        speed_ms = total_dist / dt
                        speed_kmh = speed_ms * 3.6

                        # Stop detection
                        if total_dist < STOP_THRESHOLD_M:
                            display_speed = 0
                        else:
                            prev = smoothed_speed.get(tid, speed_kmh)
                            smooth = EMA_ALPHA * speed_kmh + (1 - EMA_ALPHA) * prev
                            smoothed_speed[tid] = smooth

                            if MIN_KMH <= smooth <= MAX_KMH:
                                display_speed = int(round(smooth))
                            else:
                                display_speed = last_valid_speed.get(tid, 0)

                        last_valid_speed[tid] = display_speed

                        cv2.putText(frame, f"ID {tid}  {display_speed} km/h",
                                    (cx - 70, cy - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        # Draw calibration rectangle
        pts_poly = src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_poly], isClosed=True, color=(0, 200, 255), thickness=2)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if camera is not None:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        stop_event.set()
        if camera is not None:
            camera.release()
        print("Server stopped cleanly")