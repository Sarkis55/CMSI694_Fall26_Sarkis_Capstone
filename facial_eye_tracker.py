"""
Facial + Eye Tracker
--------------------------------------------------
Live webcam face + eye tracking using OpenCV (capture/display) and
MediaPipe Face Landmarker (face + eye landmark detection).

Each frame, this computes the Eye Aspect Ratio (EAR) for both eyes.
EAR drops sharply when the eyes close. If EAR stays below a threshold
for a sustained number of frames, the script flags "DROWSY".

"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# Setting up Pre-trained model for Facial Landmarking
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model (one-time, ~4 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

#   Eye landmark indices (MediaPipe's 478-point face mesh)
#    6-point EAR sets, ordered [p1, p2, p3, p4, p5, p6] per
#    Soukupova & Cech, "Real-Time Eye Blink Detection using Facial
#    Landmarks" — p1/p4 are the eye corners, p2/p3/p5/p6 trace the lids.
# ---------------------------------------------------------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25       # below this, an eye is considered "closed"
CLOSED_FRAMES_LIMIT = 15    # consecutive closed frames (~0.5s @ 30fps) before alert


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """Compute EAR = (vertical distances) / (2 * horizontal distance)."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = (np.array(p) for p in pts)
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# Build the Face Landmarker (VIDEO mode = frame-by-frame w/ timestamps)
# ---------------------------------------------------------------------------
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)


# Cam Loop
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0). Try a different camera index.")

closed_frame_count = 0
prev_time = time.time()

print("Starting webcam. Press 'q' or ESC to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    status_text = "No face detected"
    status_color = (0, 0, 255)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # draw eye contour points so you can see what's being tracked
        for idx in LEFT_EYE + RIGHT_EYE:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if avg_ear < EAR_THRESHOLD:
            closed_frame_count += 1
        else:
            closed_frame_count = 0

        if closed_frame_count >= CLOSED_FRAMES_LIMIT:
            status_text = "DROWSY"
            status_color = (0, 0, 255)
        else:
            status_text = "Alert"
            status_color = (0, 200, 0)

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS counter (useful for judging if this'll run fast enough live)
    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.0f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("Drowsiness Detection - POC", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):  # 'q' or ESC
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()