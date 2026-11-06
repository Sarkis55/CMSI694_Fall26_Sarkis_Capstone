"""

Turns raw MediaPipe FaceLandmarker landmarks into the full drowsiness
feature set: EAR, PERCLOS, MAR (yawning), head pose (pitch/yaw/roll,
droop + turn flags), and blink rate / duration — all behind one
function, so Sprint 3 (dashboard) and Sprint 4 (ML training) consume
the exact same metrics without duplicating any of this math.

Usage:
    from features import FeatureTracker

    tracker = FeatureTracker()          # one per session
    ...
    metrics = tracker.compute(landmarks, w, h)
    # metrics is a flat dict — see FeatureTracker.compute() docstring

Head pose note: pose is estimated from 6 face landmarks with OpenCV's
solvePnP (not MediaPipe's transformation matrix, whose axis/handedness
conventions are easy to get wrong). Output convention:
    pitch  + = head tilted DOWN (nodding off), - = up
    yaw    sign depends on mirroring; only |yaw delta| is used for flags
    roll   head tilted toward a shoulder
Angles are shown relative to a neutral baseline captured at startup.
"""

import math
import time
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Landmark index sets (MediaPipe's 478-point face mesh)
# ---------------------------------------------------------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Outer lip landmarks used for MAR: top center, bottom center, corners
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Landmarks used for head-pose (solvePnP)
POSE_NOSE = 1
POSE_CHIN = 152
POSE_EYE_CORNERS = (33, 263)      # outer eye corners
POSE_MOUTH_CORNERS = (61, 291)

# Generic 3D face model, OpenCV camera convention (x right, y down,
# z away from camera), nose tip at origin, rough millimetre scale.
# Order: nose, chin, eye (image-left), eye (image-right),
#        mouth (image-left), mouth (image-right)
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, 330.0, 65.0),
    (-225.0, -170.0, 135.0),
    (225.0, -170.0, 135.0),
    (-150.0, 150.0, 125.0),
    (150.0, 150.0, 125.0),
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Thresholds — starting points; tune against your own footage/lighting
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
PERCLOS_WINDOW_SECONDS = 45      # rolling window for PERCLOS + blink rate
BLINK_MIN_DURATION = 0.05        # ignore sub-50ms blips (landmark jitter)
YAWN_MIN_FRAMES = 8              # consecutive high-MAR frames to count as a yawn

# Head pose: measured as CHANGE from a neutral baseline, held for a
# duration in SECONDS (so it behaves the same at 15 fps or 30 fps).
CALIBRATION_SECONDS = 3          # look straight ahead for this long at startup
POSE_SMOOTHING = 0.5             # 0-1, weight of the newest frame (lower = smoother)
HEAD_DROOP_DELTA_DEG = 15        # degrees of downward pitch from neutral
HEAD_DROOP_MIN_SECONDS = 1.0
HEAD_DROOP_SIGN = 1              # +1: positive pitch delta = nodding DOWN
HEAD_TURN_DELTA_DEG = 25         # degrees of yaw from neutral (either way)
HEAD_TURN_MIN_SECONDS = 1.0


def _dist(p1, p2):
    return math.dist(p1, p2)


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """6-point EAR (Soukupova & Cech) for one eye."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w, h):
    """Same vertical/horizontal ratio idea as EAR, applied to the mouth."""
    top = (landmarks[MOUTH_TOP].x * w, landmarks[MOUTH_TOP].y * h)
    bottom = (landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h)
    left = (landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h)
    right = (landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h)
    vertical = _dist(top, bottom)
    horizontal = _dist(left, right)
    return vertical / horizontal


def estimate_head_pose(landmarks, w, h, prev_vectors=None):
    """
    Estimate head pitch/yaw/roll (degrees) from 6 landmarks via solvePnP.

    Left/right pairs are assigned by their actual x position in the image,
    so it works whether or not the frame was mirrored.

    prev_vectors: (rvec, tvec) from the previous frame, used as the
    starting guess so the solution stays stable frame to frame.

    Returns (pose_dict, (rvec, tvec), nose_line) or None if the solve fails.
    nose_line = ((x0, y0), (x1, y1)) in pixels: a short line from the nose
    pointing the way the face points — handy for visual debugging.
    """
    def px(i):
        return (landmarks[i].x * w, landmarks[i].y * h)

    eye_l, eye_r = sorted((px(POSE_EYE_CORNERS[0]), px(POSE_EYE_CORNERS[1])), key=lambda p: p[0])
    mouth_l, mouth_r = sorted((px(POSE_MOUTH_CORNERS[0]), px(POSE_MOUTH_CORNERS[1])), key=lambda p: p[0])
    image_points = np.array(
        [px(POSE_NOSE), px(POSE_CHIN), eye_l, eye_r, mouth_l, mouth_r], dtype=np.float64
    )

    focal = float(w)  # rough pinhole approximation (~60 degree FOV)
    camera = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)

    if prev_vectors is not None:
        rvec, tvec = prev_vectors[0].copy(), prev_vectors[1].copy()
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera, None, rvec, tvec, True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(rotation)[0]

    # line from nose tip toward the camera (the direction the face points)
    tip, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -300.0)]), rvec, tvec, camera, None
    )
    nose_line = (tuple(tip[0, 0]), tuple(tip[1, 0]))

    pose = {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
    return pose, (rvec, tvec), nose_line


class FeatureTracker:
    """
    Holds rolling state across frames — EAR history for PERCLOS, blink
    events, and the head-pose baseline. Create ONE instance per
    session/stream and call compute() once per frame; don't recreate it
    every frame or the rolling windows and calibration reset.
    """

    def __init__(self, perclos_window=PERCLOS_WINDOW_SECONDS):
        self.perclos_window = perclos_window
        self.ear_history = deque()     # (timestamp, is_closed)
        self.blink_events = deque()    # (start_time, duration) for completed blinks
        self._blink_start = None
        self.yawn_frame_count = 0
        self.reset_calibration()

    # ----- head pose -------------------------------------------------------

    def reset_calibration(self):
        """Forget the neutral head pose and re-run the startup calibration."""
        self.pose_baseline = None
        self._calib_start = None
        self._calib_samples = []
        self._pose_smooth = None
        self._pose_vectors = None
        self._held_since = {}
        self.nose_line = None          # for the main script to draw

    @property
    def calibrating(self):
        return self.pose_baseline is None

    def _held(self, key, active, now, min_seconds):
        """True once `active` has stayed True continuously for min_seconds."""
        if not active:
            self._held_since.pop(key, None)
            return False
        start = self._held_since.setdefault(key, now)
        return (now - start) >= min_seconds

    def _update_head_pose(self, now, landmarks, w, h):
        """Returns dict with pose, delta, flags, calibration progress (or None values)."""
        empty = {"pose": None, "delta": None, "drooping": False,
                 "turned": False, "progress": 0.0 if self.calibrating else 1.0}

        est = estimate_head_pose(landmarks, w, h, self._pose_vectors)
        if est is None:
            self.nose_line = None
            return empty
        raw, self._pose_vectors, self.nose_line = est

        if self._pose_smooth is None:
            self._pose_smooth = dict(raw)
        else:
            a = POSE_SMOOTHING
            self._pose_smooth = {k: a * raw[k] + (1 - a) * self._pose_smooth[k] for k in raw}
        pose = dict(self._pose_smooth)

        if self.calibrating:
            if self._calib_start is None:
                self._calib_start = now
            self._calib_samples.append(pose)
            progress = min(1.0, (now - self._calib_start) / CALIBRATION_SECONDS)
            if progress >= 1.0:
                self.pose_baseline = {
                    k: float(np.median([s[k] for s in self._calib_samples])) for k in pose
                }
            return {**empty, "pose": pose, "progress": progress}

        delta = {k: pose[k] - self.pose_baseline[k] for k in pose}
        drooping = self._held(
            "droop", delta["pitch"] * HEAD_DROOP_SIGN > HEAD_DROOP_DELTA_DEG,
            now, HEAD_DROOP_MIN_SECONDS)
        turned = self._held(
            "turn", abs(delta["yaw"]) > HEAD_TURN_DELTA_DEG,
            now, HEAD_TURN_MIN_SECONDS)
        return {"pose": pose, "delta": delta, "drooping": drooping,
                "turned": turned, "progress": 1.0}

    # ----- PERCLOS + blinks ------------------------------------------------

    def _update_perclos(self, now, is_closed):
        self.ear_history.append((now, is_closed))
        cutoff = now - self.perclos_window
        while self.ear_history and self.ear_history[0][0] < cutoff:
            self.ear_history.popleft()

        if not self.ear_history:
            return 0.0
        closed_count = sum(1 for _, c in self.ear_history if c)
        return closed_count / len(self.ear_history)

    def _update_blinks(self, now, is_closed):
        if is_closed and self._blink_start is None:
            self._blink_start = now
        elif not is_closed and self._blink_start is not None:
            duration = now - self._blink_start
            if duration >= BLINK_MIN_DURATION:
                self.blink_events.append((self._blink_start, duration))
            self._blink_start = None

        # keep blink history on the same rolling horizon as PERCLOS
        cutoff = now - self.perclos_window
        while self.blink_events and self.blink_events[0][0] < cutoff:
            self.blink_events.popleft()

    # ----- the one public entry point --------------------------------------

    def compute(self, landmarks, w, h, now=None):
        """
        Call once per frame with that frame's landmarks. Returns a flat dict:

            timestamp, ear, eyes_closed, perclos, mar, yawning,
            blink_rate_per_min, avg_blink_duration_sec,
            head_pitch, head_yaw, head_roll,                  (absolute, smoothed)
            head_pitch_delta, head_yaw_delta, head_roll_delta, (vs. neutral baseline)
            head_drooping, head_turned,
            calibrating, calibration_progress

        head_*_delta values are None while calibrating (first few seconds).
        """
        now = now if now is not None else time.time()

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        eyes_closed = ear < EAR_THRESHOLD

        perclos = self._update_perclos(now, eyes_closed)
        self._update_blinks(now, eyes_closed)

        blink_count = len(self.blink_events)
        window_minutes = self.perclos_window / 60.0
        blink_rate = blink_count / window_minutes if window_minutes > 0 else 0.0
        avg_blink_duration = (
            sum(d for _, d in self.blink_events) / blink_count
            if blink_count else 0.0
        )

        mar = mouth_aspect_ratio(landmarks, w, h)
        self.yawn_frame_count = self.yawn_frame_count + 1 if mar > MAR_THRESHOLD else 0
        yawning = self.yawn_frame_count >= YAWN_MIN_FRAMES

        hp = self._update_head_pose(now, landmarks, w, h)
        pose = hp["pose"] or {"pitch": None, "yaw": None, "roll": None}
        delta = hp["delta"] or {"pitch": None, "yaw": None, "roll": None}

        def r1(v):
            return round(v, 1) if v is not None else None

        return {
            "timestamp": now,
            "ear": round(ear, 3),
            "eyes_closed": eyes_closed,
            "perclos": round(perclos, 3),
            "mar": round(mar, 3),
            "yawning": yawning,
            "blink_rate_per_min": round(blink_rate, 2),
            "avg_blink_duration_sec": round(avg_blink_duration, 3),
            "head_pitch": r1(pose["pitch"]),
            "head_yaw": r1(pose["yaw"]),
            "head_roll": r1(pose["roll"]),
            "head_pitch_delta": r1(delta["pitch"]),
            "head_yaw_delta": r1(delta["yaw"]),
            "head_roll_delta": r1(delta["roll"]),
            "head_drooping": hp["drooping"],
            "head_turned": hp["turned"],
            "calibrating": self.calibrating,
            "calibration_progress": round(hp["progress"], 2),
        }