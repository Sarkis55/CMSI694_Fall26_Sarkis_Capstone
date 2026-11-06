"""
window_features.py — Sprint 4

Turns one WINDOW of per-frame metrics (e.g. 30 seconds) into a single row
of features for the classifier.

This is shared by:
  - build_windows.py  (training: windows cut from the dataset CSVs)
  - Sprint 5's live dashboard (inference: the last N seconds of live frames)
Using the same function in both places keeps training and live features
identical.

Input: a pandas DataFrame, one row per frame, with the columns that
extract_features.py writes (in the live system, t_sec = seconds since start):

    t_sec, face_detected, ear, eyes_closed, perclos, mar, yawning,
    head_pitch, head_yaw, head_roll, head_pitch_delta, head_yaw_delta,
    head_drooping, head_turned

Rows where no face was found have blank metrics (NaN) and face_detected = 0.
"""

import numpy as np
import pandas as pd

LONG_CLOSURE_SECONDS = 0.5     # a closure this long is more than a blink
MIN_FACE_FRACTION = 0.5        # windows with less face coverage are discarded
REL_CLOSED_RATIO = 0.70        # eye counts as "closed" below 70% of the person's open-eye EAR

# Features measured relative to the person's OWN open-eye EAR (so people with
# naturally narrow or wide eyes are comparable). They need ear_open_level:
#   - training: build_windows.py passes each video's 90th-percentile EAR
#   - live (Sprint 5): pass a running 90th percentile of the last few minutes
REL_FEATURE_NAMES = ["ear_rel_mean", "ear_rel_p10", "closed_rel_frac",
                     "closure_rel_per_min", "closure_rel_max_dur", "long_closures_rel"]


def _nan_stat(fn, values):
    """Apply fn to the non-NaN values; NaN if there are none."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    return float(fn(x)) if x.size else np.nan


def closure_runs(t, closed):
    """Durations (seconds) of each consecutive eyes-closed stretch."""
    runs = []
    start = None
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    for i in range(len(t)):
        if closed[i] == 1:                 # NaN == 1 is False, so lost-face frames end a run
            if start is None:
                start = i
        elif start is not None:
            runs.append(float(t[i] - t[start]))
            start = None
    if start is not None:                  # closure still going at the end of the window
        runs.append(float(t[-1] - t[start] + step))
    return runs


def window_features(df, win_seconds, ear_open_level=None):
    """
    Returns a dict of features for one window, or None if the window is
    unusable (too few frames, or the face was missing for most of it).

    ear_open_level: this person's typical eyes-open EAR (see REL_FEATURE_NAMES).
    If None, the person-relative features are NaN.
    """
    if len(df) < 2:
        return None

    face_frac = float((df["face_detected"].to_numpy(dtype=float) == 1).mean())
    if face_frac < MIN_FACE_FRACTION:
        return None

    t = df["t_sec"].to_numpy(dtype=float)
    col = lambda name: df[name].to_numpy(dtype=float)
    f = {"face_found_frac": face_frac}

    # --- eyes -------------------------------------------------------------
    ear = col("ear")
    f["ear_mean"] = _nan_stat(np.mean, ear)
    f["ear_std"] = _nan_stat(np.std, ear)
    f["ear_min"] = _nan_stat(np.min, ear)
    f["ear_p10"] = _nan_stat(lambda x: np.percentile(x, 10), ear)

    closed = col("eyes_closed")
    f["closed_frac"] = _nan_stat(np.mean, closed)          # PERCLOS over this window
    perclos = df["perclos"].dropna()
    f["perclos_last"] = float(perclos.iloc[-1]) if len(perclos) else np.nan

    runs = closure_runs(t, closed)
    minutes = win_seconds / 60.0
    f["closure_per_min"] = len(runs) / minutes
    f["closure_mean_dur"] = float(np.mean(runs)) if runs else 0.0
    f["closure_max_dur"] = float(np.max(runs)) if runs else 0.0
    f["long_closures"] = float(sum(r >= LONG_CLOSURE_SECONDS for r in runs))

    # --- eyes, relative to this person's own open-eye level ----------------
    if ear_open_level is not None and np.isfinite(ear_open_level) and ear_open_level > 0:
        rel = ear / ear_open_level
        f["ear_rel_mean"] = _nan_stat(np.mean, rel)
        f["ear_rel_p10"] = _nan_stat(lambda x: np.percentile(x, 10), rel)
        closed_rel = np.where(np.isnan(rel), np.nan, (rel < REL_CLOSED_RATIO).astype(float))
        f["closed_rel_frac"] = _nan_stat(np.mean, closed_rel)
        runs_rel = closure_runs(t, closed_rel)
        f["closure_rel_per_min"] = len(runs_rel) / minutes
        f["closure_rel_max_dur"] = float(np.max(runs_rel)) if runs_rel else 0.0
        f["long_closures_rel"] = float(sum(r >= LONG_CLOSURE_SECONDS for r in runs_rel))
    else:
        for name in REL_FEATURE_NAMES:
            f[name] = np.nan

    # --- mouth ------------------------------------------------------------
    mar = col("mar")
    f["mar_mean"] = _nan_stat(np.mean, mar)
    f["mar_std"] = _nan_stat(np.std, mar)
    f["mar_max"] = _nan_stat(np.max, mar)
    f["yawn_frac"] = _nan_stat(np.mean, col("yawning"))

    # --- head -------------------------------------------------------------
    for name in ("pitch", "yaw", "roll"):
        f[f"{name}_std"] = _nan_stat(np.std, col(f"head_{name}"))
    f["pitch_delta_mean"] = _nan_stat(np.mean, col("head_pitch_delta"))
    f["pitch_delta_max"] = _nan_stat(np.max, col("head_pitch_delta"))
    f["yaw_delta_absmean"] = _nan_stat(np.mean, np.abs(col("head_yaw_delta")))
    f["droop_frac"] = _nan_stat(np.mean, col("head_drooping"))
    f["turned_frac"] = _nan_stat(np.mean, col("head_turned"))

    dt = np.diff(t)
    dt = np.where(dt > 0, dt, np.nan)
    for name in ("pitch", "yaw", "roll"):
        speed = np.abs(np.diff(col(f"head_{name}"))) / dt          # degrees per second
        f[f"{name}_speed"] = _nan_stat(np.mean, speed)

    return f
