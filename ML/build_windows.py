"""
build_windows.py — Sprint 4, step 2
------------------------------------
Reads the per-frame CSVs written by extract_features.py, cuts each video
into overlapping time windows, computes one row of features per window
(window_features.py), and writes everything to one windows.csv.

Labels: alert (0) -> 0, drowsy (10) -> 1. Any low-vigilant (5) videos
are skipped.

Usage:
    python build_windows.py --csv-dir alert_csvs drowsy_csvs --out windows.csv

Defaults: 30-second windows every 15 seconds, and the first 45 seconds of
every video are dropped (the PERCLOS / blink window is still filling up
there, so those numbers aren't reliable yet). Videos shorter than
warmup + window produce no windows — for a quick test on short clips,
use --warmup 0.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from window_features import window_features

LABEL_MAP = {0: 0, 10: 1}          # dataset label -> class (0 = alert, 1 = drowsy)
OPEN_EAR_PERCENTILE = 90           # a person's "eyes open" EAR = this percentile over their video


def windows_for_video(df, win, stride, warmup, meta):
    df = df.sort_values("t_sec").reset_index(drop=True)
    t = df["t_sec"].to_numpy(dtype=float)
    rows = []
    if len(t) == 0:
        return rows
    start = max(warmup, float(t[0]))
    end = float(t[-1])
    s = start
    while s + win <= end + 1e-6:
        i0 = np.searchsorted(t, s, side="left")
        i1 = np.searchsorted(t, s + win, side="left")
        feats = window_features(df.iloc[i0:i1], win, ear_open_level=meta.get("ear_open_level"))
        if feats is not None:
            rows.append({**meta, "window_start": round(s, 2), **feats})
        s += stride
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build windowed feature table from per-frame CSVs")
    ap.add_argument("--csv-dir", nargs="+", required=True,
                    help="one or more folders of per-frame CSVs (searched recursively)")
    ap.add_argument("--out", default="windows.csv")
    ap.add_argument("--win", type=float, default=30.0, help="window length, seconds")
    ap.add_argument("--stride", type=float, default=15.0, help="step between windows, seconds")
    ap.add_argument("--warmup", type=float, default=45.0,
                    help="drop this many seconds from the start of every video")
    args = ap.parse_args()

    files = []
    for d in args.csv_dir:
        files += sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No CSV files found in: {args.csv_dir}")
    print(f"Found {len(files)} CSV files")

    all_rows = []
    skipped = 0
    for n, path in enumerate(files, 1):
        df = pd.read_csv(path, dtype={"fold": str, "subject": str, "video_id": str})
        if "label" not in df.columns or df.empty:
            print(f"  skip (no label column / empty): {os.path.basename(path)}")
            skipped += 1
            continue
        raw_label = int(df["label"].iloc[0])
        if raw_label not in LABEL_MAP:
            skipped += 1
            continue
        fold = df["fold"].iloc[0] if pd.notna(df["fold"].iloc[0]) else "X"
        subject = df["subject"].iloc[0]
        ear_vals = pd.to_numeric(df["ear"], errors="coerce").to_numpy(dtype=float)
        ear_vals = ear_vals[np.isfinite(ear_vals)]
        open_level = float(np.percentile(ear_vals, OPEN_EAR_PERCENTILE)) if ear_vals.size else np.nan
        meta = {
            "video_id": df["video_id"].iloc[0],
            "subject": subject,
            "fold": fold,
            "group": f"{fold}_{subject}",      # one person = one group (never split across train/test)
            "label": LABEL_MAP[raw_label],
            "ear_open_level": round(open_level, 4),   # per-video; used for person-relative features
        }
        rows = windows_for_video(df, args.win, args.stride, args.warmup, meta)
        all_rows += rows
        if n % 20 == 0 or n == len(files):
            print(f"  processed {n}/{len(files)} files, {len(all_rows)} windows so far")

    if not all_rows:
        raise SystemExit("No windows produced. Videos may be too short for the warmup + window "
                         "length — try --warmup 0 for short test clips.")

    out = pd.DataFrame(all_rows)
    out.to_csv(args.out, index=False)
    with open(args.out + ".meta.json", "w") as f:
        json.dump({"win_seconds": args.win, "stride_seconds": args.stride,
                   "warmup_seconds": args.warmup,
                   "open_ear_percentile": OPEN_EAR_PERCENTILE}, f, indent=2)

    print(f"\nWrote {len(out)} windows from {out['video_id'].nunique()} videos "
          f"({out['group'].nunique()} people) -> {args.out}")
    print("Windows per class (0 = alert, 1 = drowsy):")
    print(out["label"].value_counts().sort_index().to_string())
    per_person = out.groupby("group")["label"].nunique()
    if (per_person < 2).any():
        print(f"Note: {int((per_person < 2).sum())} people have only one class — "
              f"that's fine for training, just worth knowing.")
    if skipped:
        print(f"{skipped} files skipped (low-vigilant label or unreadable)")


if __name__ == "__main__":
    main()
