"""
cache_frames.py
===============
Pre-decodes all MP4 videos into numpy frame arrays and saves them as .npy
files. Training then reads .npy (instant memory-map) instead of decoding
MP4s on-the-fly — giving ~3-4x faster data loading.

Cache layout mirrors the dataset:
    frame_cache/
        training_set/ASD/<video_name>.npy   shape: (N_FRAMES, H, W, 3) uint8
        training_set/TD/<video_name>.npy
        testing_set/ASD/<video_name>.npy
        testing_set/TD/<video_name>.npy

Usage:
    python cache_frames.py                        # default: 16 frames, 96px
    python cache_frames.py --n_frames 16 --img_size 96 --workers 6
    python cache_frames.py --overwrite            # re-cache everything
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
AUTISM_DATA_ROOT = Path("/home/puneeth/Desktop/git hub/autism_data_anonymized")
CACHE_ROOT       = Path("/home/puneeth/Desktop/git hub/ASD-Detection-Model/frame_cache")
SPLITS           = ["training_set", "testing_set"]
CLASSES          = ["ASD", "TD"]


# ─────────────────────────────────────────────────────────────────────────────
def _sample_and_resize(video_path: str, n_frames: int, img_size: int) -> np.ndarray:
    """
    Sample n_frames uniformly from video, resize to img_size×img_size.
    Returns uint8 array of shape (n_frames, img_size, img_size, 3) in RGB.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    frames  = []
    last_ok = None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)
            last_ok = resized
            frames.append(resized)
        elif last_ok is not None:
            frames.append(last_ok)
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    cap.release()
    return np.stack(frames, axis=0)   # (T, H, W, 3) uint8


# ─────────────────────────────────────────────────────────────────────────────
def _cache_one(args_tuple):
    """Worker function — processes a single video file."""
    video_path, npy_path, n_frames, img_size, overwrite = args_tuple

    if not overwrite and npy_path.exists():
        return "skip"

    try:
        frames = _sample_and_resize(str(video_path), n_frames, img_size)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), frames)
        return "ok"
    except Exception as e:
        return f"error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
def build_job_list(n_frames: int, img_size: int, overwrite: bool):
    """Collect all (video_path, npy_path, ...) tuples to process."""
    jobs = []
    for split in SPLITS:
        for cls in CLASSES:
            src_dir   = AUTISM_DATA_ROOT / split / cls
            dst_dir   = CACHE_ROOT / split / cls
            if not src_dir.exists():
                continue
            for mp4 in sorted(src_dir.iterdir()):
                if mp4.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                    continue
                npy_path = dst_dir / (mp4.stem + ".npy")
                jobs.append((mp4, npy_path, n_frames, img_size, overwrite))
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames",  type=int, default=16)
    parser.add_argument("--img_size",  type=int, default=96)
    parser.add_argument("--workers",   type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = build_job_list(args.n_frames, args.img_size, args.overwrite)

    pending = [j for j in jobs if args.overwrite or not j[1].exists()]
    skipped = len(jobs) - len(pending)

    print(f"Frame cache  →  {CACHE_ROOT}")
    print(f"Settings     :  {args.n_frames} frames  ×  {args.img_size}px")
    print(f"Total videos :  {len(jobs)}")
    print(f"Already done :  {skipped}  (use --overwrite to redo)")
    print(f"To process   :  {len(pending)}")

    if not pending:
        print("Nothing to do — cache is complete.")
        return

    ok_count  = 0
    err_count = 0

    with mp.Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_cache_one, pending),
            total=len(pending),
            desc="Caching",
            unit="video",
        ):
            if result == "ok":
                ok_count  += 1
            elif result == "skip":
                skipped   += 1
            else:
                err_count += 1

    print(f"\nDone!  cached={ok_count}  skipped={skipped}  errors={err_count}")

    # Quick size report
    total_bytes = sum(f.stat().st_size for f in CACHE_ROOT.rglob("*.npy"))
    print(f"Cache size   :  {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
