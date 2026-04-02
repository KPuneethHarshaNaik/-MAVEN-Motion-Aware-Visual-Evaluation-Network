"""
Batch Pose Extraction ΓÇö autism_data_anonymized ΓåÆ NPZ cache

Processes all .mp4 videos in autism_data_anonymized/{training_set,testing_set}/
{ASD,TD}/ using MediaPipe Pose, maps to SMPL-24 joints and saves one compressed
NPZ file per video to POSE_CACHE_DIR.

The cached files are read by PreExtractedDataset during Stage 2 training.

Usage:
    python extract_poses.py
    python extract_poses.py --workers 4    # parallel workers (default=4)
    python extract_poses.py --limit 500    # max videos per class/split
    python extract_poses.py --overwrite    # re-extract even if NPZ exists

Output structure:
    ASD-Detection-Model/pose_cache_autism_data/
        training_set/
            ASD/  <video_stem>.npz
            TD/   <video_stem>.npz
        testing_set/
            ASD/  <video_stem>.npz
            TD/   <video_stem>.npz

Each .npz contains:
    skeleton : float32 array of shape (2, MAX_FRAMES, 24)  ΓÇö (C, T, V)
"""

import os
import sys
import argparse
import glob
import time
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import AUTISM_DATA_ROOT, POSE_CACHE_DIR, MAX_FRAMES
from pose_extractor import extract_skeleton_from_video


# ΓöÇΓöÇΓöÇ Worker function (runs in a subprocess) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def _worker(args):
    video_path, out_path, overwrite = args
    if not overwrite and os.path.exists(out_path):
        return "skip", video_path

    try:
        skel = extract_skeleton_from_video(video_path, max_frames=MAX_FRAMES)
        np.savez_compressed(out_path, skeleton=skel)
        return "ok", video_path
    except Exception as e:
        return "error", f"{video_path}: {e}"


# ΓöÇΓöÇΓöÇ Main ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe poses from videos")
    parser.add_argument("--workers",   type=int, default=min(4, cpu_count()),
                        help="Parallel worker processes (default=4)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max videos to process per split/class")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if .npz already exists")
    args = parser.parse_args()

    print("=" * 60)
    print("  MediaPipe Pose Extraction ΓÇö autism_data_anonymized")
    print(f"  Source: {AUTISM_DATA_ROOT}")
    print(f"  Cache:  {POSE_CACHE_DIR}")
    print(f"  Workers: {args.workers}  |  Limit: {args.limit}")
    print("=" * 60)

    total_ok   = 0
    total_skip = 0
    total_err  = 0
    t_start    = time.time()

    for split in ["training_set", "testing_set"]:
        for class_name in ["ASD", "TD"]:
            src_dir = os.path.join(AUTISM_DATA_ROOT, split, class_name)
            dst_dir = os.path.join(POSE_CACHE_DIR,   split, class_name)
            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.exists(src_dir):
                print(f"  [SKIP] {src_dir} not found.")
                continue

            videos = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
            if args.limit:
                videos = videos[:args.limit]

            tasks = [
                (v, os.path.join(dst_dir, os.path.splitext(os.path.basename(v))[0] + ".npz"),
                 args.overwrite)
                for v in videos
            ]

            print(f"\n  {split}/{class_name}: {len(tasks)} videos ΓÇª")

            with Pool(processes=args.workers) as pool:
                # Process in chunks and report progress
                done = 0
                for status, info in pool.imap_unordered(_worker, tasks, chunksize=4):
                    done += 1
                    if status == "ok":
                        total_ok += 1
                    elif status == "skip":
                        total_skip += 1
                    else:
                        total_err += 1
                        print(f"    [ERROR] {info}")
                    if done % 100 == 0 or done == len(tasks):
                        elapsed = time.time() - t_start
                        rate    = done / elapsed if elapsed > 0 else 0
                        print(f"    {done}/{len(tasks)}  "
                              f"({rate:.1f} vid/s)  "
                              f"ok={total_ok}  skip={total_skip}  err={total_err}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Extraction complete in {elapsed/60:.1f} min")
    print(f"  OK={total_ok}  Skipped={total_skip}  Errors={total_err}")
    print(f"  Cache directory: {POSE_CACHE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
