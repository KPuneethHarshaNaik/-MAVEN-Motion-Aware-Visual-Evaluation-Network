"""
Pose Extractor — MediaPipe Pose → SMPL-24 Skeleton

Converts a raw MP4 video into a normalised SMPL-24 2D skeleton sequence.
The resulting (T, 24, 2) array can be fed directly into the ASD classifier.

MediaPipe detects 33 landmarks per frame.  We map them to the 24-joint SMPL
convention used by the MMASD ROMP-2D dataset so that both data sources share
a common representation.

Coordinate system after extraction:
  • Centred on the Pelvis (joint 0) → translation-invariant
  • Uniform scale via torso-height normalisation → scale-invariant

Uses MediaPipe Tasks API (v0.10+) with automatic model-file download.
Model is cached at:  ASD-Detection-Model/pose_landmarker_full.task
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from config import MAX_FRAMES, NUM_JOINTS, MP_TO_SMPL24

# Reference joint indices for scale normalisation (SMPL-24)
_SCALE_REF_JOINTS = [16, 17, 1, 2]   # L/R Shoulder + L/R Hip

# ─── Model download ───────────────────────────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pose_landmarker_full.task")


def _ensure_model():
    """Download the MediaPipe Pose Landmarker model if not already cached."""
    if not os.path.exists(_MODEL_PATH):
        print(f"  Downloading MediaPipe Pose Landmarker model …")
        print(f"  → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("  Download complete.")


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_skeleton_from_video(
    video_path: str,
    max_frames: int = MAX_FRAMES,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float  = 0.5,
) -> np.ndarray:
    """
    Extract a fixed-length SMPL-24 2D skeleton sequence from a raw MP4.

    Args:
        video_path: Path to the .mp4 (or any OpenCV-readable video).
        max_frames: Output temporal length.
        min_detection_confidence: Pose detection confidence threshold.
        min_tracking_confidence:  Pose tracking confidence threshold.

    Returns:
        skeleton: (2, max_frames, 24) float32 array, centred & normalised.
    """
    _ensure_model()

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    BaseOptions  = mp_python.BaseOptions
    PoseLandmarker        = mp_vision.PoseLandmarker
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    RunningMode           = mp_vision.RunningMode

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    raw_frames = []
    frame_idx  = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp  = int(frame_idx / fps * 1000)
            result     = landmarker.detect_for_video(mp_image, timestamp)
            frame_idx += 1

            if result.pose_landmarks:
                lm  = result.pose_landmarks[0]
                pts = np.array([[l.x * w, l.y * h] for l in lm],
                               dtype=np.float32)   # (33, 2)
                smpl24 = _mediapipe_to_smpl24(pts)
                raw_frames.append(smpl24)
            else:
                # Carry-forward last good pose or zeros
                if raw_frames:
                    raw_frames.append(raw_frames[-1].copy())
                else:
                    raw_frames.append(np.zeros((NUM_JOINTS, 2), dtype=np.float32))

    cap.release()

    if not raw_frames:
        return np.zeros((IN_CHANNELS, max_frames, NUM_JOINTS), dtype=np.float32)

    seq = np.stack(raw_frames, axis=0)   # (T, 24, 2)
    seq = _normalise_sequence(seq)
    seq = _fix_length(seq, max_frames)   # (max_frames, 24, 2)
    seq = seq.transpose(2, 0, 1)         # (C, T, V)  = (2, max_frames, 24)
    return seq.astype(np.float32)


def skeleton_from_npz(npz_path: str, person_index: int = 0,
                      max_frames: int = MAX_FRAMES) -> np.ndarray:
    """
    Load a pre-extracted ROMP-2D NPZ file (MMASD format).

    Args:
        npz_path:     Path to the folder containing per-frame *.npz files,
                      OR path to a single *.npz file.
        person_index: Which detected person to use (0 = most prominent).
        max_frames:   Output temporal length.

    Returns:
        skeleton: (2, max_frames, 24) float32 array.
    """
    import glob, os
    if os.path.isdir(npz_path):
        files = sorted(glob.glob(os.path.join(npz_path, "*.npz")))
    else:
        files = [npz_path]

    frames = []
    for f in files:
        data = np.load(f)
        coords = data["coordinates"]          # (people, 24, 2)
        idx = person_index if coords.shape[0] > person_index else 0
        frames.append(coords[idx])            # (24, 2)

    if not frames:
        return np.zeros((2, max_frames, NUM_JOINTS), dtype=np.float32)

    seq = np.stack(frames, axis=0)            # (T, 24, 2)
    seq = _normalise_sequence(seq)
    seq = _fix_length(seq, max_frames)        # (max_frames, 24, 2)
    seq = seq.transpose(2, 0, 1)              # (2, T, 24)
    return seq.astype(np.float32)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _mediapipe_to_smpl24(pts: np.ndarray) -> np.ndarray:
    """
    Map MediaPipe 33-landmark array  →  SMPL-24 joint array.

    Args:
        pts: (33, 2) — pixel-space x, y for each MediaPipe landmark.
    Returns:
        smpl24: (24, 2) float32.
    """
    smpl24 = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    for joint_idx, mp_indices in MP_TO_SMPL24.items():
        smpl24[joint_idx] = pts[mp_indices].mean(axis=0)
    return smpl24


def _normalise_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Centre skeleton on Pelvis (joint 0) and normalise scale using the
    average shoulder-to-hip torso half-height.

    Args:
        seq: (T, 24, 2)
    Returns:
        Normalised (T, 24, 2).
    """
    # Centre on Pelvis joint
    pelvis = seq[:, 0:1, :]           # (T, 1, 2)
    seq = seq - pelvis

    # Scale: average inter-joint distance across shoulder and hip spread
    scale_pts = seq[:, _SCALE_REF_JOINTS, :]   # (T, 4, 2)
    # Torso height proxy: distance between mid-shoulders and mid-hips
    mid_shoulder = scale_pts[:, :2, :].mean(axis=1)  # (T, 2)
    mid_hip      = scale_pts[:, 2:, :].mean(axis=1)  # (T, 2)
    heights = np.linalg.norm(mid_shoulder - mid_hip, axis=1)  # (T,)
    scale = heights.mean()
    if scale > 1e-6:
        seq = seq / scale

    return seq.astype(np.float32)


def _fix_length(seq: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Uniformly sample or zero-pad a sequence to exactly max_frames.
    seq: (T, 24, 2)
    """
    T = seq.shape[0]
    if T == 0:
        return np.zeros((max_frames, NUM_JOINTS, 2), dtype=np.float32)
    if T >= max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        return seq[indices]
    # Zero-pad
    pad = np.zeros((max_frames - T, NUM_JOINTS, 2), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


# ─── Constants re-exported for convenience ────────────────────────────────────
IN_CHANNELS = 2   # x, y


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pose_extractor.py <path_to_video.mp4>")
        sys.exit(1)

    path = sys.argv[1]
    skel = extract_skeleton_from_video(path)
    print(f"Output skeleton shape: {skel.shape}")  # Expected: (2, 120, 24)
    print(f"Value range: [{skel.min():.3f}, {skel.max():.3f}]")
