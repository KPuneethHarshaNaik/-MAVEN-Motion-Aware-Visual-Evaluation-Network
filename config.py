"""
Central configuration for the ASD Detection Model.

Architecture: Attention-enhanced ST-GCN (Spatial-Temporal Graph Convolutional
Network) trained for binary ASD vs TD classification from 2D skeleton sequences.

Two-stage training strategy:
  Stage 1 – Pre-train on MMASD ROMP-2D skeleton data (690 ASD / 579 TD sequences)
  Stage 2 – Fine-tune on autism_data_anonymized (4840 ASD / 4840 TD videos,
             after MediaPipe pose extraction)
"""

import os
import torch

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WORKSPACE    = os.path.dirname(BASE_DIR)   # .../git hub/

MMASD_SKEL_2D = os.path.join(
    WORKSPACE,
    "MMASD_DATASET",
    "drive-download-20260226T093336Z-1-002",
    "2D skeleton",
    "ROMP_2D_Coordinates",
)

AUTISM_DATA_ROOT = os.path.join(WORKSPACE, "autism_data_anonymized")

# Folder where pre-extracted MediaPipe NPZ files are cached (created by extract_poses.py)
POSE_CACHE_DIR = os.path.join(BASE_DIR, "pose_cache_autism_data")
os.makedirs(POSE_CACHE_DIR, exist_ok=True)

CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH    = os.path.join(CHECKPOINT_DIR, "best_asd_model.pth")
STAGE1_MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "stage1_pretrain.pth")
STAGE2_MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "stage2_finetune.pth")

# ─── SKELETON SETTINGS ────────────────────────────────────────────────────────
# We use the SMPL-24 joint convention (same as ROMP 2D output)
NUM_JOINTS   = 24   # SMPL-24 joints
IN_CHANNELS  = 2    # x, y coordinates
MAX_FRAMES   = 120  # All sequences padded / uniformly sampled to this length

# SMPL-24 joint names (indices match ROMP 2D coordinates)
JOINT_NAMES = [
    "Pelvis",      # 0
    "L_Hip",       # 1
    "R_Hip",       # 2
    "Spine1",      # 3
    "L_Knee",      # 4
    "R_Knee",      # 5
    "Spine2",      # 6
    "L_Ankle",     # 7
    "R_Ankle",     # 8
    "Spine3",      # 9
    "L_Foot",      # 10
    "R_Foot",      # 11
    "Neck",        # 12
    "L_Collar",    # 13
    "R_Collar",    # 14
    "Head",        # 15
    "L_Shoulder",  # 16
    "R_Shoulder",  # 17
    "L_Elbow",     # 18
    "R_Elbow",     # 19
    "L_Wrist",     # 20
    "R_Wrist",     # 21
    "L_Hand",      # 22
    "R_Hand",      # 23
]

# SMPL-24 body graph edges (for ST-GCN adjacency matrix)
SMPL24_EDGES = [
    (0, 1), (0, 2), (0, 3),    # Pelvis → L_Hip, R_Hip, Spine1
    (1, 4), (2, 5),             # Hips → Knees
    (3, 6),                     # Spine1 → Spine2
    (4, 7), (5, 8),             # Knees → Ankles
    (6, 9),                     # Spine2 → Spine3
    (7, 10), (8, 11),           # Ankles → Feet
    (9, 12), (9, 13), (9, 14), # Spine3 → Neck, Collars
    (12, 15),                   # Neck → Head
    (13, 16), (14, 17),         # Collars → Shoulders
    (16, 18), (17, 19),         # Shoulders → Elbows
    (18, 20), (19, 21),         # Elbows → Wrists
    (20, 22), (21, 23),         # Wrists → Hands
]

# ─── MODEL HYPERPARAMETERS ────────────────────────────────────────────────────
HIDDEN_CHANNELS       = [64, 64, 128, 128, 256, 256]
DROPOUT               = 0.4
TEMPORAL_KERNEL_SIZE  = 9

# ─── TRAINING SETTINGS ────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 80   # Per-stage maximum
PATIENCE      = 15   # Early-stopping patience (epochs without val-improvement)
WARMUP_EPOCHS = 5    # Cosine-LR warmup epochs

# Stage 1 (MMASD only) and Stage 2 (combined) max epochs
STAGE1_EPOCHS = 60
STAGE2_EPOCHS = 80

# Split ratios for MMASD data
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10

# Max autism_data videos to use — None = all available
MAX_AUTISM_VIDEOS_PER_CLASS = None

SEED = 42

# ─── DEVICE ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── LABEL ENCODING ───────────────────────────────────────────────────────────
# Folder-name suffixes in MMASD:  _y = ASD,  _n = TD,  _i = skip
MMASD_LABEL_MAP = {"y": 1, "n": 0}  # 1=ASD, 0=TD

CLASS_NAMES = {0: "TD (Typically Developing)", 1: "ASD (Autism Spectrum Disorder)"}

# ─── MEDIAPIPE → SMPL-24 LANDMARK MAPPING ─────────────────────────────────────
# Maps each SMPL-24 joint to one or more MediaPipe Pose landmark indices.
# When multiple indices are listed the result is their average.
MP_TO_SMPL24 = {
    0:  [23, 24],       # Pelvis    ← avg(L_hip, R_hip)
    1:  [23],           # L_Hip
    2:  [24],           # R_Hip
    3:  [11, 12, 23, 24], # Spine1  ← midpoint of shoulder+hip girdle
    4:  [25],           # L_Knee
    5:  [26],           # R_Knee
    6:  [11, 12, 23, 24], # Spine2  (same as Spine1 proxy)
    7:  [27],           # L_Ankle
    8:  [28],           # R_Ankle
    9:  [11, 12],       # Spine3   ← avg(shoulders)
    10: [31],           # L_Foot
    11: [32],           # R_Foot
    12: [11, 12],       # Neck     ← avg(shoulders)
    13: [11],           # L_Collar ← L_Shoulder proxy
    14: [12],           # R_Collar ← R_Shoulder proxy
    15: [0],            # Head     ← Nose
    16: [11],           # L_Shoulder
    17: [12],           # R_Shoulder
    18: [13],           # L_Elbow
    19: [14],           # R_Elbow
    20: [15],           # L_Wrist
    21: [16],           # R_Wrist
    22: [17, 19, 21],   # L_Hand   ← avg(L_pinky, L_index, L_thumb)
    23: [18, 20, 22],   # R_Hand   ← avg(R_pinky, R_index, R_thumb)
}
