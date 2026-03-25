"""
Configuration file for ASD Skeleton Action Recognition Model.
Adjust these parameters to tune the model for your hardware and dataset.
"""

import os

# ============================================================
# PATHS — UPDATE THESE TO MATCH YOUR DOWNLOADED DATASET
# ============================================================
# Path to the downloaded MMASD dataset root folder
DATASET_ROOT = r"C:\git hub\MMASD_DATASET"

# Inner folder created by Google Drive download (adjust if yours differs)
_DRIVE_FOLDER = "drive-download-20260226T093336Z-1-002"

# Which 2D skeleton source to use: "openpose" or "romp"
SKELETON_2D_SUBTYPE = "openpose"

# Resolved skeleton directory paths (do not edit directly)
SKELETON_DIR_2D_OPENPOSE = os.path.join(
    DATASET_ROOT, _DRIVE_FOLDER, "2D skeleton", "2D_openpose", "output"
)
SKELETON_DIR_2D_ROMP = os.path.join(
    DATASET_ROOT, _DRIVE_FOLDER, "2D skeleton", "ROMP_2D_Coordinates"
)
SKELETON_DIR_3D = os.path.join(
    DATASET_ROOT, _DRIVE_FOLDER, "3D skeleton", "ROMP_3D_Coordinates"
)

# Path to save trained model weights
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ============================================================
# DATA SETTINGS
# ============================================================
# Which skeleton modality to use: "2d" (OpenPose, 25 joints) or "3d" (ROMP, 71 joints)
SKELETON_MODE = "2d"

# Number of keypoints per skeleton
NUM_JOINTS_2D = 25
NUM_JOINTS_3D = 71

# Input channels per joint (x, y for 2D; x, y, z for 3D)
IN_CHANNELS_2D = 2   # We drop the confidence score and just use x, y
IN_CHANNELS_3D = 3

# Fixed number of frames for each sample (sequences are padded/truncated to this)
MAX_FRAMES = 150

# Number of activity classes in the MMASD dataset (11 activities)
NUM_CLASSES = 11

# Activity labels mapping
ACTIVITY_LABELS = {
    "as": 0,   # Arm Swing
    "bs": 1,   # Body Swing
    "ce": 2,   # Chest Expansion
    "sq": 3,   # Squat
    "dr": 4,   # Drumming
    "mfs": 5,  # Maracas Forward Shaking
    "ms": 6,   # Maracas Shaking
    "sac": 7,  # Sing and Clap
    "fg": 8,   # Frog Pose
    "tr": 9,   # Tree Pose
    "tw": 10,  # Twist Pose
}

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
# Number of ST-GCN blocks (layers)
NUM_STGCN_BLOCKS = 6

# Hidden channel dimensions for each ST-GCN block
HIDDEN_CHANNELS = [64, 64, 128, 128, 256, 256]

# Dropout rate
DROPOUT = 0.3

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42
