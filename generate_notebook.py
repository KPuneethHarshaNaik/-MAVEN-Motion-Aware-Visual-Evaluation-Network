import json
import os
import re

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def strip_local_imports(code):
    lines = code.split('\n')
    out = []
    for line in lines:
        if line.startswith('from models.') or line.startswith('from video_') or line.startswith('import video_'):
            continue
        out.append(line)
    return '\n'.join(out)

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in text.split('\n')]
    })

def add_code(text):
    notebook['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in text.split('\n')]
    })

add_md("# MAVEN — Standalone Colab Version\nThis notebook contains the entire MAVEN architecture and training loop.")
add_code("!nvidia-smi")
add_code("from google.colab import drive\ndrive.mount('/content/drive')")

add_md("## 1. Dataset Discovery\nThis cell automatically finds the `autism_data_anonymized` dataset on your Google Drive.\nIt searches for the correct folder structure (`training_set/ASD`, `testing_set/ASD`).\nIf it can't find it, you can set the path manually.")

dataset_discovery_code = r'''import os
from pathlib import Path

def find_dataset(drive_root="/content/drive"):
    """
    Walk the mounted Google Drive to find the autism_data_anonymized folder
    with the expected structure: training_set/{ASD,TD} (testing_set is optional).
    """
    target_name = "autism_data_anonymized"
    candidates = []

    print(f"Searching for '{target_name}' on Google Drive...")
    for root, dirs, files in os.walk(drive_root):
        # Skip .Trash and hidden folders for speed
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if target_name in dirs:
            candidate = Path(root) / target_name
            # Only require training_set (testing_set is optional)
            has_train = (candidate / "training_set" / "ASD").is_dir()
            if has_train:
                candidates.append(candidate)

    if not candidates:
        # Fallback: maybe the user put training_set/ directly
        # in a folder without the autism_data_anonymized wrapper
        for root, dirs, files in os.walk(drive_root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            if "training_set" in dirs:
                candidate = Path(root)
                if (candidate / "training_set" / "ASD").is_dir():
                    candidates.append(candidate)

    if not candidates:
        print("\n⚠️  Could not find the dataset automatically!")
        print("   Expected folder structure:")
        print("     .../training_set/ASD/*.mp4")
        print("     .../training_set/TD/*.mp4")
        print("     .../testing_set/ASD/*.mp4  (optional)")
        print("     .../testing_set/TD/*.mp4   (optional)")
        print("\n   Set DATA_ROOT manually in the next cell.")
        return None

    # Pick the first valid match
    found = candidates[0]
    if len(candidates) > 1:
        print(f"Found {len(candidates)} matches — using the first one.")
        for c in candidates:
            print(f"  • {c}")

    # Print dataset summary
    print(f"\n✅ Dataset found at:\n   {found}\n")
    for split in ["training_set", "testing_set"]:
        split_dir = found / split
        if not split_dir.exists():
            continue
        for cls in ["ASD", "TD"]:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                vids = [f for f in cls_dir.iterdir()
                        if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
                print(f"   {split}/{cls}: {len(vids)} videos")

    return str(found)


DATA_ROOT = find_dataset()

# ── MANUAL OVERRIDE (uncomment and edit if auto-discovery fails) ──
# DATA_ROOT = "/content/drive/My Drive/path/to/autism_data_anonymized"

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Dataset not found! Set DATA_ROOT manually in this cell."
    )

print(f"\n📂 DATA_ROOT = {DATA_ROOT}")
'''

add_code(dataset_discovery_code)

imports = """import os, sys, time, json, random, argparse, warnings
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import roc_auc_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files"""

add_md("## 2. Imports")
add_code(imports)

add_md("## 3. Models: Frame Backbone")
code = read_file('models/frame_backbones.py')
add_code(strip_local_imports(code))

add_md("## 4. Models: Temporal Transformer")
code = read_file('models/temporal_transformer.py')
add_code(strip_local_imports(code))

add_md("## 5. Models: Fusion Head")
code = read_file('models/fusion.py')
add_code(strip_local_imports(code))

add_md("## 6. Main Video Classifier")
code = read_file('video_model.py')
code = code.split('if __name__ == "__main__":')[0]
add_code(strip_local_imports(code))

add_md("## 7. Dataset and Augmentation")
code = read_file('video_dataset.py')
code = code.split('if __name__ == "__main__":')[0]
add_code(strip_local_imports(code))

add_md("## 8. Training Setup")
code = read_file('train_video.py')
# Remove parse_args and main block, replace with a config class
code_parts = code.split('def parse_args():')
before_parse = code_parts[0]
after_parse = code.split('def compute_metrics')[1]

config_code = """class Args:
    batch_size = 16
    epochs = 60
    workers = 0  # Colab /dev/shm is too small for multiprocessing; 0 = main-process loading
    n_frames = 30
    img_size = 112
    lr = 3e-4
    patience = 15
    limit = None  # None = use all videos; build_video_loaders auto-splits 80/20 train/val
    resume = False
    model = "option_a"
    freeze_backbone = False
    label_smoothing = 0.1
    data_root = DATA_ROOT  # Auto-discovered from Google Drive

args = Args()
"""
combined_train = strip_local_imports(before_parse) + "\n\n" + config_code + "\ndef compute_metrics" + strip_local_imports(after_parse).split('if __name__ == "__main__":')[0]
add_code(combined_train)

add_md("## 9. Run Training\nRun this cell to train the model on the dataset.")
add_code("train(args)")

add_md("## 10. Inference Showcase\nUpload a video to test the model dynamically!")
inference_code = """def showcase_inference(model_path, video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    model = model_factory("option_a").to(device)
    ck = torch.load(model_path, map_location="cpu")
    
    # Load state dict (handling potential mismatch in key names based on how it was saved)
    state_dict = ck.get("model_state") or ck.get("model_state_dict") or ck
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Extracting frames from {video_path}...")
    raw_frames = _sample_frames(video_path, args.n_frames, strategy="uniform")
    tfm = VideoTransform(img_size=args.img_size)
    video_tensor = tfm(raw_frames).unsqueeze(0).to(device)

    with torch.no_grad():
        result = model.predict(video_tensor)

    print("-" * 40)
    print(f"Prediction : {result['label_name']}")
    print(f"Confidence : {result['confidence']*100:.2f}%")
    print("-" * 40)

    # Plot top frames
    top_indices = result["top_frames"]
    fig, axes = plt.subplots(1, len(top_indices), figsize=(15, 5))
    if len(top_indices) == 1: axes = [axes]
    for ax, idx in zip(axes, top_indices):
        frame_rgb = cv2.cvtColor(raw_frames[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {idx} (Weight: {result['frame_weights'][idx]:.2f})")
        ax.axis("off")
    plt.show()

# --- Run Inference ---
print("Upload a video file (.mp4, .avi) to test:")
uploaded = files.upload()
if uploaded:
    video_path = list(uploaded.keys())[0]
    # Point this to your actual trained checkpoint!
    showcase_inference(str(BEST_MODEL), video_path)
"""
add_code(inference_code)

with open('MAVEN_Standalone.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully: MAVEN_Standalone.ipynb")
