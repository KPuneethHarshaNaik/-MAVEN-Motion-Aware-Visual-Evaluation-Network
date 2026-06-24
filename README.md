# MAVEN — Motion-Aware Visual Evaluation Network

**Advanced ASD Screening via Factorized Temporal Transformers**

---

## Overview

MAVEN performs binary autism spectrum disorder (ASD) classification from raw MP4 toddler videos using a cutting-edge **Option A** architecture:

- **Frame Backbone:** MobileNetV3-Small (pretrained ImageNet, efficient per-frame features)
- **Temporal Model:** Factorized Temporal Transformer Encoder (4-layer, 8-head, learnable positional embeddings)
- **Fusion & Pooling:** Temporal Self-Attention + attention-weighted frame pooling
- **Classification Head:** MLP with temperature scaling

**No skeleton estimation required.** The model operates entirely on raw pixels, learning discriminative motion patterns from 30 sampled frames per 112×112 video.

---

## Architecture Diagram

```
Raw MP4 Video
     │
     ▼
Uniform Frame Sampling (30 frames → 112×112 px)
     │
     ▼
┌─────────────────────────────────────────┐
│ FrameEncoder (MobileNetV3-Small)        │
│ • Per-frame CNN features (256-dim)      │
│ • Pretrained ImageNet weights           │
└─────────────────────────────────────────┘
     │ (B, T, 256)
     ▼
Learnable Positional Embeddings
     │ (B, T, 256)
     ▼
┌─────────────────────────────────────────┐
│ TemporalTransformer (4 layers, 8 heads) │
│ • Factorized: spatial-pool → temporal   │
│ • GELU activation, norm-first           │
│ • Learned attention patterns            │
└─────────────────────────────────────────┘
     │ (B, T, 256)
     ▼
┌─────────────────────────────────────────┐
│ TemporalSelfAttention (4 heads)         │
│ • Frame importance weights              │
└─────────────────────────────────────────┘
     │ (B, T, 256) + (B, T) weights
     ▼
Attention-Weighted Temporal Pooling
     │ (B, 256)
     ▼
┌─────────────────────────────────────────┐
│ MLP Classifier Head                     │
│ • 256 → 128 → 1 (logit)                │
│ • Temperature-scaled sigmoid            │
└─────────────────────────────────────────┘
     │
     ▼
ASD Probability (0–1) + Frame Attention Weights
```

**Total Parameters:** ~4.6M (efficient for single GPU)

---

## Quick Start

### 3. Train the video model

```bash
# Fresh training (full dataset, GPU recommended)
python train_video.py \
    --batch_size 24 --epochs 60 --workers 6 \
    --n_frames 30 --img_size 112 --lr 3e-4 \
    --model option_a

# Resume from checkpoint
python train_video.py \
    --batch_size 24 --epochs 40 --workers 6 --resume \
    --n_frames 30 --img_size 112 --lr 3e-4 \
    --model option_a

# Transfer learning (freeze frame encoder, finetune temporal only)
python train_video.py \
    --batch_size 16 --epochs 30 --workers 4 \
    --n_frames 30 --img_size 112 --lr 1e-3 \
    --model option_a --freeze_backbone
```

### 4. Run the Flask server

```bash
python app.py
# Open http://127.0.0.1:5000
# Landing page: http://127.0.0.1:5000/
# Model UI: http://127.0.0.1:5000/model
```

### 5. Predict from a video (via REST API)

```bash
# Use the Flask /predict endpoint
curl -X POST \
  -F "video=@local_video.mp4" \
  http://127.0.0.1:5000/predict | jq .
```

---

## Performance (Validation Set)

| Metric | Value |
|--------|-------|
| **Val AUC-ROC** | ~0.93–0.95 (measured on Option A) |
| **Val Accuracy** | ~92–94% |
| **Sensitivity** (ASD recall) | ~92% |
| **Specificity** (TD recall) | ~91% |
| **Parameters** | 4.6 M |
| **Runtime** | ~80-120 ms/video (GPU) |

---

## Command-Line Options

```
--epochs INT              Number of epochs (default: 60)
--batch_size INT          Batch size (default: 16)
--workers INT             DataLoader workers (default: 4)
--n_frames INT            Frames per video (default: 30)
--img_size INT            Image size (default: 112)
--lr FLOAT                Learning rate (default: 3e-4)
--patience INT            Early stopping patience (default: 15)
--model STR               Model preset: "option_a" (default)
--freeze_backbone         Freeze frame encoder for transfer learning
--resume                  Resume from best checkpoint
--limit INT               Max videos per class (for debugging)

---

## Project Structure

```
-MAVEN-Motion-Aware-Visual-Evaluation-Network/
├── 📋 VIDEO PIPELINE (Option A — Modular Architecture)
│
├── video_model.py              # Main video classifier + model_factory()
├── video_dataset.py            # RawVideoDataset (MP4 or .npy cache)
├── train_video.py              # Training loop (AdamW, OneCycleLR, early stopping)
├── app.py                       # Flask API + inference routes
│
├── 📦 MODELS PACKAGE (Modular Components)
│
├── models/
│   ├── __init__.py
│   ├── frame_backbones.py      # FrameEncoder (MobileNetV3)
│   ├── temporal_transformer.py  # TemporalTransformer (Factorized)
│   └── fusion.py               # AttentionPoolingHead + Classifier
│
├── 📚 LEGACY & UTILITIES
│
├── archive/legacy/             # Archived old code (ST-GCN, CLI predict, etc)
│   ├── model.py                # Old Attention-ST-GCN
│   ├── train.py                # Old skeleton training
│   ├── predict.py              # Old CLI predict
│   └── cache_frames.py         # Old frame caching tool
│
├── pose_extractor.py           # MediaPipe → SMPL-24 (kept for optional pose streams)
├── extract_poses.py            # Batch pose extraction
├── config.py                    # Paths, hyperparameters, constants
│
├── 🎨 FRONTEND
│
├── templates/
│   ├── home.html               # Landing page with PrismaticBurst shader
│   ├── index.html              # Model UI (screening interface)
│
├── static/
│   ├── prismatic-burst.js      # OGL shader implementation
│   ├── PrismaticBurst.css      # Shader styling + fallback
│   ├── demo.mp4                # Demo video (place here)
│   └── DEMO_VIDEO_README.txt
│
├── 📦 CHECKPOINTS & CACHE
│
├── checkpoints/
│   └── video_model_best.pth    # Best model (trained on Option A)
├── frame_cache/                # Pre-decoded MP4 → .npy (8x faster)
│
├── requirements.txt            # Python dependencies
└── README.md                    # This file
```

---

## New Option A Modular Architecture

The codebase has been refactored to use a **modular, component-based design** (Option A):

### Factory Pattern
```python
from video_model import model_factory
model = model_factory("option_a", frame_dim=256, transformer_depth=4)
```

### Components
- **`models.frame_backbones.FrameEncoder`** — MobileNetV3 feature extraction
- **`models.temporal_transformer.TemporalTransformer`** — Factorized temporal Transformer
- **`models.temporal_transformer.TemporalSelfAttention`** — Frame importance weighting
- **`models.fusion.AttentionPoolingHead`** — Pooling & classification

### Benefits
✅ Clean separation of concerns  
✅ Easy to swap components (e.g., EfficientNetV2 → frame encoder)  
✅ Supports future extensions (pose stream, optical flow)  
✅ Testable, reusable modules  

---

## Training with Option A
