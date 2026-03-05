# ASD Detection Model

Binary Autism Spectrum Disorder (ASD) classifier from raw MP4 toddler video.

The model samples frames from the video, encodes them with a pretrained
**MobileNetV3-Small** CNN backbone, models temporal dynamics with a
**Bidirectional LSTM + Self-Attention** head, and outputs an ASD/TD
prediction with confidence score and temporal explanation.

**No skeleton or pose estimation required** — the model reads raw pixels
directly and learns discriminative motion patterns end-to-end.

---

## Performance (validation set, 1000 videos)

| Metric | Value |
|--------|-------|
| **Val AUC-ROC** | **0.9953** |
| **Val Accuracy** | **96.00%** |
| **Sensitivity** (ASD recall) | **96.8%** |
| **Specificity** (TD recall) | **95.2%** |
| Best epoch | 68 |
| Training time | ~35 s/epoch on RTX 3050 6GB |

---

## Architecture

```
Raw MP4 video
     │
     ▼
Uniform frame sampling  (16 frames, resized to 96×96)
     │
     ▼
MobileNetV3-Small CNN  (pretrained ImageNet)
     │   per-frame feature vector: 256-dim
     ▼
Learnable positional encoding
     │
     ▼
2-layer Bidirectional LSTM  (hidden=256, out=512)
     │
     ▼
Multi-head Self-Attention  (4 heads) + frame importance weights
     │
     ▼
Weighted temporal pooling  →  (B, 512)
     │
     ▼
MLP  512 → 128 → 1  + temperature-scaled sigmoid
     │
     ▼
ASD probability  |  Confidence  |  Top-3 most diagnostic frames
```

**Parameters:** ~4.87 M  
**Input:**      (B, T=16, 3, 96, 96)  
**Output:**     Binary ASD probability + frame attention weights

---

## Why not skeleton / pose estimation?

The original plan was MediaPipe 2D pose → ST-GCN. It was abandoned because:

- MediaPipe Pose is trained on standing adults. It cannot detect toddlers in
  320×240 video — achieving **0–1% detection rate** across all 19,360 videos,
  regardless of confidence threshold or image upscaling.
- ROMP (used by the MMASD research dataset) is a research-grade 3D mesh
  estimator not suitable for real-time deployment.

The CNN+LSTM approach reads raw pixel appearance and motion directly,
requiring no external pose estimator, and substantially outperforms the
failed skeleton path.

---

## Dataset

| Source | Size | Used for |
|--------|------|----------|
| **autism_data_anonymized** | 4840 ASD + 4840 TD (training) | Training + validation |
| **autism_data_anonymized** | 4840 ASD + 4840 TD (testing) | Hold-out test set |
| **MMASD ROMP-2D** | 690 ASD + 579 TD sequences | SMPL-24 skeleton pre-training (Stage 1, deprecated) |

Training uses class-balanced sampling (WeightedRandomSampler), label-smoothed
BCE loss, MobileNetV3 backbone fine-tuned at 10× lower LR than the LSTM head,
and OneCycleLR scheduling.

Frame cache (`frame_cache/`) pre-decodes all MP4s to `.npy` arrays — makes
each epoch **~8× faster** than on-the-fly decoding.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-cache frames (do this once — takes ~4 min, saves ~8.5 GB)
```bash
python ASD-Detection-Model/cache_frames.py --n_frames 16 --img_size 96 --workers 6
```

### 3. Train the video model
```bash
# Fresh training
python ASD-Detection-Model/train_video.py \
    --batch_size 24 --epochs 60 --workers 6 \
    --n_frames 16 --img_size 96 --lr 3e-4 --patience 15

# Resume from best checkpoint
python ASD-Detection-Model/train_video.py \
    --batch_size 24 --epochs 40 --workers 6 \
    --n_frames 16 --img_size 96 --lr 3e-4 --patience 15 --resume
```

### 4. Predict from a raw MP4 video
```bash
# Basic prediction (auto-uses video model)
python ASD-Detection-Model/predict.py path/to/toddler_video.mp4

# With explanation figure saved
python ASD-Detection-Model/predict.py path/to/toddler_video.mp4 --plot

# Save full results as JSON
python ASD-Detection-Model/predict.py path/to/toddler_video.mp4 --output-json result.json
```

**Example output:**
```
────────────────────────────────────────────────────────────
  📹  Analysing: Subj_100_part_100.mp4
────────────────────────────────────────────────────────────
  [1/3] Extracting 2D pose with MediaPipe …
        Pose quality: 0.0% — falling back to CNN-LSTM video classifier …
        Video model loaded  (epoch=68, val_AUC=0.9953)

╔══════════════════════════════════════════════════════╗
║          ASD SCREENING RESULT                        ║
╠══════════════════════════════════════════════════════╣
║  Diagnosis  :  ASD (Autism Spectrum Disorder)        ║
║  Confidence :  94.1%                                 ║
╠══════════════════════════════════════════════════════╣
║  ASD  [██████████████████░░]   94.1%                 ║
║  TD   [█░░░░░░░░░░░░░░░░░░░]    5.9%                 ║
╠══════════════════════════════════════════════════════╣
║  (CNN video model — frame-level analysis)            ║
╚══════════════════════════════════════════════════════╝
```

---

## Project Structure

```
ASD-Detection-Model/
├── video_model.py       # MobileNetV3 + BiLSTM + Self-Attention classifier
├── video_dataset.py     # RawVideoDataset — reads MP4 or .npy frame cache
├── train_video.py       # Training script with resume, AMP, early stopping
├── cache_frames.py      # Pre-decode MP4 → .npy cache (8× faster training)
├── predict.py           # Full inference pipeline with explanation output
│
├── model.py             # (Legacy) Attention-ST-GCN skeleton classifier
├── train.py             # (Legacy) Two-stage skeleton training
├── pose_extractor.py    # (Legacy) MediaPipe Tasks API extractor
├── dataset.py           # (Legacy) MMASD + skeleton dataset classes
├── extract_poses.py     # (Legacy) Batch pose extraction
├── config.py            # Paths, joint names, hyperparameters
│
├── requirements.txt
├── checkpoints/
│   ├── video_model_best.pth   ← active model (AUC=0.9953)
│   ├── stage1_pretrain.pth    ← MMASD skeleton pre-training (AUC=0.7545)
│   └── best_asd_model.pth     ← failed Stage 2 skeleton (AUC=0.5883)
└── frame_cache/               ← pre-decoded frames (created by cache_frames.py)
```

---

## Disclaimer

This tool is intended as an **AI-assisted screening aid only**.  
It does **not** constitute a clinical diagnosis.  
A formal ASD diagnosis must be performed by a licensed clinician using
validated instruments (ADOS-2, ADI-R, DSM-5 criteria).
