# ASD Detection Model

Binary Autism Spectrum Disorder (ASD) classifier from raw MP4 toddler video.

The model samples frames from the video, encodes them with a pretrained
**MobileNetV3-Small** CNN backbone, models temporal dynamics with a
**Bidirectional LSTM + Self-Attention** head, and outputs an ASD/TD
prediction with confidence score and temporal explanation.

**No skeleton or pose estimation required** ΓÇö the model reads raw pixels
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
     Γöé
     Γû╝
Uniform frame sampling  (16 frames, resized to 96├ù96)
     Γöé
     Γû╝
MobileNetV3-Small CNN  (pretrained ImageNet)
     Γöé   per-frame feature vector: 256-dim
     Γû╝
Learnable positional encoding
     Γöé
     Γû╝
2-layer Bidirectional LSTM  (hidden=256, out=512)
     Γöé
     Γû╝
Multi-head Self-Attention  (4 heads) + frame importance weights
     Γöé
     Γû╝
Weighted temporal pooling  ΓåÆ  (B, 512)
     Γöé
     Γû╝
MLP  512 ΓåÆ 128 ΓåÆ 1  + temperature-scaled sigmoid
     Γöé
     Γû╝
ASD probability  |  Confidence  |  Top-3 most diagnostic frames
```

**Parameters:** ~4.87 M  
**Input:**      (B, T=16, 3, 96, 96)  
**Output:**     Binary ASD probability + frame attention weights

---

## Why not skeleton / pose estimation?

The original plan was MediaPipe 2D pose ΓåÆ ST-GCN. It was abandoned because:

- MediaPipe Pose is trained on standing adults. It cannot detect toddlers in
  320├ù240 video ΓÇö achieving **0ΓÇô1% detection rate** across all 19,360 videos,
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
BCE loss, MobileNetV3 backbone fine-tuned at 10├ù lower LR than the LSTM head,
and OneCycleLR scheduling.

Frame cache (`frame_cache/`) pre-decodes all MP4s to `.npy` arrays ΓÇö makes 
each epoch **~8├ù faster** than on-the-fly decoding.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-cache frames (do this once ΓÇö takes ~4 min, saves ~8.5 GB)
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
ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  ≡ƒô╣  Analysing: Subj_100_part_100.mp4
ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  [1/3] Extracting 2D pose with MediaPipe ΓÇª
        Pose quality: 0.0% ΓÇö falling back to CNN-LSTM video classifier ΓÇª
        Video model loaded  (epoch=68, val_AUC=0.9953)

ΓòöΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòù
Γòæ          ASD SCREENING RESULT                        Γòæ
ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  Diagnosis  :  ASD (Autism Spectrum Disorder)        Γòæ
Γòæ  Confidence :  94.1%                                 Γòæ
ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  ASD  [ΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûêΓûæΓûæ]   94.1%                 Γòæ
Γòæ  TD   [ΓûêΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæ]    5.9%                 Γòæ
ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  (CNN video model ΓÇö frame-level analysis)            Γòæ
ΓòÜΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓò¥
```

---

## Project Structure

```
ASD-Detection-Model/
Γö£ΓöÇΓöÇ video_model.py       # MobileNetV3 + BiLSTM + Self-Attention classifier
Γö£ΓöÇΓöÇ video_dataset.py     # RawVideoDataset ΓÇö reads MP4 or .npy frame cache
Γö£ΓöÇΓöÇ train_video.py       # Training script with resume, AMP, early stopping
Γö£ΓöÇΓöÇ cache_frames.py      # Pre-decode MP4 ΓåÆ .npy cache (8├ù faster training)
Γö£ΓöÇΓöÇ predict.py           # Full inference pipeline with explanation output
Γöé
Γö£ΓöÇΓöÇ model.py             # (Legacy) Attention-ST-GCN skeleton classifier
Γö£ΓöÇΓöÇ train.py             # (Legacy) Two-stage skeleton training
Γö£ΓöÇΓöÇ pose_extractor.py    # (Legacy) MediaPipe Tasks API extractor
Γö£ΓöÇΓöÇ dataset.py           # (Legacy) MMASD + skeleton dataset classes
Γö£ΓöÇΓöÇ extract_poses.py     # (Legacy) Batch pose extraction
Γö£ΓöÇΓöÇ config.py            # Paths, joint names, hyperparameters
Γöé
Γö£ΓöÇΓöÇ requirements.txt
Γö£ΓöÇΓöÇ checkpoints/
Γöé   Γö£ΓöÇΓöÇ video_model_best.pth   ΓåÉ active model (AUC=0.9953)
Γöé   Γö£ΓöÇΓöÇ stage1_pretrain.pth    ΓåÉ MMASD skeleton pre-training (AUC=0.7545)
Γöé   ΓööΓöÇΓöÇ best_asd_model.pth     ΓåÉ failed Stage 2 skeleton (AUC=0.5883)
ΓööΓöÇΓöÇ frame_cache/               ΓåÉ pre-decoded frames (created by cache_frames.py)
```

---

## Disclaimer

This tool is intended as an **AI-assisted screening aid only**.  
It does **not** constitute a clinical diagnosis.  
A formal ASD diagnosis must be performed by a licensed clinician using
validated instruments (ADOS-2, ADI-R, DSM-5 criteria).
