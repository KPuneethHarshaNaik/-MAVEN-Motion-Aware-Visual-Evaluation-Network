gge# MAVEN Architecture — Option A: Factorized Temporal Transformers

## Executive Summary

MAVEN (Motion-Aware Visual Evaluation Network) implements **Option A** — a lightweight, modular, GPU-friendly architecture combining:
- **Efficient frame backbone:** MobileNetV3-Small
- **Advanced temporal modeling:** Factorized Temporal Transformer (4 layers, 8 heads)
- **Interpretable pooling:** Temporal Self-Attention + frame importance weighting
- **Clean modular design:** Swappable components for future extensions

---

## Component Overview

### 1. FrameEncoder (`models/frame_backbones.py`)
```python
encoder = FrameEncoder(out_dim=256, pretrained=True)
# Input:  (B, 3, 112, 112)
# Output: (B, 256)
```
- MobileNetV3-Small CNN backbone (ImageNet pretrained)
- Global average pooling → projection (LayerNorm + GELU)
- Efficient: ~100M parameters in base model, only ~2M in the feature extractor

### 2. TemporalTransformer (`models/temporal_transformer.py`)
```python
temporal = TemporalTransformer(d_model=256, n_heads=8, depth=4)
# Input:  (B, T, 256)
# Output: (B, T, 256)
```
- **Factorization:** Per-frame features already pooled spatially (256-dim) → temporal Transformer
- **Positional Encoding:** Learnable (max_len=300 frames)
- **Architecture:** 4 TransformerEncoderLayers (GELU, norm-first)
- **Efficiency:** No spatial dimensions in transformer → fast, memory-light

### 3. TemporalSelfAttention (`models/temporal_transformer.py`)
```python
attn = TemporalSelfAttention(d_model=256, n_heads=4)
# Input:  (B, T, 256)
# Output: (B, T, 256), (B, T)  [attended features + frame weights]
```
- Multi-head self-attention over time
- Linear -> softmax projection produces per-frame importance scores
- Interpretable: frame_weights reveal which frames are diagnostic

### 4. AttentionPoolingHead (`models/fusion.py`)
```python
head = AttentionPoolingHead(d_model=256, hidden=128)
# Input:  attended features (B, T, 256) + frame_weights (B, T)
# Output: logit (B, 1)
```
- Weighted sum of attended frames using importance weights
- MLP: 256 → 128 → 1 (LayerNorm, GELU, dropout)
- Temperature scaling for calibrated sigmoid probabilities

---

## Data Flow

```
Input: (B, T=30, 3, 112, 112)
         ↓
FrameEncoder (per-frame)
         ↓
       (B, T, 256)
         ↓
+ Positional Embeddings
         ↓
TemporalTransformer (4 layers)
         ↓
       (B, T, 256)
         ↓
TemporalSelfAttention
         ↓
    (B, T, 256) + (B, T) frame_weights
         ↓
Weighted Pooling: ctx = sum(attended[t] * weights[t])
         ↓
       (B, 256)
         ↓
AttentionPoolingHead
         ↓
Output: (B, 1) logit → sigmoid(logit) → probability
```

---

## Training Strategy

### Hyperparameters
- **Batch Size:** 16–24 (GPU-dependent)
- **Learning Rate:** 3e-4 (with 0.1× backbone multiplier)
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Scheduler:** OneCycleLR (pct_start=0.05, anneal_strategy='cos')
- **Loss:** BCEWithLogitsLoss (pos_weight = class balance factor)
- **Epochs:** 50–60 typical
- **Early Stopping:** patience=15 epochs

### Layer-Wise Learning Rates
- Backbone (FrameEncoder): `lr * 0.1` (slower fine-tuning)
- Temporal + Head: `lr` (faster learning for new components)

This ensures the pretrained ImageNet features are preserved while allowing the temporal model to adapt.

### Transfer Learning
```bash
python train_video.py --freeze_backbone --lr 1e-3 --epochs 30
```
- Freezes FrameEncoder parameters
- Finetunes temporal transformer + pooling head only
- Use for custom/small datasets

---

## Model Factory Pattern

```python
from video_model import model_factory

# Option A (currently implemented)
model = model_factory(
    name="option_a",
    frame_dim=256,
    n_heads=8,
    transformer_depth=4,
    ff_mult=4,
    dropout=0.4,
    pretrained=True,
)

# Future: easy to add Option B, C, etc.
model_b = model_factory(name="option_b", ...)
```

---

## Performance Characteristics

| Aspect | Value | Notes |
|--------|-------|-------|
| **Parameters** | ~4.6M | Lightweight, single GPU |
| **Inference time** | ~80–120 ms | RTX 3050 GPU |
| **Memory (inference)** | ~500 MB | Single video @ 30 frames |
| **Memory (training)** | ~2–3 GB | Batch size 16 with gradient accumulation |
| **Frame rate** | ~8 FPS | End-to-end on RTX 3050 |

---

## Design Rationale

### Why Factorized Temporal Transformers?

1. **Efficiency:** Per-frame pooling removes spatial dimensions before temporal attention
   - Standard video transformers: (B, T, C, H, W) → (B, T*H*W, C) ← many tokens
   - MAVEN approach: (B, T, 256) ← compressed, fast

2. **Modularity:** Clean separation enables component swaps
   - Replace MobileNetV3 → ConvNeXt, ViT, etc.
   - Add multi-modal streams (skeleton, optical flow) by extending fusion heads

3. **Interpretability:** Attention weights per frame explain model decisions
   - Which frames are diagnostic?
   - Top-3 frame indices in inference output

4. **Trainability:** Factorization reduces overfitting on small datasets
   - Fewer parameters in temporal model
   - Geometry of attention patterns is simpler

---

## Future Extensions (Beyond Option A)

### Option B: Multi-Stream Fusion
- **Skeleton Stream:** PoseFormer-lite on MediaPipe joints
- **RGB Stream:** Existing Option A
- **Fusion:** Cross-attention between streams

### Option C: Pretraining
- **Masked Frame Modeling (VideoMAE-style):** Mask 25% of frames, predict features
- **Contrastive Learning:** Instance discrimination on frame embeddings

### Optical Flow
- Lightweight optical flow extraction (e.g., RAFT)
- Separate temporal encoder for flow
- Late fusion with frame-based stream

---

## Checkpointing & Inference

### Saved Checkpoint Format
```python
{
    "epoch": 45,
    "model_state": <state_dict>,
    "optimizer_state": <optimizer_state>,
    "val_auc": 0.9427,
    "val_acc": 0.9312,
    "args": {
        "n_frames": 30,
        "img_size": 112,
        "batch_size": 24,
        "lr": 0.0003,
        ...
    }
}
```

### Loading & Inference
```python
from video_model import model_factory
import torch

model = model_factory("option_a")
ck = torch.load("checkpoints/video_model_best.pth", map_location="cpu")
model.load_state_dict(ck["model_state"])
model.eval()

# Predict
result = model.predict(videos_tensor)  # (1, T, 3, H, W)
print(f"ASD Prob: {result['prob']:.2%}, Confidence: {result['confidence']:.2%}")
```

---

## Disclaimer

This architecture is designed for **research and AI-assisted screening only**. Clinical diagnosis of ASD requires licensed professionals using validated instruments (ADOS-2, ADI-R, DSM-5).
