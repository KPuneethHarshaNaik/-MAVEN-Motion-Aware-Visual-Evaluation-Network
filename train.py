"""
Two-Stage Training Script for the ASD Detection Model.

Stage 1 — Pre-train on MMASD 2D skeleton sequences (clean, structured).
Stage 2 — Fine-tune on the combined MMASD + autism_data_anonymized corpus.

Both stages use:
  • Weighted Binary Cross-Entropy (handles class imbalance)
  • Label Smoothing (ε = 0.1)
  • AdamW optimiser + CosineAnnealingLR with linear warmup
  • Automatic Mixed Precision (AMP) — speeds up training on RTX 3050
  • Early stopping (patience = PATIENCE epochs)
  • Gradient clipping (max norm = 1.0)
  • Comprehensive metrics: Accuracy, AUC-ROC, Sensitivity, Specificity, F1

Usage:
  python train.py [--stage 1|2|both]  [--no-cache]  [--epochs N]

  --stage   : Run only stage 1, only stage 2, or both (default: both)
  --no-cache: Use RawVideoDataset instead of pre-extracted NPZ files
  --epochs N: Override max epochs per stage
"""

import os
import sys
import argparse
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEVICE, SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_EPOCHS, PATIENCE, STAGE1_EPOCHS, STAGE2_EPOCHS,
    BEST_MODEL_PATH, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH,
    BASE_DIR, CHECKPOINT_DIR, NUM_JOINTS, IN_CHANNELS, MAX_FRAMES,
)
from model   import ASDClassifier, model_summary
from dataset import MMASDBinaryDataset, PreExtractedDataset, RawVideoDataset
from dataset import ConcatDataset, SkeletonAugmenter, build_splits


# ─── Reproducibility ──────────────────────────────────────────────────────────
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Loss: BCE + Label Smoothing ──────────────────────────────────────────────
class SmoothedBCELoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.eps = smoothing
        self.pw  = pos_weight

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Smooth the targets
        label_smooth = label.float() * (1.0 - self.eps) + 0.5 * self.eps
        loss = F.binary_cross_entropy_with_logits(
            logit,
            label_smooth,
            pos_weight=torch.tensor(self.pw, device=logit.device),
        )
        return loss


import torch.nn.functional as F


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(all_labels, all_probs, threshold=0.5):
    preds = (np.array(all_probs) >= threshold).astype(int)
    labels = np.array(all_labels)
    acc  = (preds == labels).mean()
    try:
        auc = roc_auc_score(labels, all_probs)
    except Exception:
        auc = float("nan")
    try:
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn + 1e-8)   # recall / sensitivity
            spec = tn / (tn + fp + 1e-8)   # specificity
        else:
            sens = spec = float("nan")
    except Exception:
        sens = spec = float("nan")
    f1 = f1_score(labels, preds, zero_division=0)
    return dict(acc=acc, auc=auc, sens=sens, spec=spec, f1=f1)


# ─── Sampler: balanced mini-batches ───────────────────────────────────────────
def make_balanced_sampler(dataset):
    """Create a WeightedRandomSampler to balance ASD/TD mini-batches."""
    labels = []
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        labels.append(int(lbl))
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True,
    )


# ─── LR schedule: linear warmup + cosine decay ────────────────────────────────
def build_scheduler(optimizer, total_epochs, warmup_epochs=WARMUP_EPOCHS):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── One training epoch ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            out  = model(x)
            loss = criterion(out["logit"], y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y)
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(out["prob"].detach().cpu().tolist())

    n   = len(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / n, metrics


# ─── Evaluation epoch ─────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast('cuda'):
            out  = model(x)
            loss = criterion(out["logit"], y)
        total_loss += loss.item() * len(y)
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(out["prob"].cpu().tolist())

    n = len(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / n, metrics


# ─── Save training history plot ───────────────────────────────────────────────
def save_history_plot(history: dict, stage: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Stage {stage} Training History", fontsize=14)

    ep = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(ep, history["train_loss"],  label="train")
    axes[0, 0].plot(ep, history["val_loss"],    label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()

    axes[0, 1].plot(ep, history["train_acc"],   label="train")
    axes[0, 1].plot(ep, history["val_acc"],     label="val")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend()

    axes[1, 0].plot(ep, history["val_auc"],     color="purple")
    axes[1, 0].set_title("Val AUC-ROC")

    axes[1, 1].plot(ep, history["val_sens"],  label="sensitivity")
    axes[1, 1].plot(ep, history["val_spec"],  label="specificity")
    axes[1, 1].plot(ep, history["val_f1"],    label="F1")
    axes[1, 1].set_title("Sens / Spec / F1"); axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(CHECKPOINT_DIR, f"stage{stage}_history.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Saved history plot → {plot_path}")


# ─── Core training loop ───────────────────────────────────────────────────────
def run_training(
    model,
    train_ds,
    val_ds,
    max_epochs: int,
    stage: int,
    save_path: str,
    pos_weight: float = 1.0,
):
    device = DEVICE
    model  = model.to(device)
    print(f"\n{'='*60}")
    print(f"  Stage {stage} training  |  device={device}  |  "
          f"train={len(train_ds)}  val={len(val_ds)}")
    print(f"  {model_summary(model)}")
    print(f"{'='*60}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Optimiser + LR schedule ────────────────────────────────────────────
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler  = build_scheduler(optimizer, max_epochs)
    criterion  = SmoothedBCELoss(smoothing=0.1, pos_weight=pos_weight)
    scaler     = GradScaler('cuda')

    # ── Training loop ──────────────────────────────────────────────────────
    best_auc    = -1.0
    best_acc    = -1.0
    patience_ct = 0
    history     = {k: [] for k in ["train_loss", "val_loss",
                                    "train_acc",  "val_acc",
                                    "val_auc",    "val_sens",
                                    "val_spec",   "val_f1"]}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = train_epoch(model, train_loader, optimizer,
                                    criterion, scaler, device)
        va_loss, va_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_m["acc"])
        history["val_acc"].append(va_m["acc"])
        history["val_auc"].append(va_m["auc"])
        history["val_sens"].append(va_m["sens"])
        history["val_spec"].append(va_m["spec"])
        history["val_f1"].append(va_m["f1"])

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Ep {epoch:03d}/{max_epochs}  "
            f"loss={tr_loss:.4f}/{va_loss:.4f}  "
            f"acc={tr_m['acc']:.4f}/{va_m['acc']:.4f}  "
            f"auc={va_m['auc']:.4f}  "
            f"sens={va_m['sens']:.3f}  "
            f"spec={va_m['spec']:.3f}  "
            f"F1={va_m['f1']:.3f}  "
            f"lr={lr:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # Save best model (by AUC, tie-break by accuracy)
        improved = (va_m["auc"] > best_auc or
                    (va_m["auc"] == best_auc and va_m["acc"] > best_acc))
        if improved:
            best_auc = va_m["auc"]
            best_acc = va_m["acc"]
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "best_auc":   best_auc,
                "best_acc":   best_acc,
                "val_metrics": va_m,
            }, save_path)
            print(f"    ✓ Saved best model  (AUC={best_auc:.4f}  Acc={best_acc:.4f})")
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # Save history JSON + plot
    with open(save_path.replace(".pth", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_history_plot(history, stage)

    return model, best_auc, best_acc


# ─── Stage 1: MMASD only ──────────────────────────────────────────────────────
def stage1(model, max_epochs: int = STAGE1_EPOCHS):
    print("\n" + "═" * 60)
    print("  STAGE 1 — Pre-training on MMASD 2D skeleton data")
    print("═" * 60)

    augmenter = SkeletonAugmenter()
    from dataset import MMASDBinaryDataset, _SubsetDataset
    from config  import TRAIN_RATIO, VAL_RATIO

    rng  = np.random.default_rng(SEED)
    mmasd = MMASDBinaryDataset()
    idxs  = np.arange(len(mmasd))
    rng.shuffle(idxs)
    n_tr = int(len(idxs) * TRAIN_RATIO)
    n_va = int(len(idxs) * VAL_RATIO)
    train_ds = _SubsetDataset(mmasd, idxs[:n_tr],              transform=augmenter)
    val_ds   = _SubsetDataset(mmasd, idxs[n_tr:n_tr + n_va], transform=None)

    # Class ratio for pos_weight
    labels = [mmasd.samples[i][1] for i in idxs[:n_tr]]
    n_asd  = sum(labels); n_td = len(labels) - n_asd
    pw     = n_td / (n_asd + 1e-8)

    return run_training(model, train_ds, val_ds, max_epochs, stage=1,
                        save_path=STAGE1_MODEL_PATH, pos_weight=pw)


# ─── Stage 2: Combined ────────────────────────────────────────────────────────
def stage2(model, use_cache: bool = True, max_epochs: int = STAGE2_EPOCHS):
    print("\n" + "═" * 60)
    print("  STAGE 2 — Fine-tuning on MMASD + autism_data_anonymized")
    print("═" * 60)

    train_ds, val_ds, test_ds = build_splits(
        use_cache=use_cache, augment_train=True
    )

    # Compute pos_weight on combined training labels
    n_asd, n_td = 0, 0
    for _, l in train_ds:
        if l == 1: n_asd += 1
        else:       n_td  += 1
    pw = n_td / (n_asd + 1e-8)
    print(f"  pos_weight = {pw:.3f}  (ASD={n_asd}, TD={n_td})")

    model, best_auc, best_acc = run_training(
        model, train_ds, val_ds, max_epochs, stage=2,
        save_path=STAGE2_MODEL_PATH, pos_weight=pw
    )

    # Copy best stage-2 weights also as the final BEST_MODEL_PATH
    import shutil
    shutil.copy2(STAGE2_MODEL_PATH, BEST_MODEL_PATH)

    # Final evaluation on test set
    print("\n  Evaluating on test set …")
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            all_labels.extend(y.tolist())
            all_probs.extend(out["prob"].cpu().tolist())

    test_m = compute_metrics(all_labels, all_probs)
    print(
        f"\n  TEST RESULTS:\n"
        f"    Accuracy    : {test_m['acc']*100:.2f}%\n"
        f"    AUC-ROC     : {test_m['auc']:.4f}\n"
        f"    Sensitivity : {test_m['sens']*100:.2f}%  (ASD recall)\n"
        f"    Specificity : {test_m['spec']*100:.2f}%  (TD recall)\n"
        f"    F1 Score    : {test_m['f1']:.4f}\n"
    )

    # Save test results
    test_results_path = os.path.join(CHECKPOINT_DIR, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_m, f, indent=2)
    print(f"  Saved test results → {test_results_path}")

    return model


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train ASD Detection Model")
    parser.add_argument("--stage",    type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Which stage to run (default: both)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Use on-the-fly video extraction instead of NPZ cache")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Override max epochs per stage")
    args = parser.parse_args()

    seed_everything(SEED)
    print(f"\n  ASD Detection Model Training")
    print(f"  Device : {DEVICE}")
    print(f"  Seed   : {SEED}")

    model = ASDClassifier()
    use_cache = not args.no_cache

    ep1 = args.epochs if args.epochs else STAGE1_EPOCHS
    ep2 = args.epochs if args.epochs else STAGE2_EPOCHS

    if args.stage in ("1", "both"):
        model, auc1, acc1 = stage1(model, ep1)
        print(f"\n  Stage 1 best  →  AUC={auc1:.4f}  Acc={acc1*100:.2f}%")

    if args.stage in ("2", "both"):
        # Load best stage-1 weights before stage 2
        if args.stage == "2" and os.path.exists(STAGE1_MODEL_PATH):
            ck = torch.load(STAGE1_MODEL_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["state_dict"])
            print("  Loaded Stage 1 weights for fine-tuning.")
        elif not os.path.exists(STAGE1_MODEL_PATH) and args.stage == "2":
            print("  [WARN] Stage 1 checkpoint not found. Training Stage 2 from scratch.")

        model = stage2(model, use_cache=use_cache, max_epochs=ep2)

    print("\n  Training complete. Best model saved to:")
    print(f"    {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
