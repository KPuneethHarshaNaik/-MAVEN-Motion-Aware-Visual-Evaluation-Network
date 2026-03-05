"""
train_video.py
==============
Trains the VideoASDClassifier (MobileNetV3 + BiLSTM) on raw MP4 frames
from autism_data_anonymized.

Usage:
    python train_video.py [--batch_size 16] [--epochs 60] [--workers 4]
                          [--n_frames 30] [--img_size 112] [--lr 3e-4]
                          [--limit 2000]

The script:
  1. Builds train / val / test dataloaders
  2. Trains with AdamW + OneCycleLR + AMP + early stopping
  3. Saves best checkpoint to checkpoints/video_model_best.pth
  4. Reports final test AUC, accuracy, sensitivity, specificity
"""

import sys, os, json, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from video_model   import VideoASDClassifier
from video_dataset import build_video_loaders

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_MODEL     = CHECKPOINT_DIR / "video_model_best.pth"
HISTORY_FILE   = CHECKPOINT_DIR / "video_history.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs",     type=int, default=60)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--n_frames",   type=int, default=30)
    p.add_argument("--img_size",   type=int, default=112)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--patience",   type=int, default=15)
    p.add_argument("--limit",      type=int, default=None,
                   help="max videos per class in training set (None=all)")
    p.add_argument("--resume",     action="store_true",
                   help="resume training from best checkpoint")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(labels, probs):
    preds  = (np.array(probs) >= 0.5).astype(int)
    labels = np.array(labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-9)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    return dict(acc=acc, auc=auc, sens=sens, spec=spec, f1=f1)


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0; all_labels = []; all_probs = []
    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.float().to(device)
        with autocast("cuda"):
            logit, _ = model(videos)
            loss = criterion(logit.squeeze(1), labels)
        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logit).squeeze(1).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    m = compute_metrics(all_labels, all_probs)
    return avg_loss, m


# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  VideoASDClassifier  |  device={device}")
    print(f"  n_frames={args.n_frames}  img_size={args.img_size}")
    print(f"  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_video_loaders(
        n_frames    = args.n_frames,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.workers,
    )

    # ── Model ─────────────────────────────────────────────────
    model = VideoASDClassifier(
        frame_dim   = 256,
        lstm_hidden = 256,
        lstm_layers = 2,
        dropout     = 0.4,
        pretrained  = True,
    ).to(device)

    start_epoch = 1
    best_auc    = 0.0
    best_epoch  = 0
    history     = []

    # ── Optional resume ────────────────────────────────────────
    if args.resume and BEST_MODEL.exists():
        ck = torch.load(BEST_MODEL, weights_only=False)
        model.load_state_dict(ck["model_state"])
        best_auc    = ck.get("val_auc", 0.0)
        best_epoch  = ck.get("epoch", 0)
        start_epoch = best_epoch + 1
        # Halve the LR for fine-tuning continuation
        args.lr = args.lr * 0.3
        # Load history if present
        if HISTORY_FILE.exists():
            import json as _json
            saved = _json.load(open(HISTORY_FILE))
            history = saved.get("history", [])
        print(f"  Resumed from epoch {best_epoch}  "
              f"(best AUC={best_auc:.4f})  "
              f"new LR={args.lr:.2e}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}\n")

    # ── Loss ──────────────────────────────────────────────────
    # Label-smoothed BCE
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [train_loader.dataset.n_td / max(train_loader.dataset.n_asd, 1)]
    ).to(device))

    # ── Optimiser ─────────────────────────────────────────────
    # Different LRs for backbone vs head
    backbone_params = list(model.encoder.features.parameters())
    other_params    = [p for p in model.parameters()
                       if not any(p is q for q in backbone_params)]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=1e-4)

    n_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr      = [args.lr * 0.1, args.lr],
        total_steps = n_steps,
        pct_start   = 0.05,
        anneal_strategy = "cos",
    )

    scaler = GradScaler("cuda")

    # ── Training loop ──────────────────────────────────────────
    if not args.resume:
        best_auc   = 0.0
        best_epoch = 0
        history    = []

    patience   = args.patience

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        t0 = time.time()

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                logit, _ = model(videos)
                loss = criterion(logit.squeeze(1), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

            if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:3d}  Batch {batch_idx:4d}/{len(train_loader)}  "
                      f"Loss={epoch_loss/n_batches:.4f}  ({elapsed:.0f}s)", flush=True)

        # Epoch-level validation
        val_loss, vm = evaluate(model, val_loader, device, criterion)
        train_avg    = epoch_loss / max(n_batches, 1)

        row = dict(epoch=epoch, tr_loss=train_avg, **{f"val_{k}": v for k, v in vm.items()})
        history.append(row)

        print(f"\nEpoch {epoch:3d}/{args.epochs}  "
              f"tr_loss={train_avg:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={vm['acc']*100:.2f}%  "
              f"val_auc={vm['auc']:.4f}  "
              f"sens={vm['sens']*100:.1f}%  "
              f"spec={vm['spec']*100:.1f}%")

        # Save best
        if vm["auc"] > best_auc:
            best_auc   = vm["auc"]
            best_epoch = epoch
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc"        : best_auc,
                "val_acc"        : vm["acc"],
                "args"           : vars(args),
            }, BEST_MODEL)
            print(f"  ★ New best saved  AUC={best_auc:.4f}")

        patience_remaining = patience - (epoch - best_epoch)
        if patience_remaining <= 0:
            print(f"\nEarly stopping at epoch {epoch}  (best epoch={best_epoch})")
            break

    # ── Final test ────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading best checkpoint and evaluating on test set …")
    ckpt = torch.load(BEST_MODEL, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, tm = evaluate(model, test_loader, device, criterion)
    print(f"\n  TEST RESULTS:")
    print(f"    Accuracy    : {tm['acc']*100:.2f}%")
    print(f"    AUC-ROC     : {tm['auc']:.4f}")
    print(f"    Sensitivity : {tm['sens']*100:.2f}%  (ASD recall)")
    print(f"    Specificity : {tm['spec']*100:.2f}%  (TD recall)")
    print(f"    F1 Score    : {tm['f1']:.4f}")
    print("="*60)

    result = {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "test": tm,
        "history": history,
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nHistory saved → {HISTORY_FILE}")
    print(f"Best model   → {BEST_MODEL}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
