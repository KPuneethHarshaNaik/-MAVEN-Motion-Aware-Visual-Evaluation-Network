"""
Training script for the ASD Skeleton Action Recognition Model.

Usage:
    python train.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    SKELETON_MODE, MODEL_SAVE_DIR,
    NUM_JOINTS_2D, NUM_JOINTS_3D,
    IN_CHANNELS_2D, IN_CHANNELS_3D,
    NUM_CLASSES, MAX_FRAMES,
    HIDDEN_CHANNELS, DROPOUT, NUM_STGCN_BLOCKS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEED, ACTIVITY_LABELS,
)
from model import STGCN, count_parameters
from dataset import MMASDSkeletonDataset, SkeletonNormalizer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(dataset):
    """Split dataset into train/val/test and create DataLoaders."""
    total = len(dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(loader):
        data = data.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set. Returns loss, accuracy, predictions, and labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Save training curves as PNG images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(train_losses, label='Train Loss', color='#2196F3', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', color='#F44336', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(train_accs, label='Train Accuracy', color='#2196F3', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', color='#F44336', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"  Training curves saved to {save_dir}/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Label', xlabel='Predicted Label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_dir}/confusion_matrix.png")


def main():
    print("=" * 60)
    print("  ASD Skeleton Action Recognition - Training")
    print("=" * 60)

    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Determine input config based on skeleton mode
    if SKELETON_MODE == "2d":
        in_channels = IN_CHANNELS_2D
        num_joints = NUM_JOINTS_2D
    else:
        in_channels = IN_CHANNELS_3D
        num_joints = NUM_JOINTS_3D

    # Load dataset
    print(f"\nLoading MMASD dataset ({SKELETON_MODE.upper()} skeletons)...")
    normalizer = SkeletonNormalizer(center_joint=1, mode=SKELETON_MODE)
    dataset = MMASDSkeletonDataset(
        mode=SKELETON_MODE,
        max_frames=MAX_FRAMES,
        transform=normalizer,
    )

    if len(dataset) == 0:
        print("\n" + "!" * 60)
        print("  ERROR: No data samples found!")
        print(f"  Please download the full MMASD dataset and update")
        print(f"  DATASET_ROOT in config.py (currently: {DATASET_ROOT})")
        print("!" * 60)
        return

    # Create data loaders
    print("\nSplitting dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    # Build model
    print(f"\nBuilding ST-GCN model...")
    model = STGCN(
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        num_joints=num_joints,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:6.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:6.2f}%  |  "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  >>> New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print("-" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, MODEL_SAVE_DIR)

    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")

    # Classification report
    label_names = [k for k, v in sorted(ACTIVITY_LABELS.items(), key=lambda x: x[1])]
    # Filter to only classes present in test set
    present_classes = sorted(set(test_labels))
    present_names = [label_names[c] for c in present_classes]

    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                labels=present_classes,
                                target_names=present_names,
                                zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, present_names, MODEL_SAVE_DIR)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model saved to: {MODEL_SAVE_DIR}/best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
