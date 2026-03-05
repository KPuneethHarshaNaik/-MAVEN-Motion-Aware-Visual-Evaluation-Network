"""
video_dataset.py
================
PyTorch Dataset that loads raw MP4 frames from autism_data_anonymized and
returns fixed-length frame tensors for the VideoASDClassifier.

Directory layout expected:
    autism_data_anonymized/
        training_set/
            ASD/   *.mp4
            TD/    *.mp4
        testing_set/
            ASD/   *.mp4
            TD/    *.mp4
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Paths (edit if needed)
# ─────────────────────────────────────────────────────────────────────────────
AUTISM_DATA_ROOT = Path("/home/puneeth/Desktop/git hub/autism_data_anonymized")
FRAME_CACHE_ROOT = Path("/home/puneeth/Desktop/git hub/ASD-Detection-Model/frame_cache")

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

class VideoAugment:
    """Frame-level augmentation applied consistently across a clip."""

    def __init__(
        self,
        flip_p       : float = 0.5,
        brightness   : float = 0.3,
        contrast     : float = 0.3,
        saturation   : float = 0.2,
        hue          : float = 0.1,
        crop_scale   : Tuple[float, float] = (0.80, 1.0),
        crop_ratio   : Tuple[float, float] = (0.9, 1.1),
        img_size     : int = 112,
    ):
        self.flip_p    = flip_p
        self.img_size  = img_size

        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.rcrop = transforms.RandomResizedCrop(
            img_size,
            scale=crop_scale,
            ratio=crop_ratio,
            antialias=True,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )

    def __call__(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Args:
            frames: list of H×W×3 uint8 numpy arrays  (BGR)
        Returns:
            (T, 3, H, W) float32 tensor
        """
        from PIL import Image

        flip  = random.random() < self.flip_p
        # Compute crop params once and reuse for whole clip
        crop_params = self.rcrop.get_params(
            Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)),
            self.rcrop.scale, self.rcrop.ratio,
        )

        out = []
        for bgr in frames:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pil = transforms.functional.resized_crop(
                pil,
                *crop_params,
                (self.img_size, self.img_size),
                antialias=True,
            )
            pil = self.color_jitter(pil)
            if flip:
                pil = transforms.functional.hflip(pil)
            t = self.to_tensor(pil)     # (3, H, W) [0,1]
            t = self.normalize(t)
            out.append(t)

        return torch.stack(out)         # (T, 3, H, W)


class VideoTransform:
    """Deterministic resize + normalize (no augmentation) for val/test."""

    def __init__(self, img_size: int = 112):
        self.pipeline = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, frames: List[np.ndarray]) -> torch.Tensor:
        from PIL import Image
        out = []
        for bgr in frames:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out.append(self.pipeline(pil))
        return torch.stack(out)         # (T, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
def _sample_frames(video_path: str, n_frames: int = 30, strategy: str = "uniform") -> List[np.ndarray]:
    """
    Sample n_frames from a video.

    strategy:
        "uniform"  – evenly-spaced frames (deterministic, for val/test)
        "random"   – uniform random jitter (for training augmentation)
    """
    cap      = cv2.VideoCapture(video_path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    if strategy == "random":
        seg_len = max(1, total // n_frames)
        indices = []
        for seg in range(n_frames):
            start = seg * seg_len
            end   = min(start + seg_len, total) - 1
            idx   = random.randint(start, max(start, end))
            indices.append(idx)
    else:
        indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()

    frames   = []
    last_ok  = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            last_ok = bgr
            frames.append(bgr)
        elif last_ok is not None:
            frames.append(last_ok)       # repeat last valid frame
        else:
            frames.append(np.zeros((240, 320, 3), dtype=np.uint8))

    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
class RawVideoDataset(Dataset):
    """
    Dataset that streams raw video frames from autism_data_anonymized.

    Args:
        split      : "training_set" | "testing_set"
        n_frames   : number of frames to sample per video
        img_size   : spatial size fed to CNN (112 or 128)
        augment    : apply random augmentation (training only)
        limit      : cap files per class (None = all)
    """

    _LABEL_MAP = {"ASD": 1, "TD": 0}

    def __init__(
        self,
        split      : str  = "training_set",
        n_frames   : int  = 30,
        img_size   : int  = 112,
        augment    : bool = False,
        limit      : Optional[int] = None,
        data_root  : Path = AUTISM_DATA_ROOT,
        cache_root : Path = FRAME_CACHE_ROOT,
    ):
        self.n_frames   = n_frames
        self.augment    = augment
        self.strategy   = "random" if augment else "uniform"
        self.data_root  = data_root
        self.cache_root = cache_root

        self.transform = VideoAugment(img_size=img_size) if augment \
                         else VideoTransform(img_size=img_size)

        # Collect all (path, label) pairs
        self.samples: List[Tuple[str, int]] = []

        for cls_name, label in self._LABEL_MAP.items():
            cls_dir = data_root / split / cls_name
            if not cls_dir.exists():
                continue
            files = sorted([
                str(f) for f in cls_dir.iterdir()
                if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ])
            if limit:
                files = files[:limit]
            self.samples.extend((f, label) for f in files)

        random.shuffle(self.samples)

        lbl_list      = [s[1] for s in self.samples]
        self.n_asd    = lbl_list.count(1)
        self.n_td     = lbl_list.count(0)

    def __len__(self) -> int:
        return len(self.samples)

    def _npy_path(self, video_path: str) -> Path:
        """Map a video path to its .npy cache file."""
        rel = Path(video_path).relative_to(self.data_root)
        return self.cache_root / rel.with_suffix(".npy")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        npy_path    = self._npy_path(path)

        if npy_path.exists():
            # Fast path: load pre-decoded frames (T, H, W, 3) uint8 RGB
            arr    = np.load(str(npy_path))          # (T, H, W, 3) RGB
            # Convert to BGR list so existing transforms work unchanged
            frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in arr]
        else:
            # Slow path: decode from MP4
            frames = _sample_frames(path, self.n_frames, self.strategy)

        video_tensor = self.transform(frames)        # (T, 3, H, W)
        return video_tensor, label

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Class-balanced sampler for training."""
        total  = len(self.samples)
        w_asd  = total / (2 * max(self.n_asd, 1))
        w_td   = total / (2 * max(self.n_td,  1))
        wts    = [w_asd if s[1] == 1 else w_td for s in self.samples]
        return WeightedRandomSampler(wts, len(wts), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
def build_video_loaders(
    n_frames   : int  = 30,
    img_size   : int  = 112,
    batch_size : int  = 16,
    num_workers: int  = 4,
    val_limit  : Optional[int] = None,   # max samples per class in val
    data_root  : Path = AUTISM_DATA_ROOT,
    cache_root : Path = FRAME_CACHE_ROOT,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    * Training   → training_set with augmentation + balanced sampler
    * Validation → first 500 per class from training_set (no augmentation)
    * Test       → testing_set (no augmentation)

    When frame_cache/ .npy files exist, dataset reads from them
    (~3-4x faster than decoding MP4 on-the-fly).
    """
    use_cache = cache_root.exists() and any(cache_root.rglob("*.npy"))
    if use_cache:
        print(f"Frame cache found at {cache_root} — using fast .npy loading")
    else:
        print("No frame cache found — decoding MP4s on-the-fly (slower)")

    train_ds = RawVideoDataset(
        split="training_set", n_frames=n_frames, img_size=img_size,
        augment=True, data_root=data_root, cache_root=cache_root,
    )
    # Val uses the same training_set folder but deterministic, limited subset
    val_ds = RawVideoDataset(
        split="training_set", n_frames=n_frames, img_size=img_size,
        augment=False, limit=val_limit or 500, data_root=data_root,
        cache_root=cache_root,
    )
    test_ds = RawVideoDataset(
        split="testing_set", n_frames=n_frames, img_size=img_size,
        augment=False, data_root=data_root, cache_root=cache_root,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_ds.make_weighted_sampler(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=True,
    )

    print(f"Train : {len(train_ds)} videos (ASD={train_ds.n_asd}, TD={train_ds.n_td})")
    print(f"Val   : {len(val_ds)} videos (ASD={val_ds.n_asd},  TD={val_ds.n_td})")
    print(f"Test  : {len(test_ds)} videos (ASD={test_ds.n_asd}, TD={test_ds.n_td})")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = RawVideoDataset(split="training_set", n_frames=16, augment=True, limit=10)
    print(f"Dataset size: {len(ds)}")
    t, lbl = ds[0]
    print(f"Tensor shape: {t.shape}   label: {lbl}")
