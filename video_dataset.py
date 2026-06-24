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
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
AUTISM_DATA_ROOT = WORKSPACE_ROOT / "autism_data_anonymized"
FRAME_CACHE_ROOT = PROJECT_ROOT / "frame_cache"

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
        mask_prob    : float = 0.15,
    ):
        self.flip_p    = flip_p
        self.img_size  = img_size
        self.mask_prob = mask_prob

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

        tensor = torch.stack(out)         # (T, 3, H, W)
        
        # Temporal Frame Masking
        if self.mask_prob > 0:
            mask = torch.rand(tensor.size(0)) < self.mask_prob
            tensor[mask] = 0.0
            
        return tensor


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
        samples    : pre-built list of (path, label) tuples (overrides split/limit scanning)
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
        samples    : Optional[List[Tuple[str, int]]] = None,
    ):
        self.n_frames   = n_frames
        self.augment    = augment
        self.strategy   = "random" if augment else "uniform"
        self.data_root  = data_root
        self.cache_root = cache_root

        self.transform = VideoAugment(img_size=img_size) if augment \
                         else VideoTransform(img_size=img_size)

        if samples is not None:
            # Use pre-built sample list (for proper train/val splitting)
            self.samples: List[Tuple[str, int]] = list(samples)
        else:
            # Collect all (path, label) pairs from directory
            self.samples = []
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
        try:
            rel = Path(video_path).relative_to(self.data_root)
        except ValueError:
            # Video path is outside data_root, use hash-based name
            import hashlib
            h = hashlib.md5(video_path.encode()).hexdigest()[:12]
            rel = Path(h).with_suffix(".npy")
        return self.cache_root / rel.with_suffix(".npy")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        npy_path    = self._npy_path(path)

        if npy_path.exists():
            # Fast path: load pre-decoded frames (T, H, W, 3) uint8 RGB
            arr    = np.load(str(npy_path))          # (T, H, W, 3) RGB
            # Convert to BGR list so existing transforms work unchanged
            frames_list = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in arr]
            # Resample if cached frames don't match expected count
            if len(frames_list) != self.n_frames:
                indices = np.linspace(0, len(frames_list) - 1, self.n_frames, dtype=int)
                frames_list = [frames_list[i] for i in indices]
            frames = frames_list
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
def _collect_video_files(data_root: Path, split: str, limit: Optional[int] = None
                        ) -> List[Tuple[str, int]]:
    """Collect (path, label) pairs from data_root/split/{ASD,TD}."""
    label_map = {"ASD": 1, "TD": 0}
    samples = []
    for cls_name, label in label_map.items():
        cls_dir = data_root / split / cls_name
        if not cls_dir.exists():
            continue
        files = sorted([
            str(f) for f in cls_dir.iterdir()
            if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ])
        if limit:
            files = files[:limit]
        samples.extend((f, label) for f in files)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
def build_video_loaders(
    n_frames   : int  = 30,
    img_size   : int  = 112,
    batch_size : int  = 16,
    num_workers: int  = 4,
    val_split  : float = 0.2,            # fraction of training_set for validation
    train_limit: Optional[int] = None,   # max videos per class in training set
    data_root  : Path = AUTISM_DATA_ROOT,
    cache_root : Path = FRAME_CACHE_ROOT,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Returns (train_loader, val_loader, test_loader).

    * Training   → (1 - val_split) of training_set with augmentation + balanced sampler
    * Validation → val_split of training_set (no augmentation, no overlap with train)
    * Test       → testing_set (no augmentation), or None if testing_set is empty

    The train/val split is deterministic (seeded) for reproducibility.
    When frame_cache/ .npy files exist, dataset reads from them
    (~3-4x faster than decoding MP4 on-the-fly).
    """
    use_cache = cache_root.exists() and any(cache_root.rglob("*.npy"))
    if use_cache:
        print(f"Frame cache found at {cache_root} - using fast .npy loading")
    else:
        print("No frame cache found - decoding MP4s on-the-fly (slower)")

    # ── Collect and split training data ──────────────────────────
    all_train_samples = _collect_video_files(data_root, "training_set", limit=train_limit)

    # Deterministic shuffle so the split is reproducible across runs
    rng = random.Random(42)
    rng.shuffle(all_train_samples)

    n_val = max(1, int(len(all_train_samples) * val_split))
    val_samples   = all_train_samples[:n_val]
    train_samples = all_train_samples[n_val:]

    print(f"\nSplit {len(all_train_samples)} training_set videos -> "
          f"{len(train_samples)} train + {len(val_samples)} val  "
          f"(val_split={val_split:.0%})")

    train_ds = RawVideoDataset(
        n_frames=n_frames, img_size=img_size, augment=True,
        data_root=data_root, cache_root=cache_root, samples=train_samples,
    )
    val_ds = RawVideoDataset(
        n_frames=n_frames, img_size=img_size, augment=False,
        data_root=data_root, cache_root=cache_root, samples=val_samples,
    )

    # ── Test set (may be empty / missing) ────────────────────────
    test_samples = _collect_video_files(data_root, "testing_set", limit=train_limit)
    test_ds = None
    test_loader = None
    if test_samples:
        test_ds = RawVideoDataset(
            n_frames=n_frames, img_size=img_size, augment=False,
            data_root=data_root, cache_root=cache_root, samples=test_samples,
        )

    # ── DataLoaders ──────────────────────────────────────────────
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
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(num_workers, 4),
            pin_memory=True,
        )

    print(f"Train : {len(train_ds)} videos (ASD={train_ds.n_asd}, TD={train_ds.n_td})")
    print(f"Val   : {len(val_ds)} videos (ASD={val_ds.n_asd},  TD={val_ds.n_td})")
    if test_ds:
        print(f"Test  : {len(test_ds)} videos (ASD={test_ds.n_asd}, TD={test_ds.n_td})")
    else:
        print(f"Test  : [Warning] No testing_set found - test evaluation will be skipped")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = RawVideoDataset(split="training_set", n_frames=16, augment=True, limit=10)
    print(f"Dataset size: {len(ds)}")
    t, lbl = ds[0]
    print(f"Tensor shape: {t.shape}   label: {lbl}")
