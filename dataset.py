"""
Dataset classes for the ASD Detection Model.

Two data sources are combined:
  1. MMASDBinaryDataset  ΓÇö MMASD ROMP-2D pre-computed skeleton sequences
                           (folder name: ..._y = ASD, ..._n = TD, ..._i = skip)
  2. PreExtractedDataset ΓÇö MediaPipe-extracted skeleton NPZ files from
                           autism_data_anonymized (cached by extract_poses.py)

Both return tensors of shape (2, MAX_FRAMES, 24)  ΓÇö (C, T, V) ΓÇö so they can
be combined in a single DataLoader via ConcatDataset.

Data Augmentation (applied only during training):
  ΓÇó Horizontal flip (mirror left/right joints)
  ΓÇó Gaussian jitter on joint positions
  ΓÇó Random temporal crop + re-sampling
  ΓÇó Random temporal speed perturbation
  ΓÇó Random joint dropout (zero out 1-2 joints)
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from config import (
    MMASD_SKEL_2D, POSE_CACHE_DIR, AUTISM_DATA_ROOT,
    MMASD_LABEL_MAP, MAX_FRAMES, NUM_JOINTS, IN_CHANNELS,
    MAX_AUTISM_VIDEOS_PER_CLASS, SEED, SMPL24_EDGES,
)
from pose_extractor import skeleton_from_npz


# ΓöÇΓöÇΓöÇ Horizontal mirror mapping for SMPL-24 ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# When we flip x-axis we swap leftΓåöright joints
_SMPL24_MIRROR = {
    1: 2,  2: 1,    # L_Hip Γåö R_Hip
    4: 5,  5: 4,    # L_Knee Γåö R_Knee
    7: 8,  8: 7,    # L_Ankle Γåö R_Ankle
    10:11, 11:10,   # L_Foot Γåö R_Foot
    13:14, 14:13,   # L_Collar Γåö R_Collar
    16:17, 17:16,   # L_Shoulder Γåö R_Shoulder
    18:19, 19:18,   # L_Elbow Γåö R_Elbow
    20:21, 21:20,   # L_Wrist Γåö R_Wrist
    22:23, 23:22,   # L_Hand Γåö R_Hand
}
_MIRROR_LUT = list(range(NUM_JOINTS))
for a, b in _SMPL24_MIRROR.items():
    _MIRROR_LUT[a] = b


# ΓöÇΓöÇΓöÇ Augmentation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class SkeletonAugmenter:
    """
    Stochastic augmentation pipeline for (C=2, T, V=24) float32 tensors.
    All transforms are applied in-place on numpy arrays.
    """

    def __init__(
        self,
        flip_prob:        float = 0.5,
        jitter_std:       float = 0.02,
        dropout_prob:     float = 0.3,
        max_dropout_joints: int = 2,
        temporal_crop:    bool  = True,
        speed_perturb:    bool  = True,
    ):
        self.flip_prob  = flip_prob
        self.jitter_std = jitter_std
        self.drop_prob  = dropout_prob
        self.max_drop   = max_dropout_joints
        self.temporal_crop  = temporal_crop
        self.speed_perturb  = speed_perturb

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        arr = tensor.numpy().copy()   # (2, T, 24)

        # 1. Horizontal flip
        if random.random() < self.flip_prob:
            arr[0] *= -1                       # Negate x-channel
            arr = arr[:, :, _MIRROR_LUT]       # Swap L/R joints

        # 2. Gaussian coordinate jitter
        if self.jitter_std > 0:
            arr += np.random.randn(*arr.shape).astype(np.float32) * self.jitter_std

        # 3. Random joint dropout
        if self.drop_prob > 0 and random.random() < self.drop_prob:
            n_drop = random.randint(1, self.max_drop)
            joints = random.sample(range(NUM_JOINTS), n_drop)
            arr[:, :, joints] = 0.0

        # 4. Temporal crop
        T = arr.shape[1]
        if self.temporal_crop and T > 10:
            crop_len = random.randint(int(T * 0.8), T)
            start = random.randint(0, T - crop_len)
            cropped = arr[:, start:start + crop_len, :]
            # Re-sample to MAX_FRAMES
            idx = np.linspace(0, crop_len - 1, MAX_FRAMES, dtype=int)
            arr = cropped[:, idx, :]

        # 5. Speed perturbation (non-uniform temporal re-sampling)
        if self.speed_perturb and random.random() < 0.4:
            T2 = arr.shape[1]
            perts = np.cumsum(np.random.uniform(0.5, 1.5, T2))
            perts = (perts / perts[-1] * (T2 - 1)).astype(int)
            perts = np.clip(perts, 0, T2 - 1)
            arr = arr[:, perts, :]

        return torch.from_numpy(arr)


# ΓöÇΓöÇΓöÇ Dataset 1: MMASD ROMP-2D skeleton data ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class MMASDBinaryDataset(Dataset):
    """
    Loads all ROMP-2D folder sequences from the MMASD dataset.

    Folder naming convention:  <activity>_<subject>_<day>_<seq>_<label>
    where label Γêê {'y'=ASD, 'n'=TD, 'i'=skip-indeterminate}.

    Returns: (tensor: (2, MAX_FRAMES, 24), label: int)
    """

    def __init__(self, skel_dir: str = MMASD_SKEL_2D,
                 transform=None):
        super().__init__()
        self.transform = transform
        self.samples: list[tuple[str, int]] = []   # (folder_path, label)

        if not os.path.exists(skel_dir):
            print(f"[WARN] MMASD dir not found: {skel_dir}")
            return

        for activity in sorted(os.listdir(skel_dir)):
            act_path = os.path.join(skel_dir, activity)
            if not os.path.isdir(act_path):
                continue
            for sample_name in sorted(os.listdir(act_path)):
                seq_path = os.path.join(act_path, sample_name)
                if not os.path.isdir(seq_path):
                    continue
                suffix = sample_name.split("_")[-1].lower()
                if suffix not in MMASD_LABEL_MAP:
                    continue    # skip 'i' (indeterminate)
                label = MMASD_LABEL_MAP[suffix]
                self.samples.append((seq_path, label))

        asd = sum(1 for _, l in self.samples if l == 1)
        td  = sum(1 for _, l in self.samples if l == 0)
        print(f"[MMASDBinaryDataset] {len(self.samples)} sequences  "
              f"(ASD={asd}, TD={td})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        folder, label = self.samples[idx]
        skel = skeleton_from_npz(folder)          # (2, 120, 24)
        tensor = torch.from_numpy(skel)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


# ΓöÇΓöÇΓöÇ Dataset 2: Pre-extracted MediaPipe skeletons ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class PreExtractedDataset(Dataset):
    """
    Loads MediaPipe-extracted skeleton NPZ files produced by extract_poses.py.

    Expected cache structure:
      POSE_CACHE_DIR/
        training_set/
          ASD/   <sample_name>.npz  (shape: 2, MAX_FRAMES, 24)
          TD/    <sample_name>.npz
        testing_set/
          ASD/   ...
          TD/    ...

    Returns: (tensor: (2, MAX_FRAMES, 24), label: int)
    """

    def __init__(
        self,
        split: str = "training_set",     # "training_set" | "testing_set"
        cache_dir: str = POSE_CACHE_DIR,
        max_per_class: int | None = MAX_AUTISM_VIDEOS_PER_CLASS,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for class_name, label in [("ASD", 1), ("TD", 0)]:
            class_dir = os.path.join(cache_dir, split, class_name)
            if not os.path.exists(class_dir):
                continue
            npz_files = sorted(glob.glob(os.path.join(class_dir, "*.npz")))
            if max_per_class is not None:
                npz_files = npz_files[:max_per_class]
            for f in npz_files:
                self.samples.append((f, label))

        asd = sum(1 for _, l in self.samples if l == 1)
        td  = sum(1 for _, l in self.samples if l == 0)
        print(f"[PreExtractedDataset/{split}] {len(self.samples)} samples  "
              f"(ASD={asd}, TD={td})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        data  = np.load(path)
        skel  = data["skeleton"].astype(np.float32)  # (2, T, 24)
        tensor = torch.from_numpy(skel)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


# ΓöÇΓöÇΓöÇ Dataset 3:  Raw video dataset (on-the-fly extraction) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class RawVideoDataset(Dataset):
    """
    Loads raw MP4 videos and extracts poses on the fly via MediaPipe.
    Used only when no pre-extracted cache is available.
    NOTE: significantly slower than PreExtractedDataset.

    Returns: (tensor: (2, MAX_FRAMES, 24), label: int)
    """

    def __init__(
        self,
        root: str = AUTISM_DATA_ROOT,
        split: str = "training_set",
        max_per_class: int | None = MAX_AUTISM_VIDEOS_PER_CLASS,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for class_name, label in [("ASD", 1), ("TD", 0)]:
            class_dir = os.path.join(root, split, class_name)
            if not os.path.exists(class_dir):
                continue
            mp4s = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
            if max_per_class is not None:
                mp4s = mp4s[:max_per_class]
            for v in mp4s:
                self.samples.append((v, label))

        asd = sum(1 for _, l in self.samples if l == 1)
        td  = sum(1 for _, l in self.samples if l == 0)
        print(f"[RawVideoDataset/{split}] {len(self.samples)} videos  "
              f"(ASD={asd}, TD={td})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from pose_extractor import extract_skeleton_from_video
        path, label = self.samples[idx]
        skel   = extract_skeleton_from_video(path)    # (2, T, 24)
        tensor = torch.from_numpy(skel)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


# ΓöÇΓöÇΓöÇ Helper: build train / val / test splits ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def build_splits(
    use_cache: bool = True,
    augment_train: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Construct train / val / test datasets combining MMASD + autism_data.

    Strategy:
      ΓÇó MMASD sequences are split 75 / 15 / 10 by sequence (subject-agnostic).
      ΓÇó autism_data pre-extracted NPZ:
            train  ΓåÆ training_set/ASD + training_set/TD (split internally)
            test   ΓåÆ testing_set/ASD + testing_set/TD

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from config import TRAIN_RATIO, VAL_RATIO, SEED
    rng = np.random.default_rng(SEED)

    augmenter = SkeletonAugmenter() if augment_train else None

    # ΓöÇΓöÇ MMASD splits ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    mmasd_all = MMASDBinaryDataset()
    indices   = np.arange(len(mmasd_all))
    rng.shuffle(indices)

    n_train = int(len(indices) * TRAIN_RATIO)
    n_val   = int(len(indices) * VAL_RATIO)
    i_train, i_val, i_test = (indices[:n_train],
                               indices[n_train:n_train + n_val],
                               indices[n_train + n_val:])

    _mmasd_train = _SubsetDataset(mmasd_all, i_train, transform=augmenter)
    _mmasd_val   = _SubsetDataset(mmasd_all, i_val,   transform=None)
    _mmasd_test  = _SubsetDataset(mmasd_all, i_test,  transform=None)

    # ΓöÇΓöÇ autism_data splits ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if use_cache:
        DataCls = PreExtractedDataset
        kwargs_tr   = dict(split="training_set",  transform=augmenter)
        kwargs_val  = dict(split="training_set",  transform=None,
                           max_per_class=500)   # small validation subset
        kwargs_test = dict(split="testing_set",   transform=None)
    else:
        DataCls = RawVideoDataset
        kwargs_tr   = dict(split="training_set",  transform=augmenter)
        kwargs_val  = dict(split="training_set",  transform=None, max_per_class=200)
        kwargs_test = dict(split="testing_set",   transform=None, max_per_class=500)

    _aut_train = DataCls(**kwargs_tr)
    _aut_val   = DataCls(**kwargs_val)
    _aut_test  = DataCls(**kwargs_test)

    train_ds = ConcatDataset([_mmasd_train, _aut_train])
    val_ds   = ConcatDataset([_mmasd_val,   _aut_val])
    test_ds  = ConcatDataset([_mmasd_test,  _aut_test])

    return train_ds, val_ds, test_ds


# ΓöÇΓöÇΓöÇ Internal: index-based subset wrapper ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class _SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices, transform=None):
        self.base      = base
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        tensor, label = self.base[self.indices[idx]]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


if __name__ == "__main__":
    ds = MMASDBinaryDataset()
    print(f"MMASD binary dataset: {len(ds)} samples")
    t, l = ds[0]
    print(f"  Sample shape: {t.shape},  Label: {l}")
