"""
Dataset loader for the MMASD (Multimodal ASD) skeleton data.

Supports both:
  - 2D skeleton: OpenPose JSON format (25 keypoints)
  - 3D skeleton: ROMP NPZ format (71 keypoints)
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from config import (
    ACTIVITY_LABELS, MAX_FRAMES,
    NUM_JOINTS_2D, NUM_JOINTS_3D,
    IN_CHANNELS_2D, IN_CHANNELS_3D,
    SKELETON_2D_SUBTYPE,
    SKELETON_DIR_2D_OPENPOSE, SKELETON_DIR_2D_ROMP,
    SKELETON_DIR_3D,
)


class MMASDSkeletonDataset(Dataset):
    """
    PyTorch Dataset for MMASD skeleton sequences.

    Actual folder structure (activity-grouped):
        skeleton_dir/
            arm_swing_as/
                as_20583_D1_000_y/
                    *.json (OpenPose) or *.npz (ROMP)
                ...
            body_swing_bs/
                ...

    Each sub-subfolder is one video sample. The activity label is parsed
    from the sample folder name prefix (e.g., "as" = Arm Swing).
    """

    def __init__(self, mode="2d", subtype_2d=SKELETON_2D_SUBTYPE,
                 max_frames=MAX_FRAMES, person_index=0, transform=None):
        """
        Args:
            mode: "2d" or "3d" skeleton modality.
            subtype_2d: "openpose" (JSON, 25 joints) or "romp" (NPZ) when mode="2d".
            max_frames: Fixed sequence length (pad/truncate).
            person_index: Which detected person to use (0 = first person).
            transform: Optional callable transform on the tensor.
        """
        super().__init__()
        self.mode = mode
        self.subtype_2d = subtype_2d
        self.max_frames = max_frames
        self.person_index = person_index
        self.transform = transform

        if mode == "2d":
            self.num_joints = NUM_JOINTS_2D
            self.in_channels = IN_CHANNELS_2D
            if subtype_2d == "openpose":
                skeleton_dir = SKELETON_DIR_2D_OPENPOSE
            else:
                skeleton_dir = SKELETON_DIR_2D_ROMP
        elif mode == "3d":
            self.num_joints = NUM_JOINTS_3D
            self.in_channels = IN_CHANNELS_3D
            skeleton_dir = SKELETON_DIR_3D
        else:
            raise ValueError(f"Unknown mode: {mode}. Use '2d' or '3d'.")

        # Discover all sample folders (structure: skeleton_dir/activity/sample/)
        self.samples = []
        if not os.path.exists(skeleton_dir):
            print(f"WARNING: Skeleton directory not found: {skeleton_dir}")
            print("Make sure DATASET_ROOT and _DRIVE_FOLDER in config.py are correct.")
            return

        for activity_folder in sorted(os.listdir(skeleton_dir)):
            activity_path = os.path.join(skeleton_dir, activity_folder)
            if not os.path.isdir(activity_path):
                continue

            for sample_name in sorted(os.listdir(activity_path)):
                sample_path = os.path.join(activity_path, sample_name)
                if not os.path.isdir(sample_path):
                    continue

                # Parse activity label from sample folder name prefix
                prefix = sample_name.split("_")[0].lower()
                if prefix not in ACTIVITY_LABELS:
                    print(f"  Skipping unknown prefix '{prefix}' in: {sample_name}")
                    continue

                label = ACTIVITY_LABELS[prefix]
                self.samples.append((sample_path, label, sample_name))

        print(f"Loaded {len(self.samples)} samples in '{mode}' mode from {skeleton_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label, folder_name = self.samples[idx]

        if self.mode == "2d":
            if self.subtype_2d == "openpose":
                skeleton_seq = self._load_2d_openpose_sequence(folder_path)
            else:
                skeleton_seq = self._load_2d_romp_sequence(folder_path)
        else:
            skeleton_seq = self._load_3d_sequence(folder_path)

        # Pad or truncate to fixed length
        skeleton_seq = self._fix_length(skeleton_seq)

        # Convert to tensor: shape (C, T, V) where C=channels, T=frames, V=joints
        tensor = torch.tensor(skeleton_seq, dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

    def _load_2d_openpose_sequence(self, folder_path):
        """
        Load a sequence of OpenPose 2D skeleton JSON files.
        Returns: (C, T, V) numpy array where C=2 (x,y), T=num_frames, V=25
        """
        json_files = sorted(glob.glob(os.path.join(folder_path, "*_keypoints.json")))
        if not json_files:
            # Try alternative patterns
            json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))

        frames = []
        for jf in json_files:
            with open(jf, 'r') as f:
                data = json.load(f)

            people = data.get("people", [])
            if len(people) > self.person_index:
                kp = people[self.person_index]["pose_keypoints_2d"]
                # kp is a flat list: [x0, y0, conf0, x1, y1, conf1, ...]
                kp = np.array(kp).reshape(self.num_joints, 3)
                # Take only x, y (drop confidence)
                xy = kp[:, :2]  # (25, 2)
            else:
                # Person not detected in this frame — fill with zeros
                xy = np.zeros((self.num_joints, 2))

            frames.append(xy)

        if not frames:
            frames = [np.zeros((self.num_joints, 2))]

        # Stack: (T, V, C) -> transpose to (C, T, V)
        seq = np.stack(frames, axis=0)  # (T, V, C)
        seq = seq.transpose(2, 0, 1)    # (C, T, V)
        return seq.astype(np.float32)

    def _load_2d_romp_sequence(self, folder_path):
        """
        Load a sequence of ROMP 2D skeleton NPZ files.
        Returns: (C, T, V) numpy array where C=2 (x,y), T=num_frames, V=25
        """
        npz_files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
        frames = []

        for nf in npz_files:
            data = np.load(nf)
            coords = data["coordinates"]  # (num_people, joints, 2)
            if coords.shape[0] > self.person_index:
                xy = coords[self.person_index]  # (joints, 2)
            else:
                xy = np.zeros((self.num_joints, 2))
            frames.append(xy)

        if not frames:
            frames = [np.zeros((self.num_joints, 2))]

        # Stack: (T, V, C) -> transpose to (C, T, V)
        seq = np.stack(frames, axis=0)  # (T, V, C)
        seq = seq.transpose(2, 0, 1)    # (C, T, V)
        return seq.astype(np.float32)

    def _load_3d_sequence(self, folder_path):
        """
        Load a sequence of ROMP 3D skeleton NPZ files.
        Returns: (C, T, V) numpy array where C=3 (x,y,z), T=num_frames, V=71
        """
        npz_files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
        frames = []

        for nf in npz_files:
            data = np.load(nf)
            coords = data["coordinates"]  # (num_people, 71, 3)
            if coords.shape[0] > self.person_index:
                xyz = coords[self.person_index]  # (71, 3)
            else:
                xyz = np.zeros((self.num_joints, 3))
            frames.append(xyz)

        if not frames:
            frames = [np.zeros((self.num_joints, 3))]

        # Stack: (T, V, C) -> transpose to (C, T, V)
        seq = np.stack(frames, axis=0)  # (T, V, C)
        seq = seq.transpose(2, 0, 1)    # (C, T, V)
        return seq.astype(np.float32)

    def _fix_length(self, seq):
        """
        Pad or truncate a sequence to self.max_frames.
        seq shape: (C, T, V)
        """
        C, T, V = seq.shape
        if T >= self.max_frames:
            # Uniform sampling if too long
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            seq = seq[:, indices, :]
        else:
            # Zero-pad if too short
            pad = np.zeros((C, self.max_frames - T, V), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)
        return seq


class SkeletonNormalizer:
    """
    Normalize skeleton coordinates by centering on the hip/neck
    and scaling to unit bounding box. This makes the model
    invariant to the person's position in the frame and body size.
    """

    def __init__(self, center_joint=1, mode="2d"):
        """
        Args:
            center_joint: Joint index to use as center (1 = Neck for OpenPose).
            mode: "2d" or "3d".
        """
        self.center_joint = center_joint
        self.mode = mode

    def __call__(self, tensor):
        """
        Args:
            tensor: (C, T, V) skeleton tensor
        Returns:
            Normalized tensor of same shape
        """
        # Center on the reference joint
        center = tensor[:, :, self.center_joint:self.center_joint + 1]  # (C, T, 1)
        tensor = tensor - center

        # Scale to [-1, 1] range
        max_val = tensor.abs().max()
        if max_val > 0:
            tensor = tensor / max_val

        return tensor


if __name__ == "__main__":
    # Quick test with sample data path
    from config import SKELETON_MODE

    normalizer = SkeletonNormalizer(center_joint=1, mode=SKELETON_MODE)
    dataset = MMASDSkeletonDataset(
        mode=SKELETON_MODE,
        transform=normalizer,
    )

    if len(dataset) > 0:
        sample, label = dataset[0]
        print(f"Sample shape: {sample.shape}")  # (C, T, V)
        print(f"Label: {label}")
    else:
        print("No samples found. Please download the full MMASD dataset.")
        print("Update DATASET_ROOT and _DRIVE_FOLDER in config.py.")
