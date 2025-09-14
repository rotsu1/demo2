import os
import sys
import time
import math
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from PIL import Image

# Use non-interactive backend for HPC environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
class BrainMRIDataset(Dataset):
    """
    Dataset for preprocessed OASIS brain MR images stored as PNGs.

    Recursively scans one or more root directories for `.png` files and
    returns single-channel tensors in the range [0, 1]. By default, files
    with paths containing "seg" are excluded (to avoid segmentation masks).

    Args:
        roots: Directory path or list of directory paths to scan recursively
            for PNG images. Example: [
            "/home/groups/comp3710/OASIS/keras_png_slices_train",
            "/home/groups/comp3710/OASIS/keras_png_slices_validate",
            "/home/groups/comp3710/OASIS/keras_png_slices_test",
            ].
        transform: Optional torchvision transform to apply to each PIL image
            (e.g., grayscale conversion, resize to `IMG_SIZE`, `ToTensor`).
        limit: Optional maximum number of images to load (useful for quick
            tests). If None or <= 0, uses all discovered images.
        include_seg: If True, does not filter out files whose paths contain
            "seg"; if False (default), such files are excluded.

    Returns:
        Each `__getitem__` returns a tuple `(image_tensor, 0)`, where the
        label placeholder `0` is unused (unsupervised setting).
    """

    def __init__(self, roots, transform: T.Compose | None = None, limit: int | None = None, include_seg: bool = False):
        self.roots = [roots] if isinstance(roots, str) else list(roots)
        # Recursively find PNG files across roots
        files = []
        for root in self.roots:
            patterns = [
                os.path.join(root, "**", "*.png"),
                os.path.join(root, "**", "*.PNG"),
            ]
            for p in patterns:
                # extend all matching paths
                files.extend(glob(p, recursive=True))
        # Exclude segmentation files unless requested
        if not include_seg:
            files = [f for f in files if 'seg' not in f.lower()]
        self.files = sorted(files)
        if limit is not None and limit > 0:
            self.files = self.files[:limit]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No PNG files found under: {self.roots}")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")  # force 1-channel grayscale
        if self.transform:
            # convert it to tensor that can be used for training
            img = self.transform(img)
        return img, 0  # label unused (unsupervised)