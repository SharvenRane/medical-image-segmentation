"""Paired image/mask dataset for binary medical segmentation.

Expects two parallel folders of matching filenames:
    images/CASE001.png   masks/CASE001.png   (mask: 0 background, 255 foreground)

Plug in any medical segmentation set (lung fields, polyps, lesions) in that
layout. A synthetic generator is included so the pipeline is testable without
downloading data.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, size: int = 256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert("L").resize((self.size, self.size))
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L").resize((self.size, self.size))
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]      # (1,H,W)
        y = torch.from_numpy((np.asarray(mask, np.float32) > 127).astype(np.float32))[None]
        return x, y


def make_synthetic(n: int = 16, size: int = 128, seed: int = 0):
    """Random filled circles on noisy backgrounds -> (images, masks) tensors.

    Used by tests to verify the full train step end to end with no download.
    """
    rng = np.random.default_rng(seed)
    imgs, masks = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
        r = rng.integers(size // 8, size // 4)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2).astype(np.float32)
        img = np.clip(0.3 * rng.random((size, size)) + 0.6 * mask, 0, 1).astype(np.float32)
        imgs.append(img[None])
        masks.append(mask[None])
    return torch.tensor(np.stack(imgs)), torch.tensor(np.stack(masks))
