"""Segmentation losses and the Dice metric (the standard for medical masks)."""
from __future__ import annotations

import torch
import torch.nn as nn


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean Dice over the batch for binary masks. Inputs are raw logits."""
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    return ((2 * inter + eps) / (union + eps)).mean()


class DiceBCELoss(nn.Module):
    """Soft Dice + BCE, the workhorse loss for imbalanced medical masks."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        union = prob.sum(dims) + target.sum(dims)
        soft_dice = 1 - ((2 * inter + 1.0) / (union + 1.0)).mean()
        return self.bce_weight * self.bce(logits, target) + (1 - self.bce_weight) * soft_dice
