"""Verify the segmentation pipeline runs end to end on synthetic data (CPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from model import build_model, ARCHS
from losses import DiceBCELoss, dice_coefficient
from dataset import make_synthetic


def test_archs_build_and_forward():
    x, _ = make_synthetic(n=2, size=128)
    for arch in ARCHS:
        m = build_model(arch, encoder="resnet18", in_channels=1, classes=1)
        out = m(x)
        assert out.shape == (2, 1, 128, 128)


def test_dice_perfect_and_zero():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    perfect = (target * 20 - 10)          # logits that sigmoid->~target
    assert dice_coefficient(perfect, target).item() > 0.99
    wrong = (1 - target) * 20 - 10
    assert dice_coefficient(wrong, target).item() < 0.01


def test_one_train_step_reduces_loss():
    x, y = make_synthetic(n=8, size=128)
    model = build_model("unetplusplus", encoder="resnet18", in_channels=1, classes=1)
    crit = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first = None
    for _ in range(3):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first  # loss should drop over a few steps


if __name__ == "__main__":
    test_archs_build_and_forward()
    test_dice_perfect_and_zero()
    test_one_train_step_reduces_loss()
    print("all tests passed")
