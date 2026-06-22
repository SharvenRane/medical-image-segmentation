"""Train a binary medical segmentation model on paired image/mask folders.

Usage:
    python src/train.py --images data/images --masks data/masks \
        --arch unetplusplus --encoder resnet34 --epochs 30
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import SegDataset
from model import build_model
from losses import DiceBCELoss, dice_coefficient


@torch.no_grad()
def val_dice(model, loader, device):
    model.eval()
    scores = []
    for x, y in loader:
        scores.append(dice_coefficient(model(x.to(device)), y.to(device)).item())
    return sum(scores) / max(len(scores), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--arch", default="unetplusplus")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="best_seg.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SegDataset(args.images, args.masks, size=args.size)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = build_model(args.arch, args.encoder, in_channels=1, classes=1).to(device)
    crit = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        d = val_dice(model, vl, device)
        print(f"epoch {epoch+1}/{args.epochs}: val_dice={d:.4f}")
        if d > best:
            best = d
            torch.save(model.state_dict(), args.out)
    print(f"Best val Dice: {best:.4f}  ->  {args.out}")


if __name__ == "__main__":
    main()
