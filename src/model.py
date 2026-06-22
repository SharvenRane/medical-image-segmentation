"""Segmentation model factory: UNet++ and DeepLabV3+ via segmentation-models-pytorch."""
from __future__ import annotations

import segmentation_models_pytorch as smp

ARCHS = {
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "unet": smp.Unet,
}


def build_model(arch: str = "unetplusplus", encoder: str = "resnet34",
                in_channels: int = 1, classes: int = 1):
    """Binary medical segmentation by default (1 input channel, 1 mask channel)."""
    if arch not in ARCHS:
        raise KeyError(f"unknown arch '{arch}', options: {list(ARCHS)}")
    return ARCHS[arch](
        encoder_name=encoder,
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=classes,
    )
