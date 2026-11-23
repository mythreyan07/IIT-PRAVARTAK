"""
U-Net building footprint segmentation model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SegmentationConfig:
    backbone: str = "unet"
    num_classes: int = 2
    image_size: Tuple[int, int] = (512, 512)
    pretrained: bool = False
    checkpoint_path: Optional[str] = None
    device: str = "cpu"


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class UNet(nn.Module):
        """
        Lightweight U-Net for binary building segmentation.
        Encoder: 3→64→128→256→512, bottleneck 512→1024, symmetric decoder.
        """

        def __init__(self, num_classes: int = 2) -> None:
            super().__init__()
            self.enc1 = _ConvBlock(3, 64)
            self.enc2 = _ConvBlock(64, 128)
            self.enc3 = _ConvBlock(128, 256)
            self.enc4 = _ConvBlock(256, 512)
            self.pool = nn.MaxPool2d(2)

            self.bottleneck = _ConvBlock(512, 1024)

            self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = _ConvBlock(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = _ConvBlock(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = _ConvBlock(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = _ConvBlock(128, 64)

            self.final = nn.Conv2d(64, num_classes, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            b = self.bottleneck(self.pool(e4))

            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.final(d1)

else:
    class UNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for UNet.")


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class BuildingSegmentationModel:
    def __init__(self, config: SegmentationConfig | None = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BuildingSegmentationModel.")
        self.config = config or SegmentationConfig()
        self.device = torch.device(self.config.device)
        self.model = UNet(num_classes=self.config.num_classes).to(self.device)
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

    def predict(self, image_tensor) -> "torch.Tensor":
        """Return argmax binary mask (B, H, W)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
        return logits.argmax(dim=1)

    def predict_proba(self, image_tensor) -> "torch.Tensor":
        """Return softmax probability for the foreground class (B, 1, H, W)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
        probs = F.softmax(logits, dim=1)
        return probs[:, 1:2]  # foreground channel

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "config": asdict(self.config)},
            path,
        )
        logger.info("Segmentation checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info("Segmentation checkpoint loaded from %s", path)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
