"""
Siamese network for before/after building damage classification (4 classes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    import torchvision.transforms.functional as TF
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

DAMAGE_CLASSES = ["no_damage", "minor_damage", "major_damage", "destroyed"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DamageClassifierConfig:
    backbone: str = "resnet50"
    num_classes: int = 4
    image_size: Tuple[int, int] = (128, 128)
    pretrained: bool = False
    checkpoint_path: Optional[str] = None
    dropout_rate: float = 0.3
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Siamese architecture
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class SiameseDamageClassifier(nn.Module):
        """
        Shared ResNet50 backbone (without FC) + concatenation fusion MLP.
        forward(before, after) → logits (B, 4)
        """

        def __init__(self, config: DamageClassifierConfig) -> None:
            super().__init__()
            backbone = tv_models.resnet50(weights=None)
            feat_dim = backbone.fc.in_features
            # Strip the final classification layer
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

            # Fusion MLP over concatenated [before_feat; after_feat]
            self.fusion = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim * 2, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate),
                nn.Linear(256, config.num_classes),
            )

        def _extract(self, x):
            return self.feature_extractor(x)

        def forward(self, before, after):
            f_before = self._extract(before)
            f_after = self._extract(after)
            fused = torch.cat([f_before, f_after], dim=1)
            return self.fusion(fused)

else:
    class SiameseDamageClassifier:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required.")


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class DamageClassificationModel:
    def __init__(self, config: DamageClassifierConfig | None = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DamageClassificationModel.")
        self.config = config or DamageClassifierConfig()
        self.device = torch.device(self.config.device)
        self.model = SiameseDamageClassifier(self.config).to(self.device)
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

    # ------------------------------------------------------------------

    def predict(self, before_tensor, after_tensor) -> "torch.Tensor":
        """Return predicted class indices (B,)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                before_tensor.to(self.device),
                after_tensor.to(self.device),
            )
        return logits.argmax(dim=1)

    def predict_proba(self, before_tensor, after_tensor) -> "torch.Tensor":
        """Return class probabilities (B, 4)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                before_tensor.to(self.device),
                after_tensor.to(self.device),
            )
        return F.softmax(logits, dim=1)

    def predict_building(
        self,
        before_image_np: np.ndarray,
        after_image_np: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> int:
        """
        Crop a building patch using bbox (x_min, y_min, x_max, y_max),
        resize to config.image_size and return the predicted damage class int.
        """
        import cv2  # already a project dependency

        x0, y0, x1, y1 = [int(v) for v in bbox]
        crop_b = before_image_np[y0:y1, x0:x1]
        crop_a = after_image_np[y0:y1, x0:x1]

        h, w = self.config.image_size
        crop_b = cv2.resize(crop_b, (w, h))
        crop_a = cv2.resize(crop_a, (w, h))

        def _to_tensor(arr: np.ndarray):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            if t.ndim == 2:
                t = t.unsqueeze(0).repeat(3, 1, 1)
            else:
                t = t.permute(2, 0, 1)
            return t.unsqueeze(0)  # (1, C, H, W)

        pred = self.predict(_to_tensor(crop_b), _to_tensor(crop_a))
        return int(pred.item())

    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "config": asdict(self.config)},
            path,
        )
        logger.info("Classification checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info("Classification checkpoint loaded from %s", path)

    @staticmethod
    def get_class_names() -> List[str]:
        return list(DAMAGE_CLASSES)
