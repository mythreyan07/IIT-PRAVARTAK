"""
Training orchestration for segmentation and classification models.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_type: str = "classification"       # "segmentation" | "classification"
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 5
    device: str = "cpu"
    use_amp: bool = False
    log_every_n_steps: int = 10


@dataclass
class TrainingResult:
    model_type: str
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_metric: float = 0.0
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    duration_seconds: float = 0.0
    config_dict: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best: Optional[float] = None

    def __call__(self, val_loss: float) -> bool:
        """Return True when training should stop."""
        if self._best is None or val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            return False
        self._counter += 1
        if self._counter >= self.patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", self.patience)
            return True
        return False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _iou_score(pred: "torch.Tensor", target: "torch.Tensor", eps: float = 1e-6) -> float:
    """Binary IoU for a batch of segmentation masks."""
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def _accuracy(preds: "torch.Tensor", labels: "torch.Tensor") -> float:
    return float((preds.argmax(dim=1) == labels).float().mean())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: "nn.Module",
        train_loader,
        val_loader,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelTrainer.")
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(
            config.device
            if config.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if (config.use_amp and torch.cuda.is_available()) else None
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        result = TrainingResult(
            model_type=self.config.model_type,
            config_dict=asdict(self.config),
        )
        t0 = time.time()
        best_ckpt: Optional[str] = None

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._history.append(row)
            logger.info("Epoch %d — %s", epoch, row)

            val_loss = val_metrics["loss"]
            val_metric = val_metrics.get("iou", val_metrics.get("accuracy", 0.0))

            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_val_metric = val_metric
                result.best_epoch = epoch
                best_ckpt = self.save_best_checkpoint(val_metric, epoch)
                result.checkpoint_path = best_ckpt

            if self.early_stopping(val_loss):
                logger.info("Stopping early at epoch %d.", epoch)
                break

        result.training_history = list(self._history)
        result.duration_seconds = time.time() - t0
        return result

    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, total_metric, steps = 0.0, 0.0, 0

        for step, batch in enumerate(self.train_loader):
            loss, metric = self._step(batch, train=True)
            total_loss += loss
            total_metric += metric
            steps += 1
            if steps % self.config.log_every_n_steps == 0:
                logger.debug("Epoch %d step %d loss=%.4f", epoch, steps, loss)

        n = max(steps, 1)
        return self._metric_dict(total_loss / n, total_metric / n)

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_metric, steps = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                loss, metric = self._step(batch, train=False)
                total_loss += loss
                total_metric += metric
                steps += 1

        n = max(steps, 1)
        return self._metric_dict(total_loss / n, total_metric / n)

    # ------------------------------------------------------------------

    def _step(self, batch: Dict, train: bool):
        """Single forward pass. Returns (loss_float, metric_float)."""
        is_seg = self.config.model_type == "segmentation"

        if is_seg:
            images = batch["image"].to(self.device)
            masks = batch["mask"].long().to(self.device)
        else:
            before = batch["before"].to(self.device)
            after = batch["after"].to(self.device)
            labels = batch["label"].long().to(self.device)

        if train:
            self.optimizer.zero_grad()

        use_amp = self.scaler is not None
        with autocast(enabled=use_amp):
            if is_seg:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                probs = torch.softmax(logits, dim=1)[:, 1]
                metric = _iou_score(probs, (masks == 1).float())
            else:
                logits = self.model(before, after)
                loss = self.criterion(logits, labels)
                metric = _accuracy(logits, labels)

        if train:
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return float(loss), float(metric)

    def _metric_dict(self, loss: float, metric: float) -> Dict[str, float]:
        key = "iou" if self.config.model_type == "segmentation" else "accuracy"
        return {"loss": loss, key: metric}

    def save_best_checkpoint(self, val_metric: float, epoch: int) -> str:
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        path = str(
            Path(self.config.checkpoint_dir)
            / f"{self.config.model_type}_best_ep{epoch}_metric{val_metric:.4f}.pt"
        )
        torch.save(self.model.state_dict(), path)
        logger.info("Best checkpoint saved: %s", path)
        return path

    def get_training_history(self) -> List[Dict[str, Any]]:
        return list(self._history)
