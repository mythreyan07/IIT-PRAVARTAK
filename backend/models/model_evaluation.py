"""
Evaluation metrics and reporting for segmentation and classification models.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def compute_segmentation_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Binary IoU for single or batched masks (any shape)."""
    pred = (pred_mask > 0.5).astype(np.float32).ravel()
    true = true_mask.astype(np.float32).ravel()
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return float((intersection + eps) / (union + eps))


def compute_segmentation_metrics(
    pred_masks: np.ndarray,
    true_masks: np.ndarray,
) -> Dict[str, float]:
    """
    Compute IoU, F1, precision and recall over flat binary arrays.
    pred_masks and true_masks should be (N, H, W) or flattened.
    """
    pred = (pred_masks > 0.5).astype(np.int32).ravel()
    true = true_masks.astype(np.int32).ravel()

    iou = compute_segmentation_iou(pred_masks, true_masks)

    if _SKLEARN_AVAILABLE:
        f1 = float(f1_score(true, pred, average="binary", zero_division=0))
        precision = float(precision_score(true, pred, average="binary", zero_division=0))
        recall = float(recall_score(true, pred, average="binary", zero_division=0))
    else:
        tp = int((pred * true).sum())
        fp = int((pred * (1 - true)).sum())
        fn = int(((1 - pred) * true).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"iou": round(iou, 6), "f1": round(f1, 6),
            "precision": round(precision, 6), "recall": round(recall, 6)}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    if _SKLEARN_AVAILABLE:
        return float(accuracy_score(labels, preds))
    return float(np.mean(preds == labels))


def compute_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict:
    if not _SKLEARN_AVAILABLE:
        return {"accuracy": compute_classification_accuracy(preds, labels)}
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def compute_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    if _SKLEARN_AVAILABLE:
        return confusion_matrix(labels, preds, labels=list(range(num_classes)))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Model-level evaluation loops
# ---------------------------------------------------------------------------

def evaluate_segmentation_model(model, data_loader, device: str = "cpu") -> Dict:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")
    import torch
    dev = torch.device(device)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch["image"].to(dev)
            masks = batch["mask"].cpu().numpy()
            logits = model(imgs)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_true.append(masks)
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return compute_segmentation_metrics(all_preds, all_true)


def evaluate_classification_model(
    model, data_loader, device: str = "cpu",
    class_names: Optional[List[str]] = None,
) -> Dict:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")
    import torch
    dev = torch.device(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            before = batch["before"].to(dev)
            after = batch["after"].to(dev)
            labels = batch["label"].cpu().numpy()
            logits = model(before, after)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    names = class_names or [str(i) for i in range(preds.max() + 1)]
    report = compute_classification_report(preds, labels, names)
    report["accuracy"] = compute_classification_accuracy(preds, labels)
    return report


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
) -> Optional["plt.Figure"]:  # type: ignore[name-defined]
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available; skipping confusion matrix plot.")
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    seg_metrics: Optional[Dict] = None,
    cls_metrics: Optional[Dict] = None,
) -> Dict:
    return {
        "segmentation": seg_metrics or {},
        "classification": cls_metrics or {},
    }
