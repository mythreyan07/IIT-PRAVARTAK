"""
Dataset preparation: splits, DataLoaders, synthetic data generation,
and the XBDDataset torch.utils.data.Dataset implementation.
"""

from __future__ import annotations

import json
import logging
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

from .hf_dataset_loader import get_damage_class_names
from .dataset_inspector import compute_class_weights

logger = logging.getLogger(__name__)

_DAMAGE_CLASSES = get_damage_class_names()
N_DAMAGE_CLASSES = len(_DAMAGE_CLASSES)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """All dataset preparation parameters in one place."""

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # DataLoader
    batch_size: int = 8
    num_workers: int = 0  # 0 = main process (safe on Windows)
    pin_memory: bool = False

    # Images fed to the model
    image_size: Tuple[int, int] = (512, 512)  # (H, W)

    # Reproducibility
    seed: int = 42

    # Whether to apply weighted sampling to address class imbalance
    use_weighted_sampling: bool = True

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )


# ---------------------------------------------------------------------------
# XBDDataset
# ---------------------------------------------------------------------------


class XBDDataset(Dataset):
    """
    PyTorch Dataset for before/after satellite image pairs with building annotations.

    Each item returns::

        {
            "before": torch.Tensor [3, H, W] float32 in [0, 1],
            "after":  torch.Tensor [3, H, W] float32 in [0, 1],
            "label":  torch.Tensor scalar int64  (majority damage class),
            "annotations": list of annotation dicts,
            "pair_id": str,
        }

    Works with both real xBD data and synthetic data from create_synthetic_dataset().
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        before = self._load_image(sample.get("before_path") or sample.get("before"))
        after = self._load_image(sample.get("after_path") or sample.get("after"))

        if self.augment:
            before, after = self._augment_pair(before, after)

        before_t = self._to_tensor(before)
        after_t = self._to_tensor(after)

        annotations = sample.get("annotations", [])
        label = _majority_damage_class(annotations)

        return {
            "before": before_t,
            "after": after_t,
            "label": torch.tensor(label, dtype=torch.long),
            "annotations": annotations,
            "pair_id": str(sample.get("pair_id", f"sample_{idx}")),
        }

    # ------------------------------------------------------------------

    def _load_image(self, source: Any) -> np.ndarray:
        """Load from path string, PIL Image, or numpy array → uint8 HWC RGB."""
        if source is None:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)
        if isinstance(source, np.ndarray):
            img = source.astype(np.uint8)
        elif isinstance(source, Image.Image):
            img = np.array(source.convert("RGB"))
        elif isinstance(source, (str, Path)):
            loaded = cv2.imread(str(source))
            if loaded is None:
                logger.warning("Could not read image: %s", source)
                return np.zeros((*self.image_size, 3), dtype=np.uint8)
            img = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        else:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

        h, w = self.image_size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HWC uint8 → CHW float32 in [0, 1]."""
        return torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0

    def _augment_pair(
        self, before: np.ndarray, after: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical horizontal/vertical flips to before and after."""
        if random.random() > 0.5:
            before = np.fliplr(before).copy()
            after = np.fliplr(after).copy()
        if random.random() > 0.5:
            before = np.flipud(before).copy()
            after = np.flipud(after).copy()
        return before, after


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def prepare_dataset(
    samples: List[Dict[str, Any]],
    config: DatasetConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split a flat list of samples into train / val / test.

    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    rng = random.Random(config.seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    # Give any rounding remainder to test
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def create_data_loaders(
    dataset_dict: Dict[str, List[Dict[str, Any]]],
    config: DatasetConfig,
) -> Dict[str, DataLoader]:
    """
    Build PyTorch DataLoaders for each split.

    Training loader uses WeightedRandomSampler when config.use_weighted_sampling.
    """
    loaders: Dict[str, DataLoader] = {}
    for split_name, samples in dataset_dict.items():
        ds = XBDDataset(
            samples,
            image_size=config.image_size,
            augment=(split_name == "train"),
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and config.use_weighted_sampling and len(ds) > 0:
            sampler = _build_sampler(samples)
            shuffle = False  # sampler and shuffle are mutually exclusive

        loaders[split_name] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=xbd_collate_fn,
        )
    return loaders


def xbd_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that stacks tensors and keeps annotation lists as-is.
    """
    before = torch.stack([item["before"] for item in batch])
    after = torch.stack([item["after"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    annotations = [item["annotations"] for item in batch]
    pair_ids = [item["pair_id"] for item in batch]
    return {
        "before": before,
        "after": after,
        "label": labels,
        "annotations": annotations,
        "pair_id": pair_ids,
    }


def create_synthetic_dataset(
    n_samples: int = 20,
    image_size: Tuple[int, int] = (64, 64),
    n_damage_classes: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create a synthetic in-memory dataset — no HuggingFace download required.

    Each sample contains numpy array images and simple polygon annotations.
    Returned samples are compatible with XBDDataset and inspection functions.
    """
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []

    for i in range(n_samples):
        h, w = image_size
        before_arr = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)
        after_arr = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)

        damage_class = int(i % n_damage_classes)
        samples.append({
            "before": before_arr,
            "after": after_arr,
            "pair_id": f"SYNTH_{i:04d}",
            "annotations": [
                {
                    "uid": f"bldg_{i}_0",
                    "polygon": [[5.0, 5.0], [25.0, 5.0], [25.0, 25.0], [5.0, 25.0]],
                    "damage_class": damage_class,
                    "damage_label": _DAMAGE_CLASSES.get(damage_class, "no_damage"),
                }
            ],
        })
    return samples


def validate_dataset_split(
    dataset_dict: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Check splits for data leakage and class balance.

    Returns a report dict with keys:
    - "leakage_detected": bool
    - "overlapping_pairs": list of pair_ids that appear in multiple splits
    - "split_sizes": {split: n}
    - "split_class_distribution": {split: {class: count}}
    """
    id_sets: Dict[str, set] = {}
    for split, samples in dataset_dict.items():
        id_sets[split] = {s.get("pair_id", f"idx_{j}") for j, s in enumerate(samples)}

    split_names = list(id_sets.keys())
    overlapping: List[str] = []
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            shared = id_sets[split_names[i]] & id_sets[split_names[j]]
            if shared:
                overlapping.extend(list(shared))

    class_dist: Dict[str, Dict[str, int]] = {}
    for split, samples in dataset_dict.items():
        cnt: Dict[str, int] = {v: 0 for v in _DAMAGE_CLASSES.values()}
        for s in samples:
            for ann in s.get("annotations", []):
                lbl = ann.get("damage_label", "no_damage")
                cnt[lbl] = cnt.get(lbl, 0) + 1
        class_dist[split] = cnt

    return {
        "leakage_detected": len(overlapping) > 0,
        "overlapping_pairs": list(set(overlapping)),
        "split_sizes": {k: len(v) for k, v in dataset_dict.items()},
        "split_class_distribution": class_dist,
    }


def save_split_manifest(
    dataset_dict: Dict[str, List[Dict[str, Any]]],
    output_path: str,
) -> None:
    """
    Save a JSON manifest listing which pair_ids belong to each split.
    """
    manifest: Dict[str, Any] = {}
    for split, samples in dataset_dict.items():
        manifest[split] = {
            "n_samples": len(samples),
            "pair_ids": [str(s.get("pair_id", f"idx_{i}")) for i, s in enumerate(samples)],
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Split manifest saved to: %s", output_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _majority_damage_class(annotations: List[Dict[str, Any]]) -> int:
    """Return the most frequent damage class in a building annotation list."""
    if not annotations:
        return 0
    counts: Dict[int, int] = {}
    for ann in annotations:
        c = int(ann.get("damage_class", 0))
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _build_sampler(samples: List[Dict[str, Any]]) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler based on damage class imbalance."""
    class_weights = compute_class_weights(samples)
    sample_weights = []
    for s in samples:
        cls = _majority_damage_class(s.get("annotations", []))
        sample_weights.append(class_weights.get(cls, 1.0))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
