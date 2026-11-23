"""
Damage heatmap and overlay generation for EAGLE-DAR.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Damage class color scheme (BGR for cv2, RGB for matplotlib/plotly)
_COLORS_BGR = {
    0: (0, 200, 0),    # green  — no damage
    1: (0, 220, 255),  # yellow — minor
    2: (0, 128, 255),  # orange — major
    3: (0, 0, 220),    # red    — destroyed
}
_COLORS_RGB = {
    0: (0, 200, 0),
    1: (255, 220, 0),
    2: (255, 128, 0),
    3: (220, 0, 0),
}
_CLASS_NAMES = {0: "No Damage", 1: "Minor Damage", 2: "Major Damage", 3: "Destroyed"}


def create_damage_colormap() -> Dict[int, Tuple[int, int, int]]:
    """Return RGB color for each damage class (0–3)."""
    return dict(_COLORS_RGB)


def create_damage_heatmap(
    image: np.ndarray,
    damage_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a per-pixel damage map onto a satellite image.

    image: H×W×3 uint8 RGB
    damage_map: H×W with values 0–3 (int) or 0.0–1.0 (float continuous score)
    Returns: H×W×3 uint8 RGB
    """
    img = _ensure_rgb_uint8(image)
    h, w = img.shape[:2]
    dm = damage_map.copy()

    if dm.dtype in (np.float32, np.float64):
        # Continuous score [0,1] → rescale to class range [0,3]
        dm = (np.clip(dm, 0, 1) * 3).astype(np.int32)
    else:
        dm = np.clip(dm.astype(np.int32), 0, 3)

    if dm.shape != (h, w):
        dm = cv2.resize(dm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    overlay = np.zeros_like(img, dtype=np.uint8)
    for cls, rgb in _COLORS_RGB.items():
        mask = dm == cls
        overlay[mask] = rgb

    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return result


def create_building_overlay(
    image: np.ndarray,
    buildings: List[Dict[str, Any]],
    damage_classes: List[int],
) -> np.ndarray:
    """
    Draw building polygons/bboxes colour-coded by damage class.

    buildings: list of dicts with 'polygon' (list of [x,y]) or 'bbox' ([x0,y0,x1,y1])
    damage_classes: parallel list of int class labels
    """
    img = _ensure_rgb_uint8(image).copy()
    for bld, cls in zip(buildings, damage_classes):
        color = _COLORS_RGB.get(max(0, min(3, cls)), (128, 128, 128))
        bgr = (color[2], color[1], color[0])  # cv2 expects BGR
        poly = bld.get("polygon")
        bbox = bld.get("bbox")
        if poly and len(poly) >= 3:
            pts = np.array([[int(x), int(y)] for x, y in poly], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=bgr, thickness=2)
            cv2.fillPoly(img, [pts], color=(*bgr, 80))  # semi-transparent fill skipped (fillPoly no alpha)
        elif bbox and len(bbox) == 4:
            x0, y0, x1, y1 = (int(v) for v in bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), color=bgr, thickness=2)
    return img


def plot_before_after_comparison(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Before / After Comparison",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_ensure_rgb_uint8(before))
    axes[0].set_title("Before")
    axes[0].axis("off")
    axes[1].imshow(_ensure_rgb_uint8(after))
    axes[1].set_title("After")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_damage_distribution(evidence_items: List[Any]) -> go.Figure:
    """Bar chart of building counts per damage class."""
    counts = {name: 0 for name in _CLASS_NAMES.values()}
    for item in evidence_items:
        cls = getattr(item, "damage_class", 0)
        name = _CLASS_NAMES.get(max(0, min(3, cls)), "Unknown")
        counts[name] = counts.get(name, 0) + 1

    colors = [
        f"rgb{_COLORS_RGB[k]}"
        for k in range(4)
    ]
    fig = go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=colors,
            text=list(counts.values()),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Building Damage Distribution",
        xaxis_title="Damage Class",
        yaxis_title="Number of Buildings",
        showlegend=False,
        height=400,
    )
    return fig


def save_heatmap(image: np.ndarray, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(_ensure_rgb_uint8(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dest), img_bgr)
    logger.info("Heatmap saved to %s", dest)
    return dest


def generate_scene_heatmap(
    before_image: np.ndarray,
    after_image: np.ndarray,
    buildings_with_damage: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce a full heatmap dict for a scene.

    buildings_with_damage: list of {polygon, damage_class, confidence}
    """
    after_rgb = _ensure_rgb_uint8(after_image)
    h, w = after_rgb.shape[:2]

    # Build a pixel-level damage map from building polygons/bboxes
    damage_map = np.zeros((h, w), dtype=np.int32)
    for bld in buildings_with_damage:
        cls = max(0, min(3, int(bld.get("damage_class", 0))))
        poly = bld.get("polygon")
        bbox = bld.get("bbox")
        if poly and len(poly) >= 3:
            pts = np.array([[int(x), int(y)] for x, y in poly], dtype=np.int32)
            cv2.fillPoly(damage_map, [pts], color=cls)
        elif bbox and len(bbox) == 4:
            x0, y0, x1, y1 = (int(v) for v in bbox)
            damage_map[y0:y1, x0:x1] = cls

    heatmap = create_damage_heatmap(after_rgb, damage_map, alpha=0.5)

    # Count stats
    stats: Dict[str, int] = {_CLASS_NAMES[k]: 0 for k in range(4)}
    for bld in buildings_with_damage:
        cls = max(0, min(3, int(bld.get("damage_class", 0))))
        stats[_CLASS_NAMES[cls]] += 1

    legend = {_CLASS_NAMES[k]: f"rgb{_COLORS_RGB[k]}" for k in range(4)}
    return {"heatmap": heatmap, "legend": legend, "stats": stats}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img
