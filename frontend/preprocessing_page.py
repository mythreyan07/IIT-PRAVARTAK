"""
EAGLE-DAR — Image Preprocessing Page

Provides upload, automatic preprocessing, quality/alignment inspection,
processed image preview, and download for before/after satellite image pairs.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy import of preprocessing pipeline
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_pipeline():
    from backend.preprocessing.preprocessing_pipeline import (
        PreprocessingConfig,
        PreprocessingPipeline,
    )
    from backend.preprocessing.image_normalization import NormalizationConfig
    from backend.preprocessing.image_quality import QualityThresholds
    from backend.preprocessing.image_alignment import AlignmentConfig

    config = PreprocessingConfig(
        normalization=NormalizationConfig(target_size=(512, 512)),
        quality=QualityThresholds(),
        alignment=AlignmentConfig(),
    )
    return PreprocessingPipeline(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a uint8 numpy array (RGB or GRAY) to a PIL Image."""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _quality_badge(quality: str) -> str:
    colours = {"Good": "🟢", "Warning": "🟡", "Poor": "🔴"}
    return f"{colours.get(quality, '⚪')} {quality}"


def _plot_quality_gauge(value: float, title: str, min_val=0.0, max_val=1.0) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "bar": {"color": "steelblue"},
                "steps": [
                    {"range": [min_val, (max_val - min_val) * 0.4 + min_val], "color": "#ffcccc"},
                    {"range": [(max_val - min_val) * 0.4 + min_val, (max_val - min_val) * 0.7 + min_val], "color": "#fff3cc"},
                    {"range": [(max_val - min_val) * 0.7 + min_val, max_val], "color": "#ccffcc"},
                ],
            },
        )
    )
    fig.update_layout(height=200, margin=dict(t=40, b=0, l=10, r=10))
    return fig


def _save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(uploaded_file.read())
        return f.name


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render_preprocessing_page() -> None:
    st.title("Image Preprocessing")
    st.caption(
        "Upload before-and-after satellite image pairs. "
        "The pipeline automatically loads, validates, aligns, normalises "
        "and prepares images for the building detection model."
    )

    # Upload section
    st.header("1. Upload Image Pair")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        before_file = st.file_uploader(
            "Before image",
            type=["jpg", "jpeg", "png"],
            key="before_upload",
        )
    with col_up2:
        after_file = st.file_uploader(
            "After image",
            type=["jpg", "jpeg", "png"],
            key="after_upload",
        )

    if not before_file or not after_file:
        st.info("Upload both a before and an after image to begin preprocessing.")
        return

    # Show original images
    st.header("2. Original Images")
    col_orig1, col_orig2 = st.columns(2)
    with col_orig1:
        st.image(before_file, caption="Original Before", use_container_width=True)
    with col_orig2:
        st.image(after_file, caption="Original After", use_container_width=True)

    # Run pipeline
    if "pipeline_result" not in st.session_state or st.button("Re-run Preprocessing"):
        with st.spinner("Running preprocessing pipeline…"):
            try:
                pipeline = _get_pipeline()
                before_tmp = _save_upload_to_temp(before_file)
                after_tmp = _save_upload_to_temp(after_file)
                result = pipeline.process_image_pair(before_tmp, after_tmp)
                st.session_state["pipeline_result"] = result
                st.session_state["before_tmp"] = before_tmp
                st.session_state["after_tmp"] = after_tmp
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

    result = st.session_state.get("pipeline_result")
    if result is None:
        return

    # Status banner
    st.header("3. Processing Status")
    if result.status == "completed":
        st.success(f"Status: Completed — Pair ID: `{result.image_pair_id}`")
    elif result.status == "unreliable":
        st.warning(
            f"Status: Unreliable — alignment could not be verified. "
            f"Pair ID: `{result.image_pair_id}`"
        )
    else:
        st.error(f"Status: Failed — Pair ID: `{result.image_pair_id}`")

    if result.warnings:
        with st.expander("Warnings", expanded=True):
            for w in result.warnings:
                st.warning(w)
    if result.errors:
        with st.expander("Errors", expanded=True):
            for e in result.errors:
                st.error(e)

    # Image quality
    st.header("4. Image Quality")
    _render_quality_section(result)

    # Processed images
    st.header("5. Processed Images")
    _render_processed_images(result)

    # Alignment
    st.header("6. Alignment")
    _render_alignment_section(result)

    # Metadata
    st.header("7. Full Metadata")
    with st.expander("View JSON metadata"):
        st.json(result.to_dict())

    # Downloads
    st.header("8. Download Processed Previews")
    _render_downloads(result)


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


def _render_quality_section(result) -> None:
    col1, col2 = st.columns(2)
    for col, label, quality in [
        (col1, "Before", result.before_quality),
        (col2, "After", result.after_quality),
    ]:
        if not quality:
            continue
        with col:
            st.subheader(f"{label} Image Quality")
            st.metric("Overall", _quality_badge(quality.get("overall_quality", "N/A")))
            st.metric("Blur Score", f"{quality.get('blur_score', 0):.1f}")
            st.metric("Brightness", quality.get("brightness_status", "N/A"))
            st.metric("Contrast", quality.get("contrast_status", "N/A"))
            st.metric("Resolution", quality.get("resolution_status", "N/A"))
            if quality.get("blur_score") is not None:
                fig = _plot_quality_gauge(
                    min(quality["blur_score"] / 500, 1.0),
                    f"{label} Sharpness",
                )
                st.plotly_chart(fig, use_container_width=True)


def _render_processed_images(result) -> None:
    col1, col2 = st.columns(2)
    for col, label, path in [
        (col1, "Processed Before", result.processed_before),
        (col2, "Processed After", result.processed_after),
    ]:
        with col:
            st.subheader(label)
            if path and Path(path).exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Not available.")

    # Normalised
    if result.normalized_before or result.normalized_after:
        st.subheader("Normalised Images")
        c1, c2 = st.columns(2)
        for c, label, path in [
            (c1, "Normalised Before", result.normalized_before),
            (c2, "Normalised After", result.normalized_after),
        ]:
            with c:
                if path and Path(path).exists():
                    st.image(str(path), caption=label, use_container_width=True)


def _render_alignment_section(result) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Alignment Required", "Yes" if result.alignment_required else "No")
        st.metric("Alignment Method", result.alignment_method or "N/A")
        st.metric("Alignment Status", result.alignment_status)
        if result.alignment_quality is not None:
            st.metric("NCC Quality", f"{result.alignment_quality:.4f}")
            fig = _plot_quality_gauge(
                float(result.alignment_quality),
                "Alignment NCC",
                min_val=-1.0,
                max_val=1.0,
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if result.aligned_after and Path(result.aligned_after).exists():
            st.subheader("Aligned After Image")
            st.image(str(result.aligned_after), use_container_width=True)
        else:
            st.info("No aligned image available.")


def _render_downloads(result) -> None:
    paths = {
        "Processed Before": result.processed_before,
        "Processed After": result.processed_after,
        "Aligned After": result.aligned_after,
        "Normalised Before": result.normalized_before,
        "Normalised After": result.normalized_after,
    }
    cols = st.columns(len([v for v in paths.values() if v and Path(v).exists()]) or 1)
    ci = 0
    for label, path in paths.items():
        if path and Path(path).exists():
            img_bytes = Path(path).read_bytes()
            with cols[ci]:
                st.download_button(
                    label=f"Download {label}",
                    data=img_bytes,
                    file_name=Path(path).name,
                    mime="image/png",
                )
            ci += 1

    # Also offer metadata JSON
    meta_json = json.dumps(result.to_dict(), indent=2, default=str)
    st.download_button(
        label="Download Metadata JSON",
        data=meta_json.encode(),
        file_name=f"{result.image_pair_id}_metadata.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Entry point when run directly via streamlit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_preprocessing_page()
