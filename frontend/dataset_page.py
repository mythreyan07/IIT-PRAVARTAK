"""
EAGLE-DAR — Dataset Inspection Page

Load and inspect the xBD HuggingFace dataset.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def render_dataset_page() -> None:
    st.title("Dataset Inspection")
    st.caption(
        "Inspect the xView2 / xBD satellite building damage dataset from HuggingFace. "
        "Requires `HF_TOKEN` environment variable for private datasets."
    )

    st.header("1. Dataset Configuration")
    col1, col2 = st.columns(2)
    with col1:
        dataset_id = st.text_input(
            "HuggingFace Dataset ID",
            value="kshitijrajsharma/xview2-xbd",
            help="Dataset identifier on HuggingFace Hub",
        )
    with col2:
        split = st.selectbox("Split", ["train", "test", "hold", "tier3"], index=0)

    hf_token = st.text_input(
        "HuggingFace Token (optional)",
        type="password",
        help="Leave blank for public datasets. Set HF_TOKEN env var instead.",
    )

    if st.button("Load & Inspect Dataset"):
        with st.spinner("Loading dataset from HuggingFace…"):
            try:
                from backend.dataset.hf_dataset_loader import load_xbd_dataset, get_damage_class_names
                from backend.dataset.dataset_inspector import generate_inspection_report

                token = hf_token or None
                dataset = load_xbd_dataset(dataset_id, split=split, token=token)
                report = generate_inspection_report({"loaded_split": dataset})
                st.session_state["dataset_report"] = report
                st.session_state["dataset_id"] = dataset_id
                st.success(f"Loaded {len(dataset)} samples from `{dataset_id}` ({split})")
            except Exception as exc:
                st.error(f"Failed to load dataset: {exc}")
                st.info(
                    "Make sure the dataset ID is correct and you have access. "
                    "Public datasets do not require a token."
                )

    # Show QA dataset info
    st.header("2. QA Dataset (xbd-damage-qa)")
    with st.expander("About the QA dataset"):
        st.markdown("""
**Dataset:** `DakshJ27/xbd-damage-qa`

Contains 1,663 Q&A pairs over xBD scenes for evidence-based reasoning.

Fields include: `question`, `damage_bucket`, `disaster_type`, `gt_answer`, `required_reasoning`.

Used by the Question Reasoning module to answer damage-related questions.
        """)

    # Show inspection report if available
    report = st.session_state.get("dataset_report")
    if report:
        st.header("3. Inspection Report")
        _render_inspection_report(report)


def _render_inspection_report(report: dict) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", report.get("total_samples", "N/A"))
    with col2:
        st.metric("Splits", len(report.get("splits", {})))
    with col3:
        st.metric("Annotation Coverage", f"{report.get('annotation_coverage_pct', 0):.1f}%")

    if "damage_class_distribution" in report:
        fig = px.bar(
            x=list(report["damage_class_distribution"].keys()),
            y=list(report["damage_class_distribution"].values()),
            labels={"x": "Damage Class", "y": "Count"},
            title="Damage Class Distribution",
            color_discrete_sequence=["#2196F3"],
        )
        st.plotly_chart(fig, use_container_width=True)

    if "image_dimensions" in report:
        st.subheader("Image Dimensions")
        st.json(report["image_dimensions"])

    with st.expander("Full Report JSON"):
        st.json(report)
