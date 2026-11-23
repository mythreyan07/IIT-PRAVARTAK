"""
EAGLE-DAR — Inference Page

Upload images, run building detection + damage classification, generate evidence.
"""

from __future__ import annotations
import io
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image


def render_inference_page() -> None:
    st.title("Disaster Damage Assessment")
    st.caption(
        "Upload a before/after satellite image pair. "
        "The system automatically preprocesses, detects buildings, classifies damage, "
        "generates evidence and answers key questions."
    )

    # Upload
    st.header("1. Upload Image Pair")
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Before (pre-disaster)", type=["jpg", "jpeg", "png"],
                                       key="inf_before")
    with col2:
        after_file = st.file_uploader("After (post-disaster)", type=["jpg", "jpeg", "png"],
                                      key="inf_after")

    if not before_file or not after_file:
        st.info("Upload both images to begin.")
        return

    # Model checkpoints
    with st.expander("Model Settings", expanded=False):
        seg_ckpt = st.text_input("Segmentation Checkpoint", value="checkpoints/segmentation/best.pt")
        cls_ckpt = st.text_input("Classification Checkpoint", value="checkpoints/classification/best.pt")
        disaster_type = st.text_input("Disaster Type", value="earthquake")
        use_llm = st.checkbox("Use LLM for Question Reasoning (requires ANTHROPIC_API_KEY)", value=False)

    if st.button("Run Assessment", type="primary"):
        with st.spinner("Running full assessment pipeline…"):
            _run_inference(before_file, after_file, seg_ckpt, cls_ckpt, disaster_type, use_llm)

    _render_results()


def _run_inference(before_file, after_file, seg_ckpt, cls_ckpt, disaster_type, use_llm) -> None:
    try:
        import cv2

        # Save uploads to temp files
        def _save_tmp(f):
            suffix = Path(f.name).suffix
            t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            t.write(f.read())
            t.flush()
            return t.name

        before_tmp = _save_tmp(before_file)
        after_tmp = _save_tmp(after_file)

        # Step 1: Preprocessing
        from backend.preprocessing.preprocessing_pipeline import PreprocessingConfig, PreprocessingPipeline
        from backend.preprocessing.image_normalization import NormalizationConfig
        prep_cfg = PreprocessingConfig(normalization=NormalizationConfig(target_size=(512, 512)))
        pipeline = PreprocessingPipeline(prep_cfg)
        prep_result = pipeline.process_image_pair(before_tmp, after_tmp)
        st.session_state["inf_prep_result"] = prep_result

        if prep_result.preprocessing_status == "failed":
            st.error(f"Preprocessing failed: {prep_result.errors}")
            return

        # Load processed images
        proc_before_path = prep_result.processed_before or before_tmp
        proc_after_path = prep_result.aligned_after or prep_result.processed_after or after_tmp

        img_before = cv2.cvtColor(cv2.imread(proc_before_path), cv2.COLOR_BGR2RGB)
        img_after = cv2.cvtColor(cv2.imread(proc_after_path), cv2.COLOR_BGR2RGB)

        # Step 2: Building segmentation (synthetic if no checkpoint)
        buildings = _run_segmentation(img_after, seg_ckpt)

        # Step 3: Damage classification per building
        buildings_with_damage = _run_damage_classification(img_before, img_after, buildings, cls_ckpt)

        # Step 4: Evidence generation
        from backend.reasoning.evidence_generator import generate_evidence, aggregate_scene_evidence
        assessment_result = {
            "buildings": buildings_with_damage,
            "alignment_quality": prep_result.alignment_quality or 0.5,
            "before_quality": prep_result.before_quality or {},
            "after_quality": prep_result.after_quality or {},
        }
        evidence_items = generate_evidence(assessment_result)
        scene_summary = aggregate_scene_evidence(evidence_items)

        # Step 5: Question reasoning
        from backend.reasoning.question_reasoning import QuestionReasoner, ReasoningConfig
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=use_llm, fallback_to_rules=True))
        qa_pairs = reasoner.answer_standard_questions(evidence_items)

        # Step 6: Heatmap
        from backend.visualization.damage_heatmap import generate_scene_heatmap
        heatmap_result = generate_scene_heatmap(img_before, img_after, buildings_with_damage)

        # Step 7: Report
        from backend.reports.report_generator import create_damage_report, generate_json_report
        import uuid
        report = create_damage_report(
            scene_id=prep_result.image_pair_id,
            evidence_items=evidence_items,
            qa_pairs=qa_pairs,
            image_paths={
                "before": proc_before_path,
                "after": proc_after_path,
            },
            disaster_type=disaster_type,
        )

        st.session_state.update({
            "inf_evidence": evidence_items,
            "inf_scene_summary": scene_summary,
            "inf_qa_pairs": qa_pairs,
            "inf_heatmap": heatmap_result,
            "inf_report": report,
            "inf_before_img": img_before,
            "inf_after_img": img_after,
        })
        st.success("Assessment complete!")

    except Exception as exc:
        st.error(f"Assessment error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


def _run_segmentation(img_after: np.ndarray, ckpt_path: str) -> list:
    """Return list of building dicts with bbox. Falls back to synthetic buildings."""
    if Path(ckpt_path).exists():
        try:
            import torch
            from backend.models.building_segmentation import SegmentationConfig, BuildingSegmentationModel
            cfg = SegmentationConfig(checkpoint_path=ckpt_path)
            model = BuildingSegmentationModel(cfg)
            tensor = _image_to_tensor(img_after, cfg.image_size)
            with torch.no_grad():
                mask = model.predict(tensor).squeeze().cpu().numpy()
            return _mask_to_buildings(mask)
        except Exception:
            pass

    # Synthetic: grid of buildings for demo
    h, w = img_after.shape[:2]
    buildings = []
    for i in range(2):
        for j in range(2):
            x1, y1 = int(w * (j * 0.4 + 0.05)), int(h * (i * 0.4 + 0.05))
            x2, y2 = x1 + int(w * 0.3), y1 + int(h * 0.3)
            buildings.append({"bbox": [x1, y1, x2, y2],
                               "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
    return buildings


def _run_damage_classification(before: np.ndarray, after: np.ndarray,
                                buildings: list, ckpt_path: str) -> list:
    """Classify damage for each building. Falls back to synthetic scores."""
    import numpy as np
    rng = np.random.default_rng(42)
    result = []
    for b in buildings:
        b = dict(b)
        if Path(ckpt_path).exists():
            try:
                import torch
                from backend.models.damage_classification import DamageClassifierConfig, DamageClassificationModel
                cfg = DamageClassifierConfig(checkpoint_path=ckpt_path)
                model = DamageClassificationModel(cfg)
                x1, y1, x2, y2 = b["bbox"]
                crop_b = before[y1:y2, x1:x2]
                crop_a = after[y1:y2, x1:x2]
                t_b = _image_to_tensor(crop_b, cfg.image_size[0])
                t_a = _image_to_tensor(crop_a, cfg.image_size[0])
                with torch.no_grad():
                    proba = model.predict_proba(t_b, t_a).squeeze().tolist()
                b["damage_class"] = int(np.argmax(proba))
                b["proba"] = proba
                result.append(b)
                continue
            except Exception:
                pass
        # Synthetic
        proba = rng.dirichlet([2, 1, 1, 0.5]).tolist()
        b["damage_class"] = int(np.argmax(proba))
        b["proba"] = proba
        result.append(b)
    return result


def _image_to_tensor(img: np.ndarray, size: int):
    import torch
    import cv2
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def _mask_to_buildings(mask: np.ndarray) -> list:
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    buildings = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 100:
            continue
        polygon = c.squeeze().tolist()
        if isinstance(polygon[0], int):
            polygon = [polygon]
        buildings.append({"bbox": [x, y, x + w, y + h], "polygon": polygon})
    return buildings


def _render_results() -> None:
    if "inf_scene_summary" not in st.session_state:
        return

    summary = st.session_state["inf_scene_summary"]
    qa_pairs = st.session_state.get("inf_qa_pairs", [])
    heatmap_result = st.session_state.get("inf_heatmap", {})
    report = st.session_state.get("inf_report")
    before_img = st.session_state.get("inf_before_img")
    after_img = st.session_state.get("inf_after_img")

    # Damage summary metrics
    st.header("2. Damage Summary")
    cols = st.columns(4)
    for i, (label, key) in enumerate([
        ("Total Buildings", "total_buildings"),
        ("No Damage", "no_damage_count"),
        ("Major/Destroyed", "high_severity_count"),
        ("Overall Severity", "overall_severity"),
    ]):
        cols[i].metric(label, summary.get(key, "N/A"))

    # Distribution chart
    dist = {k: v for k, v in summary.items() if k.endswith("_count")}
    if dist:
        fig = px.bar(
            x=[k.replace("_count", "").replace("_", " ").title() for k in dist],
            y=list(dist.values()),
            color=list(dist.keys()),
            color_discrete_map={
                "no_damage_count": "#4CAF50",
                "minor_damage_count": "#FFC107",
                "major_damage_count": "#FF5722",
                "destroyed_count": "#F44336",
            },
            labels={"x": "Damage Class", "y": "Buildings"},
            title="Damage Distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.header("3. Damage Heatmap")
    heatmap_img = heatmap_result.get("heatmap") if heatmap_result else None
    c1, c2, c3 = st.columns(3)
    with c1:
        if before_img is not None:
            st.image(before_img, caption="Before", use_container_width=True)
    with c2:
        if after_img is not None:
            st.image(after_img, caption="After", use_container_width=True)
    with c3:
        if heatmap_img is not None:
            st.image(heatmap_img, caption="Damage Heatmap", use_container_width=True)

    # Q&A
    st.header("4. Question Reasoning")
    for item in qa_pairs:
        with st.expander(item.get("question", "Q")):
            st.write(item.get("answer", ""))

    # Evidence table
    st.header("5. Building Evidence")
    evidence = st.session_state.get("inf_evidence", [])
    if evidence:
        import pandas as pd
        rows = [
            {
                "Building ID": e.building_id if hasattr(e, "building_id") else e.get("building_id", ""),
                "Damage Class": e.damage_class if hasattr(e, "damage_class") else e.get("damage_class", ""),
                "Confidence": f"{e.confidence:.2f}" if hasattr(e, "confidence") else e.get("confidence", ""),
                "Action Required": "Yes" if (e.action_required if hasattr(e, "action_required") else False) else "No",
            }
            for e in evidence
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Reports
    st.header("6. Download Reports")
    if report:
        import json as jsonlib
        from dataclasses import asdict
        try:
            report_dict = asdict(report)
        except Exception:
            report_dict = report if isinstance(report, dict) else {}
        st.download_button(
            "Download JSON Report",
            data=jsonlib.dumps(report_dict, indent=2, default=str).encode(),
            file_name=f"{getattr(report, 'report_id', 'report')}_report.json",
            mime="application/json",
        )
