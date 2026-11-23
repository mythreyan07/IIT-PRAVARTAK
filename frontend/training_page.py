"""
EAGLE-DAR — Model Training Page

Train building segmentation and damage classification models.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path


def render_training_page() -> None:
    st.title("Model Training")
    st.caption(
        "Train the building segmentation and Siamese damage classification models "
        "on the prepared xBD dataset."
    )

    tab_seg, tab_cls, tab_eval = st.tabs([
        "Segmentation", "Damage Classification", "Evaluation"
    ])

    with tab_seg:
        _render_segmentation_training()
    with tab_cls:
        _render_classification_training()
    with tab_eval:
        _render_evaluation()


def _render_segmentation_training() -> None:
    st.header("Building Segmentation")
    st.markdown(
        "Trains a U-Net model to detect building footprints in satellite images."
    )

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50)
        lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1,
                             value=1e-4, format="%.6f")
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
        patience = st.number_input("Early Stopping Patience", min_value=1, max_value=50, value=10)

    image_size = st.selectbox("Image Size", [256, 512, 1024], index=0)
    checkpoint_dir = st.text_input("Checkpoint Directory", value="checkpoints/segmentation")

    if st.button("Start Segmentation Training"):
        _run_training("segmentation", dict(
            epochs=int(epochs), lr=float(lr),
            batch_size=int(batch_size), patience=int(patience),
            image_size=int(image_size), checkpoint_dir=checkpoint_dir,
        ))

    _show_training_history("segmentation")


def _render_classification_training() -> None:
    st.header("Damage Classification (Siamese)")
    st.markdown(
        "Trains a Siamese ResNet50 network to classify building damage from before/after crop pairs."
    )

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50, key="cls_epochs")
        lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1,
                             value=1e-4, format="%.6f", key="cls_lr")
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16, key="cls_bs")
        backbone = st.selectbox("Backbone", ["resnet50", "resnet18", "resnet34"])

    checkpoint_dir = st.text_input("Checkpoint Directory", value="checkpoints/classification",
                                   key="cls_ckpt")

    if st.button("Start Classification Training"):
        _run_training("classification", dict(
            epochs=int(epochs), lr=float(lr),
            batch_size=int(batch_size), backbone=backbone,
            checkpoint_dir=checkpoint_dir,
        ))

    _show_training_history("classification")


def _render_evaluation() -> None:
    st.header("Model Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        seg_ckpt = st.text_input("Segmentation Checkpoint Path", value="checkpoints/segmentation/best.pt")
    with col2:
        cls_ckpt = st.text_input("Classification Checkpoint Path", value="checkpoints/classification/best.pt")

    if st.button("Evaluate Models"):
        with st.spinner("Evaluating…"):
            try:
                results = {}
                if Path(seg_ckpt).exists():
                    from backend.models.building_segmentation import SegmentationConfig, BuildingSegmentationModel
                    from backend.models.model_evaluation import evaluate_segmentation_model
                    cfg = SegmentationConfig(checkpoint_path=seg_ckpt)
                    model = BuildingSegmentationModel(cfg)
                    st.info("Segmentation model loaded. Connect a DataLoader to evaluate.")
                    results["segmentation"] = {"status": "model loaded"}

                if Path(cls_ckpt).exists():
                    from backend.models.damage_classification import DamageClassifierConfig, DamageClassificationModel
                    cfg = DamageClassifierConfig(checkpoint_path=cls_ckpt)
                    model = DamageClassificationModel(cfg)
                    st.info("Classification model loaded. Connect a DataLoader to evaluate.")
                    results["classification"] = {"status": "model loaded"}

                if not results:
                    st.warning("No checkpoint files found at the specified paths.")
                else:
                    st.session_state["eval_results"] = results
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

    if eval_res := st.session_state.get("eval_results"):
        st.json(eval_res)


def _run_training(model_type: str, config_dict: dict) -> None:
    with st.spinner(f"Training {model_type} model… (this may take a while)"):
        try:
            from backend.models.model_training import TrainingConfig, ModelTrainer
            from backend.dataset.dataset_preparation import create_synthetic_dataset, DatasetConfig, create_data_loaders

            st.info(
                "No real dataset loaded — running a quick sanity-check training loop on synthetic data. "
                "Load the xBD dataset on the Dataset page for real training."
            )

            ds_config = DatasetConfig(
                image_size=config_dict.get("image_size", 64),
                batch_size=config_dict.get("batch_size", 4),
            )
            synthetic = create_synthetic_dataset(n_samples=20, image_size=ds_config.image_size)
            loaders = create_data_loaders(synthetic, ds_config)

            if model_type == "segmentation":
                from backend.models.building_segmentation import UNet, SegmentationConfig, BuildingSegmentationModel
                import torch
                model_cfg = SegmentationConfig(image_size=ds_config.image_size)
                model_obj = BuildingSegmentationModel(model_cfg)
                model_nn = model_obj.model
            else:
                from backend.models.damage_classification import SiameseDamageClassifier, DamageClassifierConfig, DamageClassificationModel
                import torch
                model_cfg = DamageClassifierConfig(image_size=(ds_config.image_size, ds_config.image_size))
                model_obj = DamageClassificationModel(model_cfg)
                model_nn = model_obj.model

            t_cfg = TrainingConfig(
                model_type=model_type,
                epochs=min(config_dict.get("epochs", 2), 2),  # cap at 2 for demo
                lr=config_dict.get("lr", 1e-4),
                batch_size=ds_config.batch_size,
                checkpoint_dir=config_dict.get("checkpoint_dir", f"checkpoints/{model_type}"),
                early_stopping_patience=config_dict.get("patience", 3),
            )
            trainer = ModelTrainer(t_cfg, model_nn, loaders["train"], loaders["val"])
            result = trainer.train()
            st.session_state[f"training_result_{model_type}"] = result
            st.success(f"Training complete — best val metric: {result.best_val_metric:.4f}")
        except Exception as exc:
            st.error(f"Training error: {exc}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


def _show_training_history(model_type: str) -> None:
    result = st.session_state.get(f"training_result_{model_type}")
    if not result or not result.training_history:
        return
    history = result.training_history
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", mode="lines+markers"))
    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)
