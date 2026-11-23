"""
EAGLE-DAR Dataset Module

Provides HuggingFace dataset loading, inspection, and preparation
for the xBD (xView2 Building Damage) dataset.
"""

from .hf_dataset_loader import (
    load_xbd_dataset,
    load_qa_dataset,
    get_image_pair,
    get_building_annotations,
    get_damage_class_names,
    download_and_cache,
)
from .dataset_inspector import (
    inspect_dataset,
    count_samples_per_split,
    count_damage_classes,
    check_image_dimensions,
    check_annotation_coverage,
    compute_class_weights,
    generate_inspection_report,
)
from .dataset_preparation import (
    DatasetConfig,
    XBDDataset,
    prepare_dataset,
    create_data_loaders,
    xbd_collate_fn,
    create_synthetic_dataset,
    validate_dataset_split,
    save_split_manifest,
)

__all__ = [
    # Loader
    "load_xbd_dataset",
    "load_qa_dataset",
    "get_image_pair",
    "get_building_annotations",
    "get_damage_class_names",
    "download_and_cache",
    # Inspector
    "inspect_dataset",
    "count_samples_per_split",
    "count_damage_classes",
    "check_image_dimensions",
    "check_annotation_coverage",
    "compute_class_weights",
    "generate_inspection_report",
    # Preparation
    "DatasetConfig",
    "XBDDataset",
    "prepare_dataset",
    "create_data_loaders",
    "xbd_collate_fn",
    "create_synthetic_dataset",
    "validate_dataset_split",
    "save_split_manifest",
]
