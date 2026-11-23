# EAGLE-DAR — Disaster Assessment and Response

Evidence-Augmented Geospatial Learning Engine for Disaster Assessment and Response.

A satellite image analysis system that detects building damage from before/after image pairs using OpenCV preprocessing, deep learning segmentation, and Siamese damage classification.

---

## System Architecture

```
HUGGING FACE DATASET
       │
       ▼
DATASET INSPECTION       ← backend/dataset/dataset_inspector.py
       │
       ▼
DATASET PREPARATION      ← backend/dataset/dataset_preparation.py
       │
       ▼
OPENCV PREPROCESSING     ← backend/preprocessing/preprocessing_pipeline.py
       │
       ▼
ANNOTATION TRANSFORM     ← backend/preprocessing/image_normalization.py
       │
       ▼
TRAIN / VAL / TEST       ← backend/dataset/dataset_preparation.py
       │
       ▼
MODEL TRAINING           ← backend/models/model_training.py
       │
       ▼
MODEL EVALUATION         ← backend/models/model_evaluation.py
       │
       ▼
SAVE CHECKPOINTS         ← checkpoints/
       │
       ▼
USER IMAGE UPLOAD        ← Streamlit (Damage Assessment page)
       │
       ▼
AUTO PREPROCESSING       ← backend/preprocessing/preprocessing_pipeline.py
       │
       ▼
ALIGNMENT VERIFICATION   ← backend/preprocessing/image_alignment.py
       │
       ▼
BUILDING DETECTION       ← backend/models/building_segmentation.py (U-Net)
       │
       ▼
DAMAGE CLASSIFICATION    ← backend/models/damage_classification.py (Siamese ResNet50)
       │
       ▼
QUESTION REASONING       ← backend/reasoning/question_reasoning.py
       │
       ▼
EVIDENCE GENERATION      ← backend/reasoning/evidence_generator.py
       │
       ▼
DAMAGE HEATMAP           ← backend/visualization/damage_heatmap.py
       │
       ▼
PDF AND JSON REPORTS     ← backend/reports/report_generator.py
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## Dataset

### Primary: xView2 / xBD Building Damage Dataset

**Source:** HuggingFace — `kshitijrajsharma/xview2-xbd`  
**Mirror:** `hannan022/xview2-xbd`  
**License:** CC BY-NC-SA 4.0  
**Size:** ~24 GB, 8,043 image pairs  
**Splits:** train (2,840), test (933), hold (933), tier3 (3,380)

**Download instructions:**

```python
from datasets import load_dataset
from huggingface_hub import login

# Public dataset — no token needed for public access
# For restricted datasets: login(token="hf_...")

dataset = load_dataset("kshitijrajsharma/xview2-xbd", split="train")
```

Or use the Dataset Inspection page in the app.

**Do not commit** the downloaded dataset files. They are excluded by `.gitignore`.

### QA Reasoning Dataset

**Source:** `DakshJ27/xbd-damage-qa`  
**Size:** 1,663 Q&A pairs  
**Use:** Evidence-based question reasoning module

---

## Damage Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | no_damage | Building intact |
| 1 | minor_damage | Partial damage, structurally sound |
| 2 | major_damage | Severe damage, structurally compromised |
| 3 | destroyed | Complete destruction |

---

## Module Reference

### backend/preprocessing/
| File | Purpose |
|------|---------|
| `image_loader.py` | Load, validate, preserve images |
| `image_quality.py` | Blur/brightness/contrast/resolution checks |
| `image_alignment.py` | ORB+RANSAC / ECC image registration |
| `image_normalization.py` | Resize, normalise, annotation transforms |
| `preprocessing_pipeline.py` | Full pipeline orchestration |

### backend/dataset/
| File | Purpose |
|------|---------|
| `hf_dataset_loader.py` | Load xBD from HuggingFace |
| `dataset_inspector.py` | Analyse dataset statistics |
| `dataset_preparation.py` | Train/val/test splits, DataLoaders |

### backend/models/
| File | Purpose |
|------|---------|
| `building_segmentation.py` | U-Net building footprint segmentation |
| `damage_classification.py` | Siamese ResNet50 damage classifier |
| `model_training.py` | Training loops, early stopping, checkpoints |
| `model_evaluation.py` | IoU, accuracy, F1, confusion matrix |

### backend/reasoning/
| File | Purpose |
|------|---------|
| `evidence_generator.py` | Structured evidence from assessment results |
| `question_reasoning.py` | LLM/rule-based Q&A about damage |

### backend/visualization/
| File | Purpose |
|------|---------|
| `damage_heatmap.py` | Damage heatmap overlays |

### backend/reports/
| File | Purpose |
|------|---------|
| `report_generator.py` | PDF and JSON report generation |

---

## Tests

```bash
# All tests
python -m pytest tests/ -v

# Module-specific
python -m pytest tests/preprocessing/ -v
python -m pytest tests/dataset/ -v
python -m pytest tests/models/ -v
python -m pytest tests/reasoning/ -v
python -m pytest tests/visualization/ -v
python -m pytest tests/reports/ -v
```

All tests use synthetic data — no dataset download required.

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `HF_TOKEN` | HuggingFace access token | No (public datasets) |
| `ANTHROPIC_API_KEY` | Claude API for LLM reasoning | No (falls back to rules) |

---

## Configuration

Key configuration objects (all dataclasses, all fields have defaults):

- `PreprocessingConfig` — paths, quality thresholds, alignment, normalisation
- `QualityThresholds` — blur/brightness/contrast/resolution thresholds
- `AlignmentConfig` — ORB params, RANSAC threshold, ECC iterations
- `NormalizationConfig` — target_size, mean, std, channel_order
- `DatasetConfig` — train/val/test ratios, batch_size, num_workers
- `SegmentationConfig` — backbone, image_size, checkpoint_path
- `DamageClassifierConfig` — backbone, num_classes, image_size
- `TrainingConfig` — epochs, lr, early_stopping, mixed_precision
- `ReasoningConfig` — use_llm, model, fallback_to_rules
- `ReportConfig` — output_dir, include_images, pdf_title

Preprocessing parameters are saved with model checkpoints to ensure
training/inference consistency.

---

## Citation

xBD Dataset:
```
Gupta, R. et al. (2019). Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery.
CVPR Workshops. https://xview2.org
```
