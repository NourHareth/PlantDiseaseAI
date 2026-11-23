"""Central place for file locations and tunable constants.

All paths are absolute and derived from this file's location, so the app and
scripts work no matter which directory they are launched from.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _work_dir() -> Path:
    """Where large, machine-specific files live (datasets, training runs).

    Resolution order: the PLANTDISEASE_WORKDIR environment variable, then
    ``"workdir"`` in the git-ignored ``local_paths.json``, then the project folder.
    Keeping it outside cloud-synced folders (OneDrive, Dropbox) avoids uploading
    tens of thousands of images.
    """
    if os.environ.get("PLANTDISEASE_WORKDIR"):
        return Path(os.environ["PLANTDISEASE_WORKDIR"])
    local = PROJECT_ROOT / "local_paths.json"
    if local.exists():
        workdir = json.loads(local.read_text(encoding="utf-8")).get("workdir")
        if workdir:
            return Path(workdir)
    return PROJECT_ROOT


WORK_DIR = _work_dir()
RAW_DATA_DIR = WORK_DIR / "datasets" / "raw"

# Model artifacts (everything needed to run inference lives together)
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "plant_disease_model.keras"
CLASS_NAMES_FILENAME = "class_names.json"
MODEL_INFO_FILENAME = "model_info.json"
OOD_DETECTOR_FILENAME = "ood_detector.npz"  # optional: Mahalanobis "unfamiliar image" detector

# Content and presentation
DATA_DIR = PROJECT_ROOT / "data"
DISEASE_INFO_PATH = DATA_DIR / "disease_info.json"
ASSETS_DIR = PROJECT_ROOT / "assets"
STYLES_PATH = ASSETS_DIR / "styles.css"
LOGO_PATH = ASSETS_DIR / "logo.svg"
SAMPLES_DIR = ASSETS_DIR / "samples"
SAMPLE_IMAGE_PATH = SAMPLES_DIR / "field_tomato_leaf.png"

# Dataset split manifests (small CSVs, versioned in git for reproducibility)
SPLITS_DIR = DATA_DIR / "splits"
PLANTVILLAGE_MANIFEST = SPLITS_DIR / "plantvillage_tomato.csv"
PLANTDOC_MANIFEST = SPLITS_DIR / "plantdoc_tomato.csv"

# Outputs written by scripts
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
RUNS_DIR = WORK_DIR / "runs"

# Grad-CAM: an 8x8 feature map at 128 px input. The final conv block
# ("out_relu") is only 4x4, which is too coarse to be useful on a leaf.
GRADCAM_LAYER = "block_13_expand_relu"

# Refuse absurdly large images (decompression-bomb protection).
MAX_IMAGE_PIXELS = 50_000_000

# Confidence bands used for user-facing messaging (probability, 0..1).
HIGH_CONFIDENCE = 0.80
MODERATE_CONFIDENCE = 0.60
