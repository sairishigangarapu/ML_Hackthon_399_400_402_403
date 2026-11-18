"""Central configuration for PathSnap pipeline."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Raw data directories (read-only inputs; never modified)
RAW_DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", WORKSPACE_ROOT / "Student folder"))
TRAIN_DIR = RAW_DATA_ROOT / "train"
STUDENT_TEST_DIR = RAW_DATA_ROOT / "student_test"

# Configurable evaluation directory for Laptop 1 FastAPI server
# Defaults to student_test for development; override with PATHSNAP_EVAL_DIR=/path/to/evaluator_test
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", STUDENT_TEST_DIR))

# Project directories (deliverable order)
DIR_DATA_PROC = PROJECT_ROOT / "01_data_processing"
DIR_FEAT_ENG = PROJECT_ROOT / "02_feature_engineering"
DIR_MODEL_TRAIN = PROJECT_ROOT / "03_model_training"
DIR_HPO = PROJECT_ROOT / "04_hyperparameter_optimisation"
DIR_HMM = PROJECT_ROOT / "05_hmm_viterbi"
DIR_EVAL = PROJECT_ROOT / "06_model_evaluation"
DIR_VIS = PROJECT_ROOT / "07_visualisations"
DIR_MAPS = DIR_VIS / "maps"
DIR_PLOTS = DIR_VIS / "plots"
DIR_API = PROJECT_ROOT / "08_api"
DIR_UI = PROJECT_ROOT / "09_ui"
DIR_DEPLOY = PROJECT_ROOT / "10_deployment"
DIR_ARTIFACTS = PROJECT_ROOT / "artifacts"
DIR_CACHE = DIR_ARTIFACTS / "cache"
DIR_RESULTS = PROJECT_ROOT / "results"
DIR_REPORT = PROJECT_ROOT / "report"
DIR_AGENTS = PROJECT_ROOT / "agents"
DIR_THEORY = PROJECT_ROOT / "theory_and_viva"

# Global reproducibility & pipeline parameters
RANDOM_SEED = 42
N_CV_FOLDS = 5
TOP_K_CANDIDATES = 10
CANDIDATE_RADIUS_M = 180.0
GAP_THRESHOLD_S = 5.0

# Ensure runtime output dirs exist
for _d in [DIR_MAPS, DIR_PLOTS, DIR_ARTIFACTS, DIR_CACHE, DIR_RESULTS, DIR_REPORT]:
    _d.mkdir(parents=True, exist_ok=True)
