# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — UE24CS352A: Machine Learning Hackathon (Team 399_400_402_403)**

PathSnap reconstructs realistic, topologically valid vehicle trajectories from noisy GPS observations onto directed road networks using multi-family machine learning, **Bayesian Hyperparameter Optimisation**, **HMM + Viterbi decoding**, **NN-GLS (Neural Networks with Generalized Least Squares)**, and **RLCD (Reinforcement Learning for Calibrated Decisions)**, deployed across a two-laptop **FastAPI + OpenStreetMap/Leaflet** architecture.

## Repository Structure (Deliverable Order)

| Directory | Purpose |
|---|---|
| [`agents/`](agents/) | Multi-session agent tracking, machine-readable checkpoints (`CHECKPOINT.json`), `STATE.md`, and architectural `DECISIONS.md` |
| [`01_data_processing/`](01_data_processing/) | Raw `.track`/`.nodes`/`.arcs`/`.route` parsers, validation, local Azimuthal Equidistant (AEQD) projection, and EDA |
| [`02_feature_engineering/`](02_feature_engineering/) | Spatial KD-tree candidate generation, monotonic DP ground-truth alignment, and 28 spatial/kinematic/topological features |
| [`03_model_training/`](03_model_training/) | Training pipelines for 8 models: Logistic Regression, Random Forest, LightGBM, MLP-BCE, **Model 1 (Bayesian-Opt LightGBM + HMM)**, **Model 2 (RLCD)**, **Model 3 (NN-GLS)**, **Model 4 (NN-GLS + RLCD)** |
| [`04_hyperparameter_optimisation/`](04_hyperparameter_optimisation/) | Optuna TPE Bayesian optimisation over 5-fold `GroupKFold(trip_id)` CV (classifier + HMM parameters) |
| [`05_hmm_viterbi/`](05_hmm_viterbi/) | Log-domain HMM + Viterbi decoder with bounded Dijkstra road-network distances and GPS-gap handling |
| [`06_model_evaluation/`](06_model_evaluation/) | Unified evaluation suite, calibration analysis (ECE/Brier/Log-Loss), confusion matrices, and comparison charts |
| [`07_visualisations/`](07_visualisations/) | Interactive **Folium (OpenStreetMap free tiles)** maps (`maps/`) and diagnostic figures (`plots/`) |
| [`08_api/`](08_api/) | **FastAPI** backend (Laptop 1) with live inference logging and strict server-side ground-truth isolation |
| [`09_ui/`](09_ui/) | Evaluator-facing Web UI (Laptop 2) using **Leaflet.js + OpenStreetMap** free tile tier |
| [`10_deployment/`](10_deployment/) | Two-laptop launch scripts, `.env.example`, and end-to-end API verification tests |
| [`artifacts/`](artifacts/) | Saved trained model weights, feature scalers, HMM parameters, and inference schema |
| [`results/`](results/) | Experimental CSVs, JSON metrics, HPO trial logs, and summary tables |
| [`report/`](report/) | Consolidated Technical Report (`PathSnap_Technical_Report.pdf`) |
| [`theory_and_viva/`](theory_and_viva/) | Deep mathematical theory notes and structured Viva Q&A preparation |

## Quick Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [`AGENTS.md`](AGENTS.md) for the complete technical context and session continuation guide.
