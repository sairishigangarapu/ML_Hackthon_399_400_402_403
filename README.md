# PathSnap: Distributed Map-Matching & Route Reconstruction — Model 4 (NN-GLS + RLCD)

**PES University — UE24CS352A: Machine Learning Hackathon | Team 399_400_402_403**

This repository implements **Model 4 (`NN-GLS + RLCD`)** — a pure end-to-end **Geostatistical Graph Neural Network + Calibrated Reinforcement Learning** map-matching system (zero HMM/Viterbi dependency) paired with a two-laptop **FastAPI + Leaflet/OpenStreetMap (free tier)** deployment architecture.

### Team Division of Labor (4 Members)
- **Model 1** — HMM + Viterbi + Bayesian Hyperparameter Optimisation *(Teammate 1)*
- **Model 2** — Standalone RLCD *(Teammate 2)*
- **Model 3** — Standalone NN-GLS *(Teammate 3)*
- **Model 4 (This Implementation)** — **Hybrid NN-GLS + RLCD** *(Zhan & Datta Vecchia/NNGP spatial covariance whitening + Kriging-Graph layers + GRPO strictly proper Brier & directed road-continuity RLCD post-training)*

---

## Ordered Directory Structure (Deliverable-Aligned)

| Folder | Contents |
|---|---|
| [`agents/`](agents/) | Multi-session continuation blueprint (`AGENTS.md`), `HANDOFF_PROTOCOL.md`, live `STATE.md`, `CHECKPOINT.json`, and `DECISIONS.md` |
| [`01_data_processing/`](01_data_processing/) | Raw `.track`/`.nodes`/`.arcs`/`.route` I/O, WGS-84 local Azimuthal Equidistant (AEQD) projection, validation & full EDA |
| [`02_feature_engineering/`](02_feature_engineering/) | Subdivided-arc `cKDTree` Top-10 candidate generator (0.17% test miss rate), monotonic DP reference alignment, and 28 domain features |
| [`03_model_training/`](03_model_training/) | Pure Model 4 implementation: `nngls_core.py` (Vecchia/NNGP whitening + Kriging-Graph layers), `rlcd_core.py` (GRPO Brier + continuity + Soft-ECE + KL anchor), and 5-fold `GroupKFold(trip_id)` trainer `train_model4_nngls_rlcd.py` |
| [`04_hyperparameter_optimisation/`](04_hyperparameter_optimisation/) | Reserved in team directory layout for Model 1 (Teammate 1) |
| [`05_hmm_viterbi/`](05_hmm_viterbi/) | Reserved in team directory layout for Model 1 (Teammate 1) |
| [`06_model_evaluation/`](06_model_evaluation/) | Self-contained metrics (`metrics.py`: Top-1 Acc, Macro F1, Route Continuity, Seq Arc F1, Log-Loss, Brier, ECE, Confusion Matrix) & publication-grade plot generator (`generate_eval_plots.py`) |
| [`07_visualisations/`](07_visualisations/) | Interactive **Folium OpenStreetMap (free tier)** maps (`maps/`) and diagnostic figures (`plots/`) |
| [`08_api/`](08_api/) | **FastAPI** backend (Laptop 1) with live CMD inference logging and strict server-side ground-truth isolation (`load_route=False` during inference) |
| [`09_ui/`](09_ui/) | Evaluator Web UI (Laptop 2) using **Leaflet.js + OpenStreetMap free public tile tier** (zero paid API keys) |
| [`10_deployment/`](10_deployment/) | Two-laptop launch scripts (`start_api_laptop1.sh`, `start_ui_laptop2.sh`), `.env.example`, and automated 10-trip E2E verification (`test_api_e2e.py`) |
| [`artifacts/`](artifacts/) | Saved `model4_nngls_rlcd.pt` weights, `scaler_model4_nngls_rlcd.json`, `feature_schema.json`, and `model4_nngls_rlcd_config.json` |
| [`results/`](results/) | Complete experimental CSVs, JSON summaries, CV fold logs, epoch histories, and API E2E verification logs |
| [`report/`](report/) | Consolidated Technical Report PDF (`PathSnap_Model4_NNGLS_RLCD_Report.pdf`) and generator script |
| [`theory_and_viva/`](theory_and_viva/) | Deep mathematical explanations of NN-GLS, RLCD, their synergy, and 24+ Viva Q&As |

---

## Quick Start & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Reproducible, Seed = 42)
```bash
python 01_data_processing/run_eda.py
python 02_feature_engineering/feature_extractor.py
python 03_model_training/train_model4_nngls_rlcd.py
python 06_model_evaluation/generate_eval_plots.py
python 07_visualisations/generate_folium_maps.py
python 10_deployment/test_api_e2e.py
python report/generate_report_pdf.py
```

### 3. Two-Laptop Deployment
- **Laptop 1 (Model & FastAPI Backend):**
  ```bash
  # For final evaluation, point PATHSNAP_EVAL_DIR to the 10 reserved evaluator trips:
  PATHSNAP_EVAL_DIR="/path/to/evaluator_test" ./10_deployment/start_api_laptop1.sh
  ```
- **Laptop 2 (Evaluator Web UI — OpenStreetMap Free Tier):**
  ```bash
  ./10_deployment/start_ui_laptop2.sh
  # Open http://localhost:3000 and set API URL to http://<LAPTOP_1_LAN_IP>:8000
  ```
