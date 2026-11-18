# PathSnap Live Agent State — Model 4 (NN-GLS + RLCD)

**Last Updated:** 2026-11-18 11:15 IST  
**Active Branch:** `M4`  
**Scope:** Model 4 — Pure **NN-GLS + RLCD** (Team 399_400_402_403, Member 4; zero HMM/Viterbi or Bayesian HPO)  
**Hardware:** Intel i9-13900H (20 threads), 15 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 13.2)

## 12-Stage Work Process Completion
- [x] **Stage 1 — Understand the Problem:** Audited all 5 official PathSnap PDFs + team 4-model split.
- [x] **Stage 2 — Explore & Understand Raw Dataset (`01_data_processing/run_eda.py`):** Full inventory of 80 `train` (208,417 obs) & 10 `student_test` (19,160 obs) trips; 0 missing values/files; discovered 92.2% bidirectional directed arcs, global WGS-84 coordinates, time gaps up to 153,064 s, and high-noise correlated drift trips (`00000054`, `00000033`); generated 3 EDA figures + 3 interactive OpenStreetMap Folium maps.
- [x] **Stage 3 — Independent Data Processing & Feature Engineering (`02_feature_engineering/`):** Trip-centered WGS-84 AEQD metric projection, subdivided-arc `cKDTree` Top-10 candidate generator (**0.17% Candidate Miss Rate** on `student_test`!), Monotonic DP ground-truth alignment (training & server-side eval only), 28 strictly observable spatial/kinematic/listwise/topological/gap features, directed `step_conn` graph matrices, and fixed-lag Vecchia/NNGP whitening operators.
- [x] **Stages 4–7 (Model 4 Pipeline — `03_model_training/`):** Trained & cross-validated **Model 4 (NN-GLS + RLCD)** via strict 5-fold `GroupKFold(trip_id)` CV on the 80 train trips (fold-internal `StandardScaler`, zero leakage), logged Stage 1 (NN-GLS) vs Stage 2 (NN-GLS + RLCD) progression & genuine epoch curves, and refit final model on all 80 train trips.
  - **5-Fold Train CV Macro F1:** `0.9204 ± 0.0525` (Hit-Conditioned Top-1: `88.13%`, Route Continuity: `98.71%`)
  - **Held-Out `student_test` (10 trips, 19,160 obs):**
    - Observation-level Top-1 Accuracy: **`91.50%`** (vs `54.71%` naive nearest-arc baseline; **+36.79%** gain!)
    - Candidate Macro F1-score: **`0.9530`**
    - Route Continuity (Path): **`99.24%`** (`96.61%` exact 1-hop directed continuity — **zero HMM!**)
    - Sequence Arc F1: **`0.9215`** (boosted from Stage 1's `0.9180` by RLCD continuity reward!)
    - Log-Loss: **`0.2604`** | Multi-Class Brier: **`0.1230`** | ECE: **`0.0351`**
    - Inference Latency: **`39.33 ms / trip`** (CUDA RTX 4060)
- [x] **Stage 8 — Export Trained Model & Artifacts (`artifacts/`):** Exported `model4_nngls_rlcd.pt`, `scaler_model4_nngls_rlcd.json`, `feature_schema.json`, `model4_nngls_rlcd_config.json`.
- [x] **Stage 9 — FastAPI Backend (`08_api/`):** Laptop 1 server (`main.py`, `inference_engine.py`) with `load_route=False` during inference, strict server-side GT evaluation (`ground_truth_exposed=false`), configurable `PATHSNAP_EVAL_DIR`, and verbose live CMD logs.
- [x] **Stage 10 — Evaluator Web UI (`09_ui/`):** Laptop 2 Leaflet.js + **OpenStreetMap free public tile tier** UI displaying 10 trip cards, RLCD confidence-colored GPS markers, predicted route polylines, KPI cards, and request audit log.
- [x] **Stage 11 — Integration & E2E Testing (`10_deployment/test_api_e2e.py`):** Automated verification across all 10 `student_test` trips with strict ground-truth non-leakage assertion.
- [x] **Stage 12 — Technical Report & Viva Prep (`report/`, `theory_and_viva/`):** Consolidated PDF report (`PathSnap_Model4_NNGLS_RLCD_Report.pdf`) + 4 deep theory & Viva Q&A guides + 10 interactive OpenStreetMap Folium HTML maps + 8 publication-grade figures.

## Ready for Live Evaluation
- Laptop 1: `PATHSNAP_EVAL_DIR=/path/to/evaluator_test ./10_deployment/start_api_laptop1.sh`
- Laptop 2: `./10_deployment/start_ui_laptop2.sh` (open `http://localhost:3000`)
