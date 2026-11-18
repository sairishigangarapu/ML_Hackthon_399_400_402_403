# AGENTS.md — PathSnap Master Project Context & Continuation Blueprint

> **Repository:** `https://github.com/sairishigangarapu/ML_Hackthon_399_400_402_403.git`  
> **Course:** PES University — UE24CS352A: Machine Learning (RR Campus)  
> **Challenge:** PathSnap: Distributed Map-Matching & Route Reconstruction  
> **Team ID:** 399_400_402_403  
> **Purpose of this file:** Complete single-source-of-truth project specification, empirical dataset audit, mathematical formulations, directory contract, and checkpoint protocol so any agent or teammate can resume work immediately across sessions with zero context loss.

---

## 1. Problem Statement & Core Objective

GPS receivers suffer from urban-canyon multipath interference, signal dropouts, drift, and irregular sampling intervals. As a result, raw GPS trajectories appear to jump across buildings or parallel streets.

**PathSnap** is an end-to-end distributed machine learning & sequential decoding system that reconstructs realistic, topologically continuous vehicle routes by matching noisy GPS observations (`.track`) to directed road-network segments (`.nodes`, `.arcs`), served through a **two-laptop architecture**:
- **Laptop 1 (FastAPI Backend):** Hosts trained ML models, HMM/Viterbi route decoder, road-network data, and strictly server-side evaluation against reserved ground truth.
- **Laptop 2 (Evaluator Web UI):** Interactive Leaflet + **OpenStreetMap (free public tile tier)** interface listing the 10 evaluation trips, dispatching trip inference requests to Laptop 1's FastAPI endpoint, and rendering raw GPS points, road context, predicted route, and returned evaluation metrics.

---

## 2. Official Dataset & Empirical Audit (Verified from Raw Data)

### 2.1 Partitions & Strict Leakage Prevention
| Partition | Trips | Local Path | Permitted Use |
|---|---:|---|---|
| **Training** | 80 | `../Student folder/train/` | EDA, preprocessing fit, feature engineering, 5-fold `GroupKFold(trip_id)` CV, model training, Bayesian HPO, HMM parameter tuning |
| **Student Testing** | 10 | `../Student folder/student_test/` | Development-stage model comparison and diagnostics ONLY (never fit scalers/models/HPO on it) |
| **Evaluator Testing** | 10 | Provided at final evaluation on Laptop 1 (`PATHSNAP_EVAL_DIR`) | Final live evaluation ONLY — untouched during development; `.route` ground truth strictly server-side |

### 2.2 Raw File Schema (per trip folder `<trip_id>/`)
- `<trip_id>.track`: 3 whitespace-separated headerless columns — `longitude`, `latitude`, `timestamp_sec` (nominally 1 Hz).
- `<trip_id>.nodes`: 2 whitespace-separated headerless columns — `longitude`, `latitude` (0-based row index = `node_id`).
- `<trip_id>.arcs`: 2 headerless integer columns — `from_node_id`, `to_node_id` (0-based row index = `arc_id`).
- `<trip_id>.route`: 1 headerless integer column — chronological sequence of ground-truth `arc_id`s traversed by the vehicle.

### 2.3 Critical Empirical Findings Already Verified
1. **Scale:**
   - `train`: 80 trips, **208,417** GPS points (min=361, med=1,529, max=19,313); nodes 299–155,003; arcs 649–307,392; route arcs 59–2,266.
   - `student_test`: 10 trips, **19,160** GPS points (min=262, med=1,420, max=3,958); nodes 1,770–26,635; arcs 3,577–56,027; route arcs 95–1,323.
   - Zero NaNs, zero negative/zero timestamps, zero out-of-bounds arc indices, zero missing files.
2. **Global Coordinates -> Local Metric Projection (AEQD):**
   - Trips span worldwide (`lon ∈ [-100.42°, +137.24°]`, `lat ∈ [-34.64°, +61.27°]`). Ordinary Euclidean distance in degrees is invalid, and a single UTM zone cannot span Earth.
   - **Solution:** Per-trip local Azimuthal Equidistant (East-North-Up tangent plane) projection centered at each trip's centroid `(lon0, lat0)` in meters.
3. **92.2% Bidirectional Directed Arcs:**
   - Two-way streets are encoded as opposite directed pairs `(u -> v)` and `(v -> u)` with identical perpendicular distance to any GPS point.
   - **Consequence:** Distance-only scoring hits an ~50% ceiling! Multi-scale smoothed vehicle **heading alignment** ($\cos\Delta\theta, \sin\Delta\theta$) and **directed graph connectivity** (`prev.to_node == curr.from_node`) are decisive signals.
4. **High-Noise & Correlated Spatial Error Trips (Empirical Motivation for NN-GLS):**
   - `train/00000054` (19,313 pts, Amsterdam): 169.2 km cumulative GPS jitter over a 31.4 km route; median distance to ANY road arc is 83.6m (138.5m to reference route).
   - `train/00000033` (5,844 pts, Poland): 122.4m median distance to reference route.
   - `train/00000002` (7,243 pts, Germany): max GPS-to-route distance 569.5m.
   - `student_test`: 98.56% within 25m, 99.86% within 50m, 99.98% within 100m, 100.00% within 200m.
5. **GPS Time Gaps & Zero-Length Arcs:**
   - 96.50% (train) and 98.62% (student_test) of intervals are exactly 1 s; however `train` contains gaps up to **153,064 s (~42.5 h)** and `student_test` up to **210 s**. Trajectory smoothing and HMM transitions MUST detect and handle $\Delta t > \tau_{\text{gap}}$.
   - Zero-length arcs exist (`100` in train, `6` in student_test) and must be guarded against divide-by-zero in bearing/projection calculations.
6. **Revisited Arcs & Reference Alignment:**
   - `.route` is 99.96% directed-connected (13 breaks / 31,256 transitions in train; 1 / 3,467 in student_test).
   - 9/80 train trips and 1/10 student_test trips revisit arcs (loops). Therefore, mapping each GPS point $t$ to its reference `arc_id` in `.route` uses **Monotonic Dynamic Programming alignment** along `.route` rather than greedy nearest-arc lookup.

---

## 3. Complete Model Suite & Scientific Formulations

To strictly exceed the PDF requirement of **at least 5 distinct models across multiple algorithm families** AND deliver all **4 user-specified core paradigms**, we implement, cross-validate (`GroupKFold(5)` on 80 train trips), and evaluate **8 distinct models** on identical candidate sets and features:

### Group A — Multi-Family Classical & Deep Baselines
1. **`M01_LogisticRegression`** (*Linear & Probabilistic*): L2-regularized `LogisticRegression` on standardized candidate features — interpretable linear baseline with native probabilities.
2. **`M02_RandomForest`** (*Bagging Ensemble*): `RandomForestClassifier` — non-linear axis-aligned ensemble baseline.
3. **`M03_LightGBM_Untuned`** (*Gradient Boosting*): Default `LGBMClassifier` — pre-HPO boosting reference required for untuned-vs-tuned Bayesian HPO comparison (Report Section 4).
4. **`M04_MLP_BCE`** (*Standard Neural Network*): PyTorch residual MLP trained with standard i.i.d. binary/listwise cross-entropy — crucial deep-learning ablation baseline to isolate gains from **NN-GLS** and **RLCD**.

### Group B — The Four Core Requested Systems
5. **`M05_Model1_BayesianOpt_LightGBM_HMM`** (**Model 1: HMM + Bayesian Optimisation**):
   - **Stage 1 (Bayesian Classifier HPO):** Optuna TPE Bayesian optimization over LightGBM hyperparameters (`num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `n_estimators`) using 5-fold `GroupKFold(trip_id)` CV Macro F1 / Log-Loss objective on the 80 train trips.
   - **Stage 2 (HMM + Viterbi Sequential Decoder + Bayesian HMM HPO):** Hidden states = candidate arcs $a \in \mathcal{C}_t$. Emission log-probabilities combine calibrated classifier probabilities with geometric emission likelihood. Transition log-probabilities follow Newson & Krumm (2009) using true directed shortest-path network distance $d_{\text{route}}(a_{t-1}, a_t)$ (via cached bounded Dijkstra on `.arcs`) vs great-circle displacement $d_{\text{GC}}(z_{t-1}, z_t)$: $\log P(a_t \mid a_{t-1}) = -|d_{\text{route}} - d_{\text{GC}}| / \beta - \lambda_{\text{turn}}$, with explicit disconnected-segment penalty and adaptive GPS-gap scaling ($\Delta t > \tau_{\text{gap}}$). Decoder hyperparameters $(\alpha_{\text{emit}}, \beta, \gamma_{\text{disc}}, \tau_{\text{gap}})$ are Bayesian-tuned strictly on the 80 train trips.
6. **`M06_Model2_RLCD`** (**Model 2: RLCD — Reinforcement Learning for Calibrated Decisions**):
   - Stochastic listwise decision policy $\pi_\theta(a \mid s_t) = \text{softmax}_k(f_\theta(x_{t,k}))$ over the candidate set $\mathcal{C}_t$ (with recurrent/previous-step contextual features).
   - Trained via policy-gradient RL (**REINFORCE / GRPO group-relative baseline**) using a **strictly proper scoring rule reward** (negative multi-class **Brier Score**: $R_{\text{Brier}}(p_t, y_t) = 1 - \sum_{k=1}^K (p_{t,k} - \mathbb{I}[k = y_t])^2$) combined with a **topological route-continuity reward** $R_{\text{cont}}(a_{t-1}, a_t)$ (rewarding directed-graph-connected consecutive choices) and an explicit **differentiable ECE calibration penalty**.
   - Avoids the overconfidence trap of 0/1 RLVR rewards, producing sharply calibrated probabilities (low Log-Loss, low Brier Score, low ECE) that pair seamlessly with both standalone argmax decoding and HMM/Viterbi.
7. **`M07_Model3_NN_GLS`** (**Model 3: NN-GLS — Neural Networks with Generalized Least Squares**, *Zhan & Datta, JASA 2025*):
   - Embeds a neural network mean function $f_\theta(X)$ inside a **Nearest-Neighbor Gaussian Process (NNGP)** spatial covariance model to account for strong spatiotemporal autocorrelation of GPS errors along trajectories.
   - Constructs the sparse **Vecchia / NNGP Cholesky pre-whitening operator** $L^{-1} = D^{-1/2}(I - B)$ over each observation's $m$ nearest preceding spatiotemporal trajectory neighbors using an exponential spatial-temporal kernel $C_\phi(\Delta s, \Delta t) = \sigma^2 \exp(-\Delta s / \rho_s - \Delta t / \rho_t) + \tau^2 \delta_{ij}$.
   - Trains $f_\theta$ with the **GLS decorrelated loss** $\mathcal{L}_{\text{GLS}}(\theta) = \| L^{-1}(y - \hat{p}_\theta(X)) \|_2^2 + \lambda_{\text{cal}} \mathcal{L}_{\text{CE}}$ (mathematically equivalent to a GNN spatial-whitening layer).
   - At inference time, performs **NNGP Spatial Residual Kriging / Smoothing** over each trajectory's local neighborhood graph to propagate correlated spatial corrections.
8. **`M08_Model4_NN_GLS_RLCD`** (**Model 4: NN-GLS + RLCD Hybrid**):
   - **Stage 1:** Spatially-debiased backbone pretrained via **NN-GLS** (NNGP Vecchia GLS whitening).
   - **Stage 2:** Post-trained with **RLCD** calibrated policy gradient (strictly proper Brier reward + sequential route-continuity reward + KL divergence anchor $\beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \| \pi_{\text{NN-GLS}})$ preserving spatial generalization while calibrating decision probabilities).
   - Evaluated both **standalone** and **coupled with HMM/Viterbi**, verifying that spatial debiasing + proper-scoring calibration directly boost sequential route reconstruction!

---

## 4. Ordered Project Directory Structure (Deliverable-Aligned)

```text
PathSnap/
├── AGENTS.md                           # Master project context & continuation blueprint (this file)
├── README.md                           # Installation, execution & two-laptop deployment guide
├── requirements.txt                    # Pinned Python dependencies
├── .gitignore                          # Excludes .venv, caches, secrets
├── agents/                             # Durable session tracking, checkpoints & decision logs
│   ├── AGENTS.md                       # Mirror of master context
│   ├── HANDOFF_PROTOCOL.md             # Session resume protocol
│   ├── STATE.md                        # Live milestone checklist
│   ├── CHECKPOINT.json                 # Machine-readable pipeline state & artifact manifest
│   └── DECISIONS.md                    # Architectural & scientific decision log
├── 01_data_processing/                 # Raw inspection, validation, AEQD CRS projection, EDA
├── 02_feature_engineering/             # Spatial index, Top-K candidates, monotonic labeling, features
├── 03_model_training/                  # Base models + NN-GLS + RLCD + NN-GLS+RLCD trainers
├── 04_hyperparameter_optimisation/     # Optuna TPE Bayesian HPO (classifier & HMM params)
├── 05_hmm_viterbi/                     # HMM + Viterbi decoder, bounded Dijkstra, gap handling
├── 06_model_evaluation/                # Unified metrics, comparison tables, calibration, plots
├── 07_visualisations/                  # Folium interactive HTML maps (OSM) + PNG plots/captures
│   ├── maps/
│   └── plots/
├── 08_api/                             # FastAPI backend (Laptop 1) — strict server-side GT isolation
├── 09_ui/                              # Evaluator Web UI (Laptop 2) — Leaflet + OpenStreetMap
├── 10_deployment/                      # Two-laptop launch scripts, .env.example, E2E test suite
├── artifacts/                          # Saved weights, scalers, configs, HMM params, feature schema
├── results/                            # CSVs, JSON experiment logs, HPO trials, metric summaries
├── report/                             # Consolidated Technical Report PDF (Sections 2–8)
└── theory_and_viva/                    # Deep theory explanations & Viva Q&A preparation guides
```

---

## 5. Required Evaluation Metrics & Deliverables Checklist

### 5.1 Metrics Logged for Every Model (Zero Fabrication)
- Candidate-level: Accuracy, Precision, Recall, Macro F1-score, Binary/Listwise Log-Loss, Brier Score, Expected Calibration Error (ECE), Confusion Matrix.
- Observation-level & Route-level: **Observation-level Top-1 Accuracy** (primary selection metric paired with Route Continuity & Log-Loss), **Candidate Miss Rate**, **Route Continuity** (fraction of consecutive non-gap predictions that stay on the same arc or follow a valid directed graph path), **Sequence-level Arc F1 / Precision / Recall**, **Training Time**, **Inference Time (ms/trip)**.
- Epoch-wise training & validation loss curves **ONLY** for epoch-trained neural models (`MLP_BCE`, `RLCD`, `NN-GLS`, `NN-GLS+RLCD`) — NEVER fabricated for LR, RF, or LightGBM.

### 5.2 Deliverable 1 (Public GitHub Repository)
- All source code across `01_` through `10_`, saved model artifacts & configs, `results/` CSV/JSON logs, interactive Folium HTML maps + generation code, `README.md`, `requirements.txt`, `agents/`, `theory_and_viva/`, and `report/`. No API keys or secrets committed.

### 5.3 Deliverable 2 (Consolidated Technical Report PDF — `report/PathSnap_Technical_Report.pdf`)
Must follow exact section structure of `Deliverables.pdf`:
- Section 2.2: Dataset Understanding & EDA (missing values, trajectory plots, road-network visualisations, >=1 Folium map, explicit partition tags).
- Section 3.1: Overall Model Comparison Table (all 8 models, exact columns).
- Section 3.2: Compulsory Questions (A.1–A.2, B.5–B.6, C.7–C.9) answered individually for **EVERY** model, grounded in actual numbers.
- Section 3.3: Compulsory Model Visualisations (CV Macro F1, Student-test Top-1 & Macro F1, Log-Loss, Confusion Matrices, genuine epoch loss curves).
- Section 4: Bayesian Hyperparameter Optimisation (Q1–Q6 + untuned vs tuned comparison table).
- Section 5: HMM and Viterbi Route Reconstruction (Q1–Q6 + Classifier-only vs Classifier+HMM comparison + failure-case analysis).
- Section 6: Geographical Visualisation Using Folium (actual rendered OSM Folium map views showing GPS, road network, reference route, predicted route, legend, layer controls).
- Section 7: API, UI and Deployment (real UI screenshots + Laptop 1 FastAPI CMD/terminal request logs).
- Section 8: Academic Integrity & AI Assistance Disclosure.

---

## 6. Two-Laptop Deployment Contract

- **Laptop 1 (`08_api/main.py` via Uvicorn on `0.0.0.0:8000`):**
  - Reads `PATHSNAP_EVAL_DIR` (defaults to `../Student folder/student_test` in dev; switchable to reserved `evaluator_test` at evaluation time).
  - Endpoints: `GET /health`, `GET /models`, `GET /trips`, `POST /predict` (`{"trip_id": "...", "model_name": "..."}`).
  - Prediction pipeline reads ONLY `.track`, `.nodes`, `.arcs`.
  - Server-side evaluator independently reads `.route` (if available) to compute accuracy/F1/continuity metrics and NEVER returns ground-truth labels/coordinates to the UI.
  - Emits verbose live terminal logs (`[PATHSNAP-API]`) for evaluator CMD verification.
- **Laptop 2 (`09_ui/` served via static HTTP server on port `3000`):**
  - Configurable API Base URL (`http://<LAPTOP_1_IP>:8000`).
  - Displays the 10 evaluation trip cards + model selector (defaulting to best model, with ability to compare Model 1–4).
  - Uses **Leaflet.js + OpenStreetMap free tile tier** (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`), zero API key required.

---

## 7. Hardware & Runtime Notes
- Machine: Intel Core i9-13900H (20 threads), 15 GB RAM, **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, CUDA 13.2)**.
- Python: 3.12.3 inside `PathSnap/.venv`.
- Always check `agents/CHECKPOINT.json` and `agents/STATE.md` before executing or skipping a stage!
