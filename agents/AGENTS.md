# AGENTS.md — PathSnap Master Context & Continuation Blueprint (Model 4: NN-GLS + RLCD)

> **Repository:** `https://github.com/sairishigangarapu/ML_Hackthon_399_400_402_403.git`  
> **Course:** PES University — UE24CS352A: Machine Learning (RR Campus)  
> **Challenge:** PathSnap: Distributed Map-Matching & Route Reconstruction  
> **Team ID:** 399_400_402_403 (4-member split)  
> **Active Implementation Scope:** **Model 4 — Pure `NN-GLS + RLCD`** (zero HMM/Viterbi, zero Bayesian HPO; those belong to Teammate 1's Model 1).

---

## 1. Team Split & Scope Contract
| Member | Model | Notes |
|---|---|---|
| Teammate 1 | **Model 1:** HMM + Viterbi + Bayesian HPO | `04_hyperparameter_optimisation/` & `05_hmm_viterbi/` reserved |
| Teammate 2 | **Model 2:** Standalone RLCD | Plugs into same candidate/feature schema |
| Teammate 3 | **Model 3:** Standalone NN-GLS | Plugs into same candidate/feature schema |
| **This Workspace** | **Model 4: Pure NN-GLS + RLCD** | Full end-to-end pipeline, trained artifacts, Folium OSM maps, FastAPI backend, Leaflet OSM UI, PDF report & Viva prep |

---

## 2. Verified Empirical Dataset Audit
- **Partitions:** `train` = 80 trips (**208,417** GPS obs); `student_test` = 10 trips (**19,160** GPS obs); `evaluator_test` = 10 reserved trips (untouched; configured at eval time via `PATHSNAP_EVAL_DIR`).
- **Missing data:** 0 missing files, 0 NaNs, 0 negative/zero timestamps, 0 out-of-bounds arcs.
- **Projection:** Trip-centered local WGS-84 **Azimuthal Equidistant (AEQD)** metric plane in meters (trips span globally: `lon ∈ [-100.42°, +137.24°]`, `lat ∈ [-34.64°, +61.27°]`).
- **92.2% Bidirectional Directed Arcs:** Naive nearest-arc Top-1 accuracy is only **50.14%** (`train`) / **54.71%** (`student_test`). Multi-scale gap-aware heading alignment and cross-candidate self-attention resolve this ambiguity.
- **High-Noise Correlated Drift Trips:** `train/00000054` (19,313 obs, 138.5 m median offset, 51.4% Top-10 miss) and `train/00000033` (5,844 obs, 122.4 m offset, 65.8% Top-10 miss) prove strong spatial-temporal multipath autocorrelation — motivating **NN-GLS** Vecchia precision whitening and balanced window sampling (`max_windows_per_trip = 140` during training).
- **Candidate Recall:** Subdivided-arc `cKDTree` (40 m step, Top-$K = 10$) achieves **0.17% Candidate Miss Rate** (99.83% recall!) on `student_test`.
- **GPS Gaps & Loops:** Max gap = 153,064 s (`train`) / 210 s (`student_test`); strictly gated at $\Delta t > 5\text{ s}$. Ground-truth `.route` alignment uses Monotonic DP (exclusively for training & server-side evaluation) to handle 9/80 train + 1/10 student_test loop trips.

---

## 3. Model 4 (`NN-GLS + RLCD`) Technical Specification
1. **Features (`02_feature_engineering/feature_extractor.py`):** 28 strictly observable features (zero `.route` leakage) across 5 groups: Geometric/Proximity (6), Kinematic & Multi-Scale Heading (8), Listwise Candidate Competition (6), Local Road Topology & 1-Step Context (5), Temporal/Gap Context (3).
2. **Architecture (`03_model_training/nngls_core.py` — `NNGLSSpatioTemporalNet`):**
   - Input Residual Encoder (`28 -> 128`)
   - 2× `NNGPKrigingGraphLayer`: Intra-Step Multi-Head Cross-Candidate Self-Attention + Zhan & Datta NNGP Kriging-Weighted Spatial Aggregation (lags $r = 1\dots 5$ aligned by physical `arc_id`) + Forward/Backward Directed Road-Network Message Passing (`step_conn[t, i, j]`, gap-gated)
   - Contrastive Listwise Head + Integrated Differentiable NNGP Spatial Residual Kriging + Learnable Calibration Temperature $\tau = \exp(\text{log\_temp})$ (so Stage 2 RLCD calibrates the exact post-kriging probabilities with zero train/inference mismatch!).
3. **Stage 1 Training (NN-GLS):** Exact fixed-lag Vecchia / NNGP pre-whitened Generalized Least Squares loss $\|D^{-1/2}(I - B)(y - p_\theta)\|_2^2 + 0.60 \cdot \mathcal{L}_{\text{precision-weighted CE}}$ ($m=5$, $\rho_s=45\text{ m}$, $\rho_t=6\text{ s}$, $\sigma^2=0.85$, $\tau^2=0.15$).
4. **Stage 2 Post-Training (RLCD — `03_model_training/rlcd_core.py`):** Group-Relative Policy Optimization (GRPO, $G = 8$ parallel rollouts) combining confidence-aware asymmetric proper Brier rewards, directed road-network continuity rewards (`+1.0` connected / `-0.5` bidirectional U-turn flip), direct multi-class Brier + Log-Score losses, differentiable Soft-ECE penalty, and KL divergence anchor $\beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \| \pi_{\text{NN-GLS}})$.
5. **Inference:** Raised-cosine overlap-averaged sliding windows ($W=64$, stride $32$) $\to$ calibrated softmax $\to$ direct `argmax` decision (pure neural, zero HMM/Viterbi).

---

## 4. Two-Laptop Deployment & Strict Integrity Contract
- **Laptop 1 (`08_api/main.py`):** FastAPI + Uvicorn (`0.0.0.0:8000`). Prediction phase strictly uses `load_route=False`. Server-side evaluation reads `.route` independently and NEVER returns ground-truth labels/coordinates to the UI (`ground_truth_exposed: false`).
- **Laptop 2 (`09_ui/`):** Static Leaflet.js + **OpenStreetMap free public tile tier** (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`) dashboard on port 3000.
- **Checkpoints:** Always inspect `agents/CHECKPOINT.json` and `agents/STATE.md` when resuming across sessions.
