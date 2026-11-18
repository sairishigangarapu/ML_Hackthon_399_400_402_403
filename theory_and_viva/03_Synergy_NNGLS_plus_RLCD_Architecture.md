# Theory Deep Dive 03: Why NN-GLS + RLCD (Model 4) Works End-to-End Without an HMM

---

## 1. Complementary Error Decomposition

Map-matching errors arise from **three distinct mathematical pathologies**:

| Pathology in GPS Map-Matching | Failure Mode of Naive Classifiers | Solved in Model 4 By |
|---|---|---|
| **1. Spatial-Temporal Multipath Autocorrelation** (e.g., Trips `00000054`, `00000033`) | Correlated error bursts dominate i.i.d. gradients; single-point features ignore local spatial drift | **Stage 1: NN-GLS** — Vecchia NNGP whitening $L^{-1}=D^{-1/2}(I-B)$ + Kriging-weighted GNN + inference residual kriging |
| **2. Bidirectional Arc & Topological Discontinuity** (92.2% two-way roads, stops, intersections) | Independent step-wise predictions flicker between `(u->v)` and `(v->u)` or jump across disconnected streets | **NN-GLS Graph Layers + Stage 2 RLCD** — Cross-candidate self-attention, bidirectional road-graph message passing, and GRPO road-network continuity reward ($+1.0$ connected / $-0.5$ U-turn flip) |
| **3. Miscalibrated / Overconfident Probabilities** | Standard cross-entropy or 0/1 RLVR overcommits to 99.9% confidence even at ambiguous junctions | **Stage 2: RLCD** — Strictly proper Brier + Log scoring rules, asymmetric bluff penalty, differentiable Soft-ECE, learnable temperature $\tau$, and KL anchor to NN-GLS |

---

## 2. End-to-End Model 4 Architecture & Tensor Flow

```text
Raw Trip (.track, .nodes, .arcs — ZERO .route leakage!)
   │
   ▼  [01_data_processing]
Local WGS-84 Azimuthal Equidistant (AEQD) Projection (meters, centered at trip centroid)
   │
   ▼  [02_feature_engineering]
Subdivided-Arc cKDTree (40m max segment step) ──► Top-K = 10 Candidate Arcs per GPS obs
   │                                              (0.17% miss rate on student_test!)
   ├─► 28 Domain Features X ∈ R^(T × 10 × 28)
   ├─► Directed Road Connectivity Matrix step_conn ∈ R^(T × 10 × 10)  (gap-masked >5s)
   └─► Exact Vecchia/NNGP Whitening (b_w ∈ R^(T×5), d_inv_sqrt ∈ R^T, align_k ∈ Z^(T×5×10))
   │
   ▼  [03_model_training: NNGLSSpatioTemporalNet]
Sliding Contiguous Windows (B, W=48 train / 64 eval, K=10, F=28)
   ├─► Input Linear(28→128) + LayerNorm + SiLU + ResidualBlock(128)
   ├─► 2 × NNGPKrigingGraphLayer:
   │     ├── Intra-Step Multi-Head Self-Attention across K=10 competing arcs (4 heads)
   │     ├── NNGP Kriging-Weighted Spatial Aggregation over lags r=1..5 (arc_id-aligned)
   │     ├── Forward Directed Road-Graph Aggregation (t-1 → t via step_conn > 0)
   │     ├── Backward Directed Road-Graph Aggregation (t+1 → t via step_conn > 0)
   │     └── Gated Residual Fusion + ResidualBlock(128)
   └─► Contrastive Listwise Head: [h, h - mean_K(h)] → MLP → logits / exp(log_temp)
   │
   ▼  [Two-Stage Training on 80 Train Trips (5-Fold GroupKFold by trip_id)]
   ├── Stage 1 (NN-GLS): Vecchia pre-whitened GLS loss + d_t^{-1}-weighted CE
   └── Stage 2 (RLCD):   GRPO (G=8 rollouts) with Brier + Continuity reward
                         + Direct Multi-Class Brier + Soft-ECE + KL(π_θ || π_NN-GLS)
   │
   ▼  [Pure Neural Inference (Zero HMM / Zero Viterbi)]
Raised-Cosine Overlap-Averaged Windows (W=64, stride=32)
   ──► NNGP Spatial Residual Kriging (krige_strength = 0.22)
   ──► Calibrated Softmax π_θ(k | s_t)
   ──► Direct Argmax Decision: â_t = argmax_k π_θ(k | s_t)
```

---

## 3. Why Sliding-Window Overlap + NNGP Kriging Replaces HMM Latency Bottlenecks

- Classical HMM + Viterbi requires $O(T \cdot K^2)$ shortest-path Dijkstra queries at inference time, which can take hundreds of milliseconds to seconds on dense urban graphs (up to 307k arcs!).
- Model 4 replaces runtime Viterbi search with **learned bidirectional road-graph message passing + NNGP spatial kriging**, executing on GPU in **tens of milliseconds per trip** while jointly optimizing **Top-1 Accuracy, Route Continuity, AND Statistical Calibration (ECE/Brier/Log-Loss)**!
- Even better, because Model 4's output probabilities $\pi_\theta(k \mid s_t)$ are strictly calibrated by RLCD, they can be consumed directly via `argmax` (standalone Model 4) OR plugged seamlessly into downstream decision/dispatch engines that require trustworthy confidence bounds!

---

## 4. Strict Data-Leakage & Deployment Isolation Guarantees

1. **Partition Separation:** `train` (80 trips) is used exclusively for scaling, 5-fold `GroupKFold(trip_id)` CV, and final refitting. `student_test` (10 trips) is touched **only once** for final reporting. Reserved `evaluator_test` (10 trips) is never touched during development.
2. **Fold-Internal Scaling:** `StandardScaler` mean/std are computed strictly on the 4 training folds inside every CV split — never across all 80 trips simultaneously during CV.
3. **Server-Side Ground-Truth Lock (`08_api/inference_engine.py`):**
   - `load_trip(..., load_route=False)` ensures `.route` is physically absent from memory during prediction.
   - Server-side metric evaluation reads `.route` in a completely isolated post-inference step and never returns ground-truth arc IDs or coordinates to Laptop 2's UI.
