# PathSnap Viva Master Preparation — 32 High-Probability Questions & Model Answers

> Organized directly around **Section 6 (Viva and Demonstration)** of `Assessment and Plagiarism Policy.pdf`, tailored specifically to **Model 4 (`NN-GLS + RLCD`)**.

---

## Part A: Dataset Structure, Preprocessing & Data Quality

### Q1. How is the raw PathSnap dataset organized, and why can't you join `.track` and `.nodes` row-by-row?
**Answer:** Each trip folder `<trip_id>/` has 4 headerless whitespace-separated files: `.track` (`lon, lat, ts`, 1 Hz GPS observations, median ~1,529 rows), `.nodes` (`lon, lat`, 0-based `node_id`, median ~9,608 rows), `.arcs` (`from_node_id, to_node_id`, 0-based `arc_id`, median ~19,526 rows), and `.route` (ordered ground-truth `arc_id` sequence, median ~280 rows). Rows in `.track` are time-indexed vehicle GPS measurements, while `.nodes`/`.arcs` define the entire regional road graph — so they must be linked geometrically via spatial projection and candidate search, never by row index.

### Q2. Why can't you compute Euclidean distances directly on `(longitude, latitude)`? Why didn't you pick a single UTM zone?
**Answer:** Longitude and latitude are angular degrees on the WGS-84 ellipsoid: $1^\circ$ of latitude is always $\approx 111.1\text{ km}$, whereas $1^\circ$ of longitude shrinks by $\cos(\text{lat})$ (e.g., at $60^\circ\text{N}$ in Russia/Scandinavia, $1^\circ$ lon is only $\approx 55.6\text{ km}$ — a 2× distortion!). Furthermore, our EDA revealed that the 80 training trips span globally (`lon ∈ [-100.42°, +137.24°]`, `lat ∈ [-34.64°, +61.27°]`), so no single UTM zone covers all trips. Therefore, we project every trip into a **trip-centered local WGS-84 Azimuthal Equidistant (AEQD)** East-North-Up plane (`+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m`), preserving true geodesic distances and bearings in meters everywhere on Earth.

### Q3. What were the 4 most important data-quality discoveries from your EDA, and how did each shape your pipeline?
**Answer:**
1. **92.2% Bidirectional Directed Arcs:** Two-way roads are stored as opposite directed pairs `(u->v)` and `(v->u)` with *identical* perpendicular distance. That's why naive nearest-arc snapping gets only **50.14% (train) / 54.71% (student_test)** Top-1 accuracy! This forced us to engineer multi-scale smoothed heading features (`cos_heading_smooth`, `cos_heading_wide`, `has_reverse_pair`, `heading_rank_near`) and cross-candidate self-attention.
2. **High-Noise Correlated Drift Trips (`00000054`, `00000033`, `00000002`):** Trip `00000054` (19,313 points in Amsterdam) has 138.5 m median error to `.route` and 169 km of cumulative jitter over a 31 km route; Trip `00000033` has 122.4 m median error. This directly motivated **NN-GLS** to decorrelate spatially autocorrelated error bursts instead of treating observations as i.i.d.!
3. **Large GPS Time Gaps:** Although 96.5% (train) / 98.6% (student_test) of intervals are 1 s, `train` contains gaps up to **153,064 s (~42.5 hours)** and `student_test` up to **210 s**. We strictly gate trajectory smoothing, NNGP spatial covariance, and road-graph message passing at $\Delta t > 5\text{ s}$.
4. **Zero-Length Arcs & Revisited Loops:** There are 100 zero-length arcs in `train` and 6 in `student_test` (guarded against divide-by-zero bearings), and 9/80 train + 1/10 student_test trips revisit arcs — motivating **Monotonic DP route alignment** instead of naive nearest-arc lookup.

### Q4. How did you generate ground-truth per-observation labels from `.route` when `.route` has ~280 arcs and `.track` has ~1,500 GPS points?
**Answer:** Because 9 train trips and 1 student_test trip revisit streets (loops/roundtrips), snapping each GPS point to its globally closest arc in `.route` mismatches outgoing vs. return segments. Instead, `02_feature_engineering/route_aligner.py` runs **Monotonic Dynamic Programming** aligning observation $t \in \{0, \dots, T-1\}$ to route index $j_t \in \{0, \dots, M-1\}$, minimizing perpendicular distance + heading mismatch while enforcing forward chronological progression ($j_t \ge j_{t-1}$ with tiny 1-arc jitter backtrack tolerance). Importantly, this runs **only** for training labels and strictly server-side evaluation — never during prediction.

### Q5. How does your candidate generator handle multi-kilometer highway arcs without missing them?
**Answer:** Road arc lengths range from 0 m up to **9,297 m**. A naive midpoint KD-tree misses a 4 km highway segment if the GPS point is near its endpoint (2 km from the midpoint!). Our `ArcSpatialIndex` subdivides every arc longer than 40 m into proxy points spaced every 40 m, builds a fast `scipy.spatial.cKDTree`, queries the nearest proxies + 65 m ball, deduplicates `arc_id`s, computes **exact perpendicular point-to-segment distance**, and retains Top-$K = 10$ directed arcs (covering both directions of ~5 physical streets). This achieved a **0.17% Candidate Miss Rate** on `student_test` (99.83% recall!).

---

## Part B: Feature Engineering

### Q6. Explain your 28 engineered features and why you chose those 5 groups.
**Answer:**
1. **Geometric / Proximity (6):** `dist_m`, `log_dist_m` (compresses multipath outliers), `proj_ratio` ($u \in [0,1]$ along segment), `signed_lateral_offset` (cross-product sign captures right/left-hand lane offset!), `arc_len_m`, `log_arc_len_m`.
2. **Kinematic & Multi-Scale Heading (8):** `gps_speed_mps`, `smooth_speed_mps`, `is_stationary` ($< 0.8\text{ m/s}$), `cos/sin_heading_inst`, `cos_heading_smooth` ($\pm 3\text{ s}$), `cos_heading_wide` ($\pm 7\text{ s}$), `heading_progress_dot`. Multi-scale gap-aware headings survive traffic-light stops and high-frequency jitter to break the 92.2% bidirectional tie!
3. **Listwise Competition (6):** `dist_rank_norm`, `dist_diff_best`, `dist_ratio_best`, `has_reverse_pair`, `heading_rank_near` (heading rank among candidates within 15 m of closest arc), `local_cand_density` (flags dense urban intersections).
4. **Road Topology & Local Context (5):** `node_out_degree`, `node_in_degree`, `prev_step_same_arc`, `prev_step_connected` (1-hop / 2-hop reachability from $\mathcal{C}_{t-1}$), `next_step_connected`.
5. **Temporal & Gap Context (3):** `dt_prev_log`, `dt_next_log`, `is_gap_boundary`.

### Q7. Do any of your 28 features leak ground-truth information?
**Answer:** Zero! Every single feature is computed strictly from `.track`, `.nodes`, and `.arcs` inside `extract_trip_features(trip)` where `trip.route` is `None`.

---

## Part C: Model 4 — NN-GLS Theory & Implementation

### Q8. What is NN-GLS, and who proposed it?
**Answer:** **NN-GLS (Neural Networks with Generalized Least Squares)** was introduced by **Wentao Zhan & Abhirup Datta** (Johns Hopkins University, *JASA* 2025). It bridges deep neural networks with model-based geostatistics (Gaussian Processes) by replacing standard i.i.d. OLS/CE loss with a **Generalized Least Squares (GLS)** objective weighted by a sparse **Nearest-Neighbor Gaussian Process (NNGP / Vecchia)** spatial precision matrix, and proving that the resulting architecture is mathematically equivalent to a Kriging-weighted Graph Neural Network.

### Q9. How do you avoid $O(N^3)$ matrix inversion in NN-GLS?
**Answer:** Using the **Vecchia / NNGP factorization** $\tilde{\Sigma}^{-1} = (I - B)^\top D^{-1}(I - B)$. For each point $t$, we only invert a tiny $m \times m$ covariance matrix ($m = 5$) over its $m$ nearest preceding spatiotemporal neighbors inside the same contiguous trajectory block, yielding Kriging weights $b_t = C_{\mathcal{N},\mathcal{N}}^{-1} c_{\mathcal{N},t}$ and conditional variance $d_t = C(t,t) - c_{\mathcal{N},t}^\top b_t$. Total cost is $O(N m^3)$ instead of $O(N^3)$ — taking < 9 seconds for all 208,417 training observations!

### Q10. Why did you have to align residuals by physical `arc_id` inside the NNGP whitening operator?
**Answer:** At step $t-1$, candidate slot 0 might be `arc_id = 4521`, while at step $t$ (due to GPS noise), `arc_id = 4521` might move to candidate slot 2! If you subtracted raw candidate vectors $r_t - b_{t,1} r_{t-1}$ slot-by-slot, you'd subtract residuals of *different* roads. Our `align_k[t, r, k]` aligns every neighbor residual to the exact matching physical `arc_id`, ensuring mathematically valid spatial decorrelation.

### Q11. What happens in NN-GLS when a trip has a 10-minute GPS dropout?
**Answer:** In `build_nngp_whitening_for_trip()`, whenever $\Delta t > 5\text{ s}$, we reset `block_start = t`. Thus $b_t = 0$, `align_k = -1`, and $d_t = \sigma^2 + \tau^2$ — completely preventing spatial covariance or Kriging smoothing from leaking across GPS dropouts.

### Q12. What is the role of the nugget $\tau^2$ vs spatial variance $\sigma^2$ in your kernel?
**Answer:** $\sigma^2 = 0.85$ captures spatially/temporally correlated multipath drift along a trajectory, while the nugget $\tau^2 = 0.15$ captures irreducible independent white GPS receiver jitter (and guarantees $C_{\mathcal{N},\mathcal{N}}$ is strictly positive definite and numerically invertible!).

### Q13. How does `NNGPKrigingGraphLayer` relate to Zhan & Datta's GNN equivalence theorem?
**Answer:** Zhan & Datta proved that applying $(I - B)$ to spatial vectors is literally a graph convolution whose edge weights are the Kriging weights $b_{t,r}$. Our `NNGPKrigingGraphLayer` executes this Kriging-weighted message passing across lags $r=1 \dots 5$ on matched physical `arc_id`s, fused via a learned residual gate with forward/backward directed road-network message passing (`step_conn`) and intra-step cross-candidate self-attention.

---

## Part D: Model 4 — RLCD Theory & Implementation

### Q14. What does RLCD stand for, and how is it different from RLHF or RLVR?
**Answer:** **RLCD** stands for **Reinforcement Learning for Calibrated Decisions**. Unlike RLHF (human text preferences) or RLVR (binary +1/0 correctness rewards that trigger overconfident logits), RLCD optimizes a stochastic decision policy using **strictly proper scoring rule rewards** (Brier score / log score) plus task-specific utility (road-network continuity) so the model outputs **statistically calibrated probabilities** alongside high-accuracy decisions.

### Q15. Prove why the Brier score prevents the model from bluffing with 99% confidence.
**Answer:** See Theory Guide 02: for true probability $q$ and reported probability $p$, expected negative Brier reward $J(p) = 1 - q(p-1)^2 - (1-q)p^2$ has derivative $J'(p) = 2(q-p) = 0 \iff p^* = q$, and $J''(p) = -2 < 0$ (strictly concave). If the model knows $q = 0.70$ and bluffs $p = 0.99$, expected loss worsens from $q(1-q) = 0.21$ to $0.70(0.01)^2 + 0.30(0.99)^2 = 0.2941$ — a huge penalty!

### Q16. Why did you use GRPO (Group-Relative Policy Optimization) with $G = 8$ rollouts instead of PPO with a critic?
**Answer:** In PPO, training a separate neural value/critic network introduces high bias/variance and extra hyperparameters. Because sampling $G = 8$ parallel candidate trajectories from a $K = 10$ categorical policy on GPU takes microseconds, GRPO computes exact group-relative advantages $A_t^{(g)} = (R_t^{(g)} - \mu_t)/(\sigma_t + \varepsilon)$ directly across the 8 rollouts — zero critic network, lower variance, faster convergence!

### Q17. Explain the role of the KL Divergence Anchor $\beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \| \pi_{\text{NN-GLS}})$.
**Answer:** Stage 1 (NN-GLS) carefully learned spatially-debiased representations using the NNGP precision matrix. Without a KL anchor during Stage 2 RL rollouts, policy gradients could drift away from those spatial representations or overfit the training windows. The KL penalty ($\beta_{\text{KL}} = 0.12$) keeps the RLCD-calibrated policy anchored close to the NN-GLS spatial backbone while refining calibration and road-graph continuity.

### Q18. What is Expected Calibration Error (ECE), and how does your Soft-ECE penalty optimize it?
**Answer:** Standard ECE partitions predictions into 15 hard confidence bins and computes $\sum_b \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$, which is non-differentiable because of hard bin boundaries. Our `compute_soft_ece_loss()` replaces hard bin membership with smooth Gaussian-kernel soft bin assignments over 10 centers, allowing direct gradient descent on calibration error alongside the Brier score and learnable temperature $\tau$.

### Q19. Why does Model 4 NOT need an HMM/Viterbi decoder to achieve high Route Continuity?
**Answer:** Traditional classifiers score every GPS point $t$ independently, so they require an external HMM/Viterbi post-processor to enforce graph continuity. In Model 4, **both the architecture and the RL objective encode road-network structure directly**:
1. Inside the network, 2 layers of bidirectional `NNGPKrigingGraphLayer` pass messages along `step_conn[t, i, j]` and exact NNGP Kriging neighbors across 64-step sliding windows.
2. During Stage 2 RLCD training, the GRPO reward explicitly gives `+1.0` for directed connected transitions and `-0.5` for bidirectional U-turn flip-flops.
Thus sequential continuity and calibrated probabilities emerge end-to-end in a single fast GPU pass!

---

## Part E: Evaluation, Leakage Prevention & System Architecture

### Q20. How did you prevent data leakage during cross-validation and scaling?
**Answer:**
1. **Group-Aware CV (`GroupKFold(n_splits=5)` grouped by `trip_id`):** Observations from the same trip NEVER appear in both train and validation folds (which would artificially inflate accuracy due to spatiotemporal autocorrelation!).
2. **Fold-Internal Scaling:** `StandardScaler` is fit strictly on the 4 training folds inside every CV split, and the final scaler is fit strictly on the 80 train trips.
3. **Strict Partition Lock:** `student_test` (10 trips) is never used for training, early stopping, or scaling. `evaluator_test` is untouched until final live evaluation on Laptop 1.

### Q21. Why do you report BOTH Observation-level Top-1 Accuracy AND Candidate-level Macro F1 / Log-Loss / Route Continuity?
**Answer:** Candidate classification has a 9:1 class imbalance (1 matching arc vs. 9 non-matching arcs per observation), so candidate-level accuracy alone would be misleadingly high (~90% even for an all-zero classifier!). Therefore:
- **Observation-level Top-1 Accuracy** measures exact end-to-end GPS snapping correctness (counting candidate misses strictly as errors).
- **Macro F1-score** equally weights positive match vs. non-match classes.
- **Route Continuity & Sequence Arc F1** verify topological road-network realism.
- **Log-Loss, Brier Score & ECE** verify genuine probabilistic calibration (the hallmark of RLCD!).

### Q22. How do you ensure ground-truth labels never leak to Laptop 2 (UI) or the prediction model during live evaluation?
**Answer:** In `08_api/inference_engine.py`, Phase 1 calls `load_trip(eval_dir, trip_id, load_route=False)`, so `.route` is never even read from disk during inference. Phase 2 independently loads `.route` strictly inside Laptop 1 to compute scalar evaluation metrics (`obs_top1_accuracy`, `macro_f1`, `route_continuity`, `ece`, etc.) and returns ONLY predicted route polylines + scalar metrics — our automated E2E test (`10_deployment/test_api_e2e.py`) asserts zero forbidden ground-truth keys in every JSON response!

### Q23. How do you switch from `student_test` to the evaluator's 10 reserved trips on evaluation day?
**Answer:** Zero code changes! On Laptop 1, simply launch:
```bash
PATHSNAP_EVAL_DIR=/path/to/evaluator_test ./10_deployment/start_api_laptop1.sh
```
Laptop 1 immediately loads the 10 reserved trips, prints live inference logs per request to the CMD/terminal for the evaluator, and Laptop 2's UI fetches the trip list dynamically via `GET /trips` and plots routes on **OpenStreetMap's free public tile tier** (zero API keys required!).

### Q24. What are the main limitations of Model 4 (`NN-GLS + RLCD`) and how could it be improved further?
**Answer:**
1. **Candidate Miss Ceiling:** Like any candidate-based neural matcher, if an extreme GPS jump (>180 m, e.g. `00000002` at 569 m) pushes the true arc outside Top-$K=10$, argmax cannot recover it without increasing $K$ or falling back to shortest-path interpolation.
2. **Soft vs Hard Topological Constraint:** Unlike Viterbi decoding (which guarantees a globally optimal path over an entire 10-km trip), Model 4 enforces continuity via finite-window (64-step) graph message passing and RL continuity rewards, so rare 1-step topological glitches can still occur at complex multi-level overpasses. Combining Model 4's RLCD-calibrated probabilities with Teammate 1's Viterbi decoder is a natural ensemble extension!
3. **Training Compute:** Two-stage GPU training (NNGP whitening + GRPO rollouts) requires a GPU for fast iteration (~minutes on RTX 4060), though inference runs fast on either GPU or CPU.
