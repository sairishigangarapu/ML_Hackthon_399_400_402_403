# PathSnap Engineering & Scientific Decision Log

## D01 — Coordinate Reference System (Local Tangent Plane / Azimuthal Equidistant Projection per Trip)
- **Observation:** The 80 training trips span globally (`lon ∈ [-100.42°, +137.24°]`, `lat ∈ [-34.64°, +61.27°]`). A single UTM zone cannot cover worldwide trips without massive distortion.
- **Decision:** Use a trip-centered local East-North-Up (ENU) metric projection using WGS-84 ellipsoid radii (via `pyproj` Azimuthal Equidistant `+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m`, with fast analytical equirectangular fallback verified to <0.05% error within 30 km).
- **Why:** Every trip spans <50 km; local AEQD preserves true geodesic distances and bearings in meters everywhere on Earth without UTM zone boundary discontinuities.

## D02 — Key Empirical Discovery: 92.2% Bidirectional Directed Arcs
- **Observation:** In both `train` (92.2%) and `student_test` (92.7%), two-way roads are encoded as pairs of opposite directed arcs `(u->v)` and `(v->u)` sharing identical geometry.
- **Impact:** Distance-only matching suffers an ~50% direction ambiguity ceiling. Therefore, **heading alignment** ($\cos\Delta\theta$, $\sin\Delta\theta$) using multi-scale smoothed vehicle trajectory vectors (to survive stationary stops and jitter) PLUS **directed graph connectivity** in HMM/Viterbi are primary design pillars.

## D03 — Key Empirical Discovery: High-Noise/Correlated GPS Trips (`00000054`, `00000033`, `00000002`)
- **Observation:** Trip `00000054` (19,313 pts, Amsterdam) exhibits ~84–140m median distance to the road network and 169 km of cumulative jitter on a 31 km route; Trip `00000033` (5,844 pts) has 122m median distance; Trip `00000002` has max 569.5m offset; `train` contains GPS time gaps up to 153,064s (~42.5 hours).
- **Impact:**
  1. Validates the physical motivation for **NN-GLS** (accounting for spatially correlated GPS residuals along trajectories via NNGP Vecchia covariance whitening rather than naive i.i.d. assumptions).
  2. Motivates **GPS gap detection** ($\Delta t > \tau_{\text{gap}}$) in both trajectory smoothing (never smooth across time gaps!) and HMM/Viterbi transition dynamics (adaptive transition dispersion / reset on huge gaps).
  3. Motivates strict handling of zero-length arcs (100 in train, 6 in student_test) to prevent NaN bearings.

## D04 — Monotonic Sequence Alignment for Ground-Truth Observation Labeling
- **Observation:** `.route` provides the ordered sequence of traversed `arc_id`s (median ~280 arcs), while `.track` provides ~1,500 1-Hz GPS observations. 9/80 train trips and 1/10 student_test trips revisit arcs (loops/roundtrips).
- **Decision:** Align each GPS observation $t=0,\dots,T-1$ to its reference route index $j_t \in \{0,\dots,M-1\}$ via **Monotonic Dynamic Programming (constrained Viterbi alignment along `.route`)** minimizing perpendicular GPS-to-arc distance + small heading mismatch + monotonicity constraint ($j_t \ge j_{t-1}$ with small backward tolerance only if loop-free distance requires, or strict $j_t \ge j_{t-1}$ along the ordered route).
- **Why:** Prevents self-intersecting or revisited routes from snapping an early GPS point to the return-trip arc!

## D05 — Eight-Model Experimental Suite (Exceeding PDF 5-Model Minimum + All 4 Requested Paradigms)
- Classical/Neural Baselines: (1) Logistic Regression, (2) Random Forest, (3) LightGBM Untuned, (4) Standard MLP (BCE).
- Four Core Requested Models:
  - **Model 1:** Bayesian-Optimised LightGBM + HMM/Viterbi (Optuna TPE on 5-fold GroupKFold CV + Bayesian-tuned HMM parameters).
  - **Model 2:** **RLCD** (Reinforcement Learning for Calibrated Decisions) — stochastic listwise policy trained via REINFORCE/GRPO with strictly proper **Brier score reward** + **topological route continuity reward** + **expected calibration error (ECE) regularization**.
  - **Model 3:** **NN-GLS** (Neural Networks with Generalized Least Squares, Zhan & Datta, JASA 2025) — neural mean network trained with sparse **NNGP/Vecchia Cholesky whitening loss** over local spatiotemporal trajectory neighborhoods + **NNGP residual kriging** at inference.
  - **Model 4:** **NN-GLS + RLCD** — two-stage hybrid initializing from spatially-debiased NN-GLS representations and post-training via RLCD calibrated policy gradient (with KL anchor to NN-GLS), evaluated both standalone and with HMM/Viterbi!

## D06 — Maps & Two-Laptop Architecture
- **Maps:** OpenStreetMap free public tile tier (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`) via **Folium** (offline maps & report) and **Leaflet.js** (live Laptop 2 Web UI), zero paid API keys.
- **Security Isolation:** FastAPI (Laptop 1) executes full prediction pipeline using ONLY `.track`, `.nodes`, `.arcs`. Ground-truth `.route` files remain strictly server-side for computing evaluation metrics and are NEVER returned to the UI or passed to the predictor.
