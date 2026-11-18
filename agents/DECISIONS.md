# PathSnap Engineering & Scientific Decision Log — Model 4 (NN-GLS + RLCD)

## D01 — Coordinate Reference System (Local WGS-84 Azimuthal Equidistant Projection)
- **Observation:** The 80 training trips span globally (`lon ∈ [-100.42°, +137.24°]`, `lat ∈ [-34.64°, +61.27°]`). Euclidean degree distance is invalid and no single UTM zone spans Earth.
- **Decision:** Per-trip local WGS-84 Azimuthal Equidistant (AEQD) East-North-Up metric plane centered at each road network centroid (`+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m`).

## D02 — Empirical Discovery: 92.2% Bidirectional Directed Arcs
- **Observation:** 92.16% (train) and 92.69% (student_test) of directed road segments have an opposite-direction twin `(v->u)` with identical perpendicular distance, capping naive nearest-arc Top-1 accuracy at 50.14% (train) / 54.71% (student_test).
- **Decision:** Engineered multi-scale gap-aware heading features (`cos_heading_inst`, `cos_heading_smooth`, `cos_heading_wide`, `has_reverse_pair`, `heading_rank_near`) + intra-step cross-candidate self-attention inside `NNGLSSpatioTemporalNet`. Gradient saliency confirmed `cos_heading_wide` as the #1 most important feature (`6.94%` relative saliency).

## D03 — Empirical Discovery: High-Noise Correlated Drift Trips (`00000054`, `00000033`)
- **Observation:** `train/00000054` (19,313 obs, 138.5 m median error, 51.4% Top-10 miss) and `train/00000033` (5,844 obs, 122.4 m error, 65.8% Top-10 miss) demonstrate strong spatial-temporal multipath autocorrelation and could dominate i.i.d. gradients.
- **Decision:**
  1. Model spatial autocorrelation directly via **NN-GLS** (Zhan & Datta, JASA 2025): fixed-lag Vecchia / NNGP precision whitening $L^{-1} = D^{-1/2}(I - B)$ aligned by physical `arc_id` ($m = 5$, $\rho_s = 45\text{ m}$, $\rho_t = 6\text{ s}$, $\sigma^2 = 0.85$, $\tau^2 = 0.15$).
  2. Cap training windows at `max_windows_per_trip = 140` (strictly unsupervised, based only on trip length) so one massive anomalous trip cannot overwhelm 77 clean trips.
  3. Strictly gate all temporal smoothing, NNGP covariance, and road-graph message passing across GPS time gaps ($\Delta t > 5\text{ s}$).

## D04 — Subdivided-Segment Spatial Index & Monotonic DP Labeling
- **Observation:** Highway arcs reach 9,297 m in length (causing midpoint KD-trees to miss long segments) and 9/80 train + 1/10 student_test trips revisit arcs (loops).
- **Decision:** Subdivide arcs >40 m into proxy points for `cKDTree` Top-10 search (**0.17% Candidate Miss Rate** on `student_test`). Align `.track` to `.route` via Monotonic DP exclusively during training label generation and strictly server-side evaluation.

## D05 — Pure Model 4 Scope (Zero HMM / Zero Bayesian HPO)
- **Team Split:** Teammate 1 owns Model 1 (HMM + Viterbi + Bayesian HPO); Teammate 2 owns Model 2 (RLCD); Teammate 3 owns Model 3 (NN-GLS); **Branch `M4`** delivers **Model 4 (Pure NN-GLS + RLCD)** end-to-end.
- **Key Architectural Refinement (Integrated NNGP Kriging):** Applying NNGP residual kriging *outside* the RLCD training graph initially sharpened post-RLCD logits and inflated ECE (`0.1226 -> 0.1501` on Fold 0). Moving differentiable bounded NNGP residual kriging **inside** `NNGLSSpatioTemporalNet.forward()` *before* the learnable calibration temperature $\tau$ eliminated train/eval calibration mismatch and drove held-out `student_test` ECE down to **`0.0351`**, Route Continuity to **`99.24%`** (`96.61%` exact 1-hop), Sequence Arc F1 to **`0.9215`**, and Top-1 Accuracy to **`91.50%`** at **`39.33 ms/trip`** without any HMM!

## D06 — OpenStreetMap Free Tier & Strict Server-Side Ground-Truth Isolation
- **Maps:** 100% free public OpenStreetMap + OpenTopoMap tile layers in both offline Folium HTML maps (`07_visualisations/maps/`) and Laptop 2's Leaflet.js Web UI (`09_ui/`), zero API keys.
- **Security Isolation:** Laptop 1 FastAPI (`08_api/inference_engine.py`) executes prediction with `load_route=False`. Ground-truth `.route` is loaded only in an isolated server-side metric evaluator and is never exposed in JSON payloads (`ground_truth_exposed: false`, verified by `10_deployment/test_api_e2e.py`).
