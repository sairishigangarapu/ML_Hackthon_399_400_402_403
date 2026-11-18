# Theory Deep Dive 01: NN-GLS (Neural Networks with Generalized Least Squares)

> **Primary Reference:** Wentao Zhan & Abhirup Datta (2024/2025), *"Neural Networks for Geospatial Data"*, Journal of the American Statistical Association (JASA) / `geospaNN`.

---

## 1. Why Standard i.i.d. Neural Networks Fail on GPS Trajectory Data

Standard deep learning classifiers minimize an unweighted Empirical Risk (Ordinary Least Squares or standard Cross-Entropy):
$$\mathcal{L}_{\text{i.i.d.}}(\theta) = \frac{1}{N} \sum_{t=1}^{N} \ell\big(y_t, f_\theta(x_t)\big)$$
This objective implicitly assumes that observation errors $\varepsilon_t = y_t - \mathbb{E}[y_t \mid x_t]$ are **independent and identically distributed (i.i.d.)**.

In PathSnap GPS map-matching, our Exploratory Data Analysis (EDA) proved that this assumption is severely violated:
- **Spatial-Temporal Multipath Autocorrelation:** When a vehicle enters an urban canyon, tunnel, or parking structure, GPS errors do not jump independently every second — they **drift systematically** in the same direction for 10 to 300+ consecutive seconds.
- **Empirical Evidence from Our Dataset:**
  - `train/00000054` (19,313 GPS points — nearly 10% of the entire training set!) exhibits a **138.5 m median GPS-to-route error** and 169.2 km of cumulative jitter along a 31.4 km route.
  - `train/00000033` (5,844 GPS points) exhibits a **122.4 m median GPS-to-route error**.
- Under an i.i.d. loss, 100 tightly correlated GPS points from a single drifted burst receive **100× the gradient weight** of a single clean observation, biasing the neural mean function $f_\theta$ and causing poor generalization to unseen cities/trips.

---

## 2. The Geostatistical Gaussian Process Model & GLS Objective

Following classical model-based geostatistics, Zhan & Datta embed a flexible neural mean function $f_\theta(x_t)$ inside a spatial Gaussian Process (GP):
$$y(s_t) = f_\theta\big(X(s_t)\big) + w(s_t) + \varepsilon(s_t), \qquad w(\cdot) \sim \mathcal{GP}\big(0, C_\phi(\cdot, \cdot)\big), \quad \varepsilon(s_t) \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \tau^2)$$
where the total spatiotemporal covariance matrix is $\Sigma = C_\phi + \tau^2 I$.

In our PathSnap implementation (`03_model_training/nngls_core.py`), we use the **Exponential (Matérn-$\tfrac{1}{2}$) Spatiotemporal Covariance Kernel**:
$$C_\phi\big((s_t, t),\, (s_{t'}, t')\big) = \sigma^2 \exp\!\left(-\frac{\|s_t - s_{t'}\|_2}{\rho_s} - \frac{|t - t'|}{\rho_t}\right) + \tau^2 \delta_{t, t'}$$
with parameters:
- $\rho_s = 45\text{ m}$: spatial correlation lengthscale (typical urban block / canyon scale)
- $\rho_t = 6\text{ s}$: temporal correlation lengthscale (typical multipath dwell window)
- $\sigma^2 = 0.85$: correlated spatial process variance
- $\tau^2 = 0.15$: white-noise measurement nugget
- **GPS-Gap Cutoff ($\Delta t > 5\text{ s}$):** Across GPS signal dropouts (up to 153,064 s in `train`), covariance is set strictly to $0$ so disconnected segments never corrupt one another.

The **Generalized Least Squares (GLS)** objective decorrelates residuals $r = y - p_\theta(X)$ using the inverse covariance (precision) matrix $\Sigma^{-1}$:
$$\mathcal{L}_{\text{GLS}}(\theta) = \big(y - p_\theta(X)\big)^\top \Sigma^{-1} \big(y - p_\theta(X)\big) = \big\|L^{-1}\big(y - p_\theta(X)\big)\big\|_2^2$$
where $\Sigma = L L^\top$ is the Cholesky factorization.

---

## 3. Scaling from $O(N^3)$ to $O(N m^3)$ via NNGP / Vecchia Approximation

Exact Cholesky inversion of $\Sigma \in \mathbb{R}^{N \times N}$ requires $O(N^3)$ compute and $O(N^2)$ memory — impossible for $N = 208{,}417$ GPS observations.

Zhan & Datta replace $\Sigma^{-1}$ with the **Nearest-Neighbor Gaussian Process (NNGP / Vecchia)** sparse precision matrix (Datta et al., 2016):
$$\tilde{\Sigma}^{-1} = (I - B)^\top D^{-1} (I - B) \implies L^{-1} = D^{-1/2}(I - B)$$
For each observation $t$ along a trajectory, let $\mathcal{N}(t) = \{t - 1, \dots, t - m\}$ ($m = 5$) be its $m$ nearest preceding spatiotemporal neighbors inside the same contiguous non-gap block. The exact $m \times m$ Kriging system is solved once per point:
$$b_t = C_{\mathcal{N}(t), \mathcal{N}(t)}^{-1}\, c_{\mathcal{N}(t), t} \in \mathbb{R}^m, \qquad d_t = C(t, t) - c_{\mathcal{N}(t), t}^\top b_t > 0$$

### Physical `arc_id` Alignment (Our Map-Matching Innovation)
At consecutive GPS steps $t-1$ and $t$, the Top-$K$ candidate set $\mathcal{C}_t$ is re-sorted by distance, so the same physical road segment might be slot $k=0$ at step $t-1$ and slot $k=2$ at step $t$.
- Our `build_nngp_whitening_for_trip()` builds a **fixed-lag alignment tensor** `align_k[t, r, k]` mapping candidate $k$ at step $t$ to the exact slot $k'$ at step $t - (r+1)$ sharing the **identical physical `arc_id`**.
- The exact Vecchia pre-whitened residual vector for candidate $k$ at step $t$ is therefore:
$$\tilde{r}_{t, k} = \frac{1}{\sqrt{d_t}} \left( r_{t, k} - \sum_{r=1}^{m} b_{t, r}\, r_{t - r,\; \text{align}(t, r, k)} \right)$$
- **Exact NN-GLS Training Objective:**
$$\mathcal{L}_{\text{NN-GLS}}(\theta) = \frac{1}{N_{\text{hit}}} \sum_{t} \|\tilde{r}_t\|_2^2 + \lambda_{\text{CE}} \cdot \frac{1}{N_{\text{hit}}} \sum_{t} \tilde{w}_t \cdot \text{CE}\big(\text{logits}_t, y_t\big), \qquad \tilde{w}_t \propto d_t^{-1}$$
Notice how $d_t^{-1}$ (conditional precision) automatically downweights tightly redundant points and upweights spatially informative transitions!

---

## 4. Zhan & Datta's GNN Equivalence Theorem & Inference Kriging

A landmark result of Zhan & Datta (2025) is that multiplying by $(I - B)$ is **mathematically equivalent to a Graph Neural Network (GNN) layer** whose edge weights are the spatial Kriging weights $b_{t,r}$:
$$\big[(I - B) h\big]_t = h_t - \sum_{j \in \mathcal{N}(t)} b_{t, j} h_j$$

We instantiate this theorem directly inside `NNGLSSpatioTemporalNet` (`NNGPKrigingGraphLayer`):
1. **Cross-Candidate Self-Attention:** Compares all $K = 10$ competing candidates at GPS point $t$ (instantly resolving 92.2% bidirectional street pairs `(u->v)` vs `(v->u)`).
2. **NNGP Kriging-Weighted Spatial Aggregation:** Aggregates hidden embeddings across lags $r = 1 \dots m$ along matched physical `arc_id`s weighted by exact Vecchia Kriging weights $b_{t,r}$.
3. **Bidirectional Directed Road-Graph Message Passing:** Aggregates predecessor (`t-1 -> t`) and successor (`t+1 -> t`) embeddings along positive directed road connectivity `step_conn[t, i, j]` (zeroed strictly across GPS gaps!).
4. **Inference-Time NNGP Residual Kriging:** After forward-pass sliding-window logit estimation, zero-centered spatial preferences are kriged across each observation's $m$ neighbors using $b_{t,r}$ before calibrated softmax normalization.

---

## 5. Key Takeaways for Viva
- **Complexity:** $O(N m^3)$ preprocessing ($m=5 \implies m^3 = 125$ flops/point; runs across all 208,417 points in **8.7 seconds**!).
- **Why NN-GLS beats standard i.i.d. MLPs:** Explicitly removes spatial-temporal error autocorrelation from gradients (preventing high-noise trips like `00000054` from dominating updates) and performs Kriging-weighted graph aggregation along physical road segments.
