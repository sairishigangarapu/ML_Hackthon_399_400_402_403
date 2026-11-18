# Theory Deep Dive 02: RLCD (Reinforcement Learning for Calibrated Decisions)

---

## 1. The Motivation: Why Standard RL (RLVR) Breaks Calibration

In standard Reinforcement Learning with Verifiable Rewards (RLVR) — or naive policy-gradient map-matching — a policy $\pi_\theta(a \mid s_t)$ receives a binary **0/1 accuracy reward**:
$$R_{\text{naive}}(a_t, y_t) = \mathbb{I}[a_t = y_t]$$

**The Overconfidence Trap:**
- Whenever a policy has even a slight edge (say, true probability $q = 0.60$ of arc A vs $0.40$ of arc B), maximizing $\mathbb{E}_{a \sim \pi}[R_{\text{naive}}] = 0.60\,\pi(A) + 0.40\,\pi(B)$ is uniquely maximized by setting $\pi(A) = 1.00$ and $\pi(B) = 0.00$!
- Consequently, standard policy gradient pushes logits toward $\pm\infty$, collapsing entropy and creating a **miscalibrated, overconfident model** that reports 99.5% confidence even on ambiguous intersections (causing catastrophic Log-Loss and high Expected Calibration Error, ECE).

**RLCD (Reinforcement Learning for Calibrated Decisions)** solves this by designing the RL reward around **Strictly Proper Scoring Rules** combined with **Sequential Decision Utility** (here, directed road-network continuity).

---

## 2. Mathematical Proof: Why Brier Score is a Strictly Proper Scoring Rule

A scoring rule $S(p, y)$ grades a predicted probability distribution $p \in \Delta^{K-1}$ against a realized outcome $y \in \{1, \dots, K\}$. It is **strictly proper** if and only if the expected score under true distribution $q$ is **uniquely maximized when the model reports $p = q$**:
$$\arg\max_{p \in \Delta^{K-1}} \mathbb{E}_{Y \sim q}\big[S(p, Y)\big] = q$$

### 3-Line Whiteboard Proof (Binary Case — Memorize for Viva!)
Let the true event probability be $q = P(Y = 1)$, and suppose the model reports probability $p \in [0, 1]$. Under the **Negative Brier Reward** $R_{\text{Brier}}(p, Y) = 1 - (p - Y)^2$:
1. **Expected Reward:**
   $$J(p) = \mathbb{E}_{Y \sim \text{Bern}(q)}\big[1 - (p - Y)^2\big] = 1 - q(p - 1)^2 - (1 - q)p^2$$
2. **First-Order Optimality Condition:**
   $$\frac{dJ}{dp} = -2q(p - 1) - 2(1 - q)p = 2(q - p) = 0 \iff \mathbf{p^* = q}$$
3. **Second-Order Verification (Strict Concavity):**
   $$\frac{d^2 J}{dp^2} = -2 < 0 \quad \forall p \in [0, 1]$$

**Conclusion:** Any attempt to "bluff" (reporting $p = 0.99$ when true certainty is $q = 0.70$) **strictly lowers expected reward** because the quadratic penalty $(0.99 - 0)^2 = 0.9801$ on the 30% error cases outweighs the tiny gain on the 70% correct cases! The same exact proof generalizes to the **Multi-Class Brier Score** over all $K = 10$ candidate road arcs:
$$R_{\text{Brier}}(p_t, y_t) = 1 - \sum_{k=1}^{K} \big(p_{t, k} - \mathbb{I}[k = y_t]\big)^2$$

---

## 3. Our RLCD Formulation for PathSnap (`03_model_training/rlcd_core.py`)

In Stage 2 of Model 4, we initialize policy $\pi_\theta(\cdot \mid s_t)$ from Stage 1's spatially-debiased **NN-GLS** weights, keep a frozen anchor copy $\pi_{\text{NN-GLS}}$, and perform **Group-Relative Policy Optimization (GRPO)** over contiguous trajectory windows of length $W = 48$:

### Step A: Parallel Group Rollouts ($G = 8$, Zero Critic Variance)
For every window step $t$, we sample $G = 8$ parallel candidate-arc trajectories $a_t^{(g)} \sim \pi_\theta(\cdot \mid s_t)$. Using $G$ parallel rollouts eliminates the need for a noisy learned value/critic network!

### Step B: Calibrated Composite Decision Reward
For each rollout $g \in \{1, \dots, G\}$ at step $t$, the step reward is:
$$R_t^{(g)} = w_{\text{Brier}} \, r_{\text{cal}}\big(a_t^{(g)}, p_t, y_t\big) + w_{\text{hit}} \, \mathbb{I}[a_t^{(g)} = y_t] + w_{\text{cont}} \, C_{\text{graph}}\big(a_{t-1}^{(g)}, a_t^{(g)}\big)$$
where:
1. **Confidence-Aware Asymmetric Proper Reward ($r_{\text{cal}}$):**
   $$r_{\text{cal}} = \begin{cases} 1 - \big(1 - \pi_\theta(a_t^{(g)} \mid s_t)\big)^2 & \text{if } a_t^{(g)} = y_t \quad (\text{rewards justified confidence}) \\ -\pi_\theta(a_t^{(g)} \mid s_t)^2 - 0.35 & \text{if } a_t^{(g)} \neq y_t \quad (\text{heavily penalizes overconfident bluffs!}) \end{cases}$$
2. **Directed Road-Network Continuity Reward ($C_{\text{graph}}$):**
   Precomputed strictly from `.arcs` (`step_conn[t, i, j]`):
   - **`+1.0`**: Same directed arc OR direct 1-hop directed transition (`prev.to_node == curr.from_node`)
   - **`+0.5`**: Reachable within 2 directed hops along the street graph
   - **`-0.5`**: Immediate reverse-direction flip `(u->v)` to `(v->u)` on a 2-way road (directly suppresses bidirectional flicker!)
   - **`0.0`**: Disconnected transition or across a GPS time gap ($\Delta t > 5\text{ s}$)

### Step C: Group-Relative Advantage & Total RLCD Objective
$$A_t^{(g)} = \frac{R_t^{(g)} - \mu_t}{\sigma_t + 10^{-4}}, \qquad \mu_t = \frac{1}{G}\sum_{g=1}^{G} R_t^{(g)}, \quad \sigma_t = \text{std}_g\big(R_t^{(g)}\big)$$

The full Stage 2 RLCD loss combines four synergistic terms:
$$\mathcal{L}_{\text{RLCD}}(\theta) = \underbrace{-\lambda_{\text{PG}} \, \mathbb{E}_g\big[A_t^{(g)} \log \pi_\theta(a_t^{(g)} \mid s_t)\big]}_{\text{GRPO Sequential Continuity & Bluff-Aware PG}} + \underbrace{\lambda_{\text{Brier}} \, \mathcal{L}_{\text{Brier}}(\pi_\theta, y) + \lambda_{\text{NLL}} \, \mathcal{L}_{\text{LogScore}}(\pi_\theta, y)}_{\text{Strictly Proper Scoring Rules}} + \underbrace{\lambda_{\text{ECE}} \, \mathcal{L}_{\text{SoftECE}}(\pi_\theta, y)}_{\text{Differentiable Bin Calibration}} + \underbrace{\beta_{\text{KL}} \, D_{\text{KL}}\big(\pi_\theta \,\|\, \pi_{\text{NN-GLS}}\big)}_{\text{Spatial Representation Anchor}}$$
alongside a learnable calibration temperature $\tau = \exp(\text{log\_temp})$ applied directly to the output logits.

---

## 4. Measuring Calibration Quality (ECE, Brier, Log-Loss)

- **Expected Calibration Error (ECE, 15 bins):**
  $$\text{ECE} = \sum_{b=1}^{15} \frac{|B_b|}{N} \left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$
  An ECE of `0.01` means that whenever Model 4 reports 85% confidence, it is empirically right ~84%–86% of the time!
- **Why this matters for PathSnap:**
  1. In real-world autonomous driving / fleet routing, knowing *when* a GPS snap is ambiguous (e.g., dense urban overpass or parallel service road) is as important as Top-1 accuracy.
  2. In our Laptop 2 Evaluator UI, GPS points are color-coded directly by Model 4's RLCD-calibrated confidence (`green >= 85%`, `amber 55-85%`, `red < 55%`), giving human operators an immediate, statistically trustworthy uncertainty overlay!
