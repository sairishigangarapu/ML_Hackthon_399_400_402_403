"""NN-GLS Core Module (Zhan & Datta, JASA 2025 — Neural Networks for Geospatial Data).

Implements the complete NN-GLS framework tailored for GPS trajectory map-matching:
1. Exact sparse Vecchia / Nearest-Neighbor Gaussian Process (NNGP) Cholesky pre-whitening
   operator L^{-1} = D^{-1/2}(I - B) over spatiotemporal trajectory neighborhoods with
   fixed-lag physical arc_id alignment.
2. NNGLSSpatioTemporalNet: Neural mean function incorporating Cross-Candidate Self-Attention
   and Zhan & Datta's NNGP Kriging-Weighted Graph Convolution + Bidirectional Road-Graph
   Message Passing layers (100% pure neural spatial-temporal reasoning, zero HMM).
3. Exact Vecchia / NNGP Generalized Least Squares (GLS) decorrelated training objective and
   inference-time NNGP Spatial Residual Kriging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NNGPConfig:
    m_neighbors: int = 5          # number of preceding spatiotemporal neighbors in Vecchia ordering
    rho_spatial_m: float = 45.0   # spatial correlation lengthscale (meters)
    rho_temporal_s: float = 6.0   # temporal correlation lengthscale (seconds)
    sigma2: float = 0.85          # spatial GP variance
    tau2: float = 0.15            # nugget (white measurement noise variance)
    gap_threshold_s: float = 5.0  # break spatial covariance across GPS dropouts > gap_threshold_s
    krige_strength: float = 0.25  # inference-time NNGP spatial residual kriging weight


def build_nngp_whitening_for_trip(
    coords: np.ndarray,
    cand_arcs: np.ndarray,
    cand_valid: np.ndarray,
    cfg: NNGPConfig,
) -> Dict[str, np.ndarray]:
    """Precompute exact fixed-lag Vecchia / NNGP kriging weights b_t, conditional std^{-1} d_t^{-1/2},
    and physical arc_id alignment indices across spatiotemporal trajectory neighbors.

    Fixed-lag convention: slot r in {0, ..., m-1} ALWAYS corresponds to lag = r + 1 (step t - (r + 1)).
    Across GPS time gaps > gap_threshold_s or before trip start, b_weights = 0 and align_k = -1.
    """
    T, K = cand_arcs.shape
    m = cfg.m_neighbors
    nbr_idx = np.full((T, m), -1, dtype=np.int32)
    b_weights = np.zeros((T, m), dtype=np.float32)
    d_inv_sqrt = np.ones(T, dtype=np.float32)
    align_k = np.full((T, m, K), -1, dtype=np.int16)

    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    ts = coords[:, 2].astype(np.float64)

    sig2 = cfg.sigma2
    tau2 = cfg.tau2
    c0 = sig2 + tau2
    inv_rs = 1.0 / max(cfg.rho_spatial_m, 1e-3)
    inv_rt = 1.0 / max(cfg.rho_temporal_s, 1e-3)

    arc_to_slot: List[Dict[int, int]] = []
    for t in range(T):
        d: Dict[int, int] = {}
        for k in range(K):
            if cand_valid[t, k]:
                d[int(cand_arcs[t, k])] = k
        arc_to_slot.append(d)

    block_start = 0
    for t in range(T):
        if t > 0 and (ts[t] - ts[t - 1]) > cfg.gap_threshold_s:
            block_start = t

        n_avail = min(m, t - block_start)
        if n_avail <= 0:
            d_inv_sqrt[t] = float(1.0 / np.sqrt(c0))
            continue

        # Fixed-lag ordering: slot 0 -> t-1 (lag 1), slot 1 -> t-2 (lag 2), ...
        nbrs_rev = np.arange(t - 1, t - n_avail - 1, -1, dtype=np.int32)
        nbr_idx[t, :n_avail] = nbrs_rev

        nx, ny, nt = x[nbrs_rev], y[nbrs_rev], ts[nbrs_rev]
        ds_nn = np.hypot(nx[:, None] - nx[None, :], ny[:, None] - ny[None, :])
        dt_nn = np.abs(nt[:, None] - nt[None, :])
        C_nn = sig2 * np.exp(-ds_nn * inv_rs - dt_nn * inv_rt)
        C_nn[np.arange(n_avail), np.arange(n_avail)] += tau2

        ds_nt = np.hypot(nx - x[t], ny - y[t])
        dt_nt = np.abs(nt - ts[t])
        c_nt = sig2 * np.exp(-ds_nt * inv_rs - dt_nt * inv_rt)

        b = np.linalg.solve(C_nn, c_nt)
        cond_var = max(float(c0 - np.dot(c_nt, b)), tau2 * 0.5)
        b_weights[t, :n_avail] = b.astype(np.float32)
        d_inv_sqrt[t] = float(1.0 / np.sqrt(cond_var))

        for r in range(n_avail):
            nb = int(nbrs_rev[r])
            nb_map = arc_to_slot[nb]
            for k in range(K):
                if cand_valid[t, k]:
                    aid = int(cand_arcs[t, k])
                    if aid in nb_map:
                        align_k[t, r, k] = nb_map[aid]

    return {
        "nbr_idx": nbr_idx,
        "b_weights": b_weights,
        "d_inv_sqrt": d_inv_sqrt,
        "align_k": align_k,
    }


def build_nngp_whitening_all(
    coords: np.ndarray,
    cand_arcs: np.ndarray,
    cand_valid: np.ndarray,
    trip_starts: np.ndarray,
    trip_ends: np.ndarray,
    cfg: NNGPConfig,
) -> Dict[str, np.ndarray]:
    """Build global-indexed NNGP Vecchia whitening arrays across a concatenated partition."""
    N, K = cand_arcs.shape
    m = cfg.m_neighbors
    nbr_idx = np.full((N, m), -1, dtype=np.int32)
    b_weights = np.zeros((N, m), dtype=np.float32)
    d_inv_sqrt = np.ones(N, dtype=np.float32)
    align_k = np.full((N, m, K), -1, dtype=np.int16)

    for s, e in zip(trip_starts, trip_ends):
        s, e = int(s), int(e)
        res = build_nngp_whitening_for_trip(coords[s:e], cand_arcs[s:e], cand_valid[s:e], cfg)
        loc_nbr = res["nbr_idx"]
        nbr_idx[s:e] = np.where(loc_nbr >= 0, loc_nbr + s, -1)
        b_weights[s:e] = res["b_weights"]
        d_inv_sqrt[s:e] = res["d_inv_sqrt"]
        align_k[s:e] = res["align_k"]

    return {
        "nbr_idx": nbr_idx,
        "b_weights": b_weights,
        "d_inv_sqrt": d_inv_sqrt,
        "align_k": align_k,
    }


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class NNGPKrigingGraphLayer(nn.Module):
    """Zhan & Datta NNGP Kriging-Weighted Graph Convolution + Bidirectional Road-Graph Layer.

    Propagates candidate representations across contiguous trajectory steps using:
      1. Exact NNGP spatial-temporal Kriging weights b_{t,r} along matched physical arc_ids.
      2. Forward directed road-network transitions (step t-1 -> t) weighted by step_conn > 0.
      3. Backward directed road-network transitions (step t+1 -> t) weighted by step_conn > 0.
      4. Intra-step cross-candidate self-attention (resolves bidirectional street pairs & intersections).
    Strictly blocks message flow across GPS gaps (where b_w == 0 and step_conn == 0).
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.cand_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_msg = nn.LayerNorm(dim)
        self.msg_proj = nn.Linear(dim * 3, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.res = ResidualBlock(dim, dropout=dropout)

    def forward(
        self,
        h: torch.Tensor,         # (B, W, K, D)
        valid: torch.Tensor,     # (B, W, K) bool
        b_w: torch.Tensor,       # (B, W, m)
        align_k: torch.Tensor,   # (B, W, m, K) int16
        conn: torch.Tensor,      # (B, W, K, K) float32
    ) -> torch.Tensor:
        B, W, K, D = h.shape
        vf = valid.unsqueeze(-1).float()

        # 1. Intra-step cross-candidate self-attention across K competing arcs
        h_flat = self.norm_attn(h).reshape(B * W, K, D)
        kp = ~valid.reshape(B * W, K)
        all_masked = kp.all(dim=-1)
        if all_masked.any():
            kp = kp.clone()
            kp[all_masked, 0] = False
        attn_out, _ = self.cand_attn(h_flat, h_flat, h_flat, key_padding_mask=kp, need_weights=False)
        h = (h + attn_out.reshape(B, W, K, D)) * vf

        # 2. Spatiotemporal NNGP Kriging + Bidirectional Road-Graph Message Passing
        hn = self.norm_msg(h) * vf
        m = b_w.shape[-1]

        # 2a. Exact NNGP Kriging-weighted spatial aggregation over lags r = 1..m (same physical arc_id)
        krige_msg = torch.zeros_like(hn)
        krige_wsum = torch.zeros((B, W, K, 1), device=h.device, dtype=h.dtype)
        for r in range(m):
            lag = r + 1
            if lag >= W:
                break
            prev_h = hn[:, :-lag, :, :]                               # (B, W-lag, K, D)
            ak = align_k[:, lag:, r, :]                               # (B, W-lag, K)
            has_match = (ak >= 0).unsqueeze(-1).float()
            safe_ak = ak.clamp(min=0).long().unsqueeze(-1).expand(-1, -1, -1, D)
            gathered = torch.gather(prev_h, dim=2, index=safe_ak) * has_match
            bw = b_w[:, lag:, r].clamp(min=0.0).unsqueeze(-1).unsqueeze(-1)  # (B, W-lag, 1, 1)
            krige_msg[:, lag:, :, :] = krige_msg[:, lag:, :, :] + bw * gathered
            krige_wsum[:, lag:, :, :] = krige_wsum[:, lag:, :, :] + bw * has_match
        krige_msg = krige_msg / krige_wsum.clamp(min=1.0)

        # 2b. Forward & Backward directed road-network transition messages (step_conn[t, i, j])
        fwd_msg = torch.zeros_like(hn)
        bwd_msg = torch.zeros_like(hn)
        if W > 1:
            pos_conn = conn[:, 1:, :, :].clamp(min=0.0)  # (B, W-1, K_prev, K_curr)
            # Forward: target j at step t receives from connected predecessors i at step t-1
            fwd_denom = pos_conn.sum(dim=2, keepdim=False).unsqueeze(-1).clamp(min=1.0)  # (B, W-1, K_curr, 1)
            fwd_raw = torch.einsum("btij,btid->btjd", pos_conn, hn[:, :-1, :, :])
            fwd_msg[:, 1:, :, :] = fwd_raw / fwd_denom

            # Backward: source i at step t-1 receives from connected successors j at step t
            bwd_denom = pos_conn.sum(dim=3, keepdim=False).unsqueeze(-1).clamp(min=1.0)  # (B, W-1, K_prev, 1)
            bwd_raw = torch.einsum("btij,btjd->btid", pos_conn, hn[:, 1:, :, :])
            bwd_msg[:, :-1, :, :] = bwd_raw / bwd_denom

        msg = F.silu(self.msg_proj(torch.cat([krige_msg, fwd_msg, bwd_msg], dim=-1)))
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, msg], dim=-1)))
        h = (h + gate * msg) * vf
        h = self.res(h) * vf
        return h


class NNGLSSpatioTemporalNet(nn.Module):
    """Model 4 Backbone: Pure NN-GLS Spatial-Temporal Kriging-Graph Network (Zero HMM).

    Architecture:
      Input (B, W, K, F) -> Candidate Encoder -> 2x NNGPKrigingGraphLayer (Cross-Candidate Attention
      + Zhan & Datta NNGP Kriging Conv + Bidirectional Road-Graph Message Passing)
      -> Contrastive Listwise Calibrated Head -> Logits (B, W, K) with learnable temperature.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim, dropout=dropout),
        )
        self.graph_layers = nn.ModuleList([
            NNGPKrigingGraphLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Learnable bounded NNGP spatial residual kriging strength inside forward pass
        # so RLCD calibrates the exact post-kriging distribution with zero train/eval mismatch
        self.raw_krige_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,         # (B, W, K, F)
        valid: torch.Tensor,     # (B, W, K) bool
        b_w: torch.Tensor,       # (B, W, m)
        align_k: torch.Tensor,   # (B, W, m, K)
        conn: torch.Tensor,      # (B, W, K, K)
    ) -> torch.Tensor:
        vf = valid.unsqueeze(-1).float()
        h = self.input_proj(x) * vf

        for layer in self.graph_layers:
            h = layer(h, valid, b_w, align_k, conn)

        # Contrast each candidate against the masked mean of competing candidates at step t
        mean_ctx = (h * vf).sum(dim=2, keepdim=True) / vf.sum(dim=2, keepdim=True).clamp(min=1.0)
        h_cat = torch.cat([h, h - mean_ctx], dim=-1)

        raw_logits = self.head(h_cat).squeeze(-1)  # (B, W, K)

        # Exact differentiable NNGP Spatial Residual Kriging over matched physical arc_ids
        # Subtract per-step mean so kriging propagates zero-centered spatial preferences
        v_float = valid.float()
        mean_l = (raw_logits * v_float).sum(dim=-1, keepdim=True) / v_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
        centered = (raw_logits - mean_l) * v_float

        B, W, K = raw_logits.shape
        m = b_w.shape[-1]
        krige_acc = torch.zeros_like(raw_logits)
        krige_wsum = torch.zeros_like(raw_logits)
        for r in range(m):
            lag = r + 1
            if lag >= W:
                break
            prev_c = centered[:, :-lag, :]                   # (B, W-lag, K)
            ak = align_k[:, lag:, r, :]                      # (B, W-lag, K)
            has_match = (ak >= 0).float()
            safe_ak = ak.clamp(min=0).long()
            gathered = torch.gather(prev_c, dim=-1, index=safe_ak) * has_match
            bw = b_w[:, lag:, r].clamp(min=0.0).unsqueeze(-1)
            krige_acc[:, lag:, :] = krige_acc[:, lag:, :] + bw * gathered
            krige_wsum[:, lag:, :] = krige_wsum[:, lag:, :] + bw * has_match

        krige_alpha = 0.35 * torch.sigmoid(self.raw_krige_alpha)
        kriged_logits = raw_logits + krige_alpha * (krige_acc / (krige_wsum + 1.0))

        # Calibrated temperature scaling applied AFTER NNGP spatial kriging
        temp = torch.exp(self.log_temp.clamp(-1.2, 1.5))
        logits = kriged_logits / temp
        logits = logits.masked_fill(~valid, -1e4)
        return logits


def compute_nngls_window_loss(
    logits_win: torch.Tensor,
    y_idx_win: torch.Tensor,
    valid_win: torch.Tensor,
    b_weights_win: torch.Tensor,
    d_inv_sqrt_win: torch.Tensor,
    align_k_win: torch.Tensor,
    ce_weight: float = 0.55,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Exact Vecchia / NNGP Generalized Least Squares (GLS) loss on contiguous trajectory windows.

    Decorrelates spatial-temporal residuals r_t = y_t - p_t using the sparse NNGP Cholesky
    factor L^{-1} = D^{-1/2}(I - B):
      r_tilde_t = d_t^{-1/2} * (r_t - sum_{r=1}^m b_{t,r} * r_{t-r -> t})
      L_GLS = E[ ||r_tilde_t||_2^2 ] + ce_weight * E[ w_t * CE(logits_t, y_t) ]
    where w_t proportional to d_t^{-1} downweights redundant correlated GPS bursts.
    """
    B, W, K = logits_win.shape
    m = b_weights_win.shape[-1]

    probs = F.softmax(logits_win, dim=-1) * valid_win.float()
    hit_mask = y_idx_win >= 0
    safe_y = y_idx_win.clamp(min=0).long()
    y_oh = F.one_hot(safe_y, num_classes=K).float() * hit_mask.unsqueeze(-1).float()

    resid = (y_oh - probs) * valid_win.float()
    pred_resid_from_nbrs = torch.zeros_like(resid)

    for r in range(m):
        lag = r + 1
        if lag >= W:
            break
        nbr_resid = resid[:, :-lag, :]
        ak = align_k_win[:, lag:, r, :]
        has_match = (ak >= 0).float()
        safe_ak = ak.clamp(min=0).long()
        gathered = torch.gather(nbr_resid, dim=-1, index=safe_ak) * has_match
        bw = b_weights_win[:, lag:, r].unsqueeze(-1)
        pred_resid_from_nbrs[:, lag:, :] = pred_resid_from_nbrs[:, lag:, :] + bw * gathered

    whitened_resid = d_inv_sqrt_win.unsqueeze(-1) * (resid - pred_resid_from_nbrs)

    gls_mask = hit_mask.clone()
    if W > m:
        gls_mask[:, :m] = False
    if gls_mask.any():
        gls_loss = (whitened_resid ** 2).sum(dim=-1)[gls_mask].mean()
    else:
        gls_loss = (whitened_resid ** 2).sum(dim=-1)[hit_mask].mean() if hit_mask.any() else logits_win.sum() * 0.0

    if hit_mask.any():
        ce_per_obs = F.cross_entropy(logits_win[hit_mask], y_idx_win[hit_mask].long(), reduction="none")
        w_obs = d_inv_sqrt_win[hit_mask] ** 2
        w_obs = w_obs / w_obs.mean().clamp(min=1e-6)
        ce_loss = (ce_per_obs * w_obs).mean()
    else:
        ce_loss = logits_win.sum() * 0.0

    total = gls_loss + ce_weight * ce_loss
    return total, {
        "gls_loss": float(gls_loss.detach().cpu()),
        "wce_loss": float(ce_loss.detach().cpu()),
    }


def apply_nngp_kriging_inference(
    raw_logits: np.ndarray,
    cand_valid: np.ndarray,
    whitening: Dict[str, np.ndarray],
    trip_starts: np.ndarray,
    trip_ends: np.ndarray,
    krige_strength: float = 0.25,
) -> np.ndarray:
    """Apply inference-time NNGP Spatial Residual Kriging along matched physical arc_ids."""
    N, K = raw_logits.shape
    kriged = raw_logits.astype(np.float64).copy()
    nbr_idx = whitening["nbr_idx"]
    b_weights = whitening["b_weights"]
    align_k = whitening["align_k"]

    if krige_strength > 1e-6:
        valid_f = cand_valid.astype(np.float64)
        mean_l = np.where(cand_valid, raw_logits, 0.0).sum(axis=1, keepdims=True) / np.maximum(valid_f.sum(axis=1, keepdims=True), 1.0)
        centered = np.where(cand_valid, raw_logits - mean_l, 0.0)

        m = nbr_idx.shape[1]
        for s, e in zip(trip_starts, trip_ends):
            s, e = int(s), int(e)
            for t in range(s, e):
                acc = np.zeros(K, dtype=np.float64)
                for r in range(m):
                    nb = int(nbr_idx[t, r])
                    if nb < s:
                        continue
                    bw = float(b_weights[t, r])
                    if abs(bw) < 1e-6:
                        continue
                    ak_row = align_k[t, r]
                    for k in range(K):
                        kp = int(ak_row[k])
                        if kp >= 0:
                            acc[k] += bw * centered[nb, kp]
                kriged[t] = np.where(cand_valid[t], raw_logits[t] + krige_strength * acc, -1e4)

    kriged = np.where(cand_valid, kriged, -1e9)
    lmax = kriged.max(axis=1, keepdims=True)
    ex = np.where(cand_valid, np.exp(kriged - lmax), 0.0)
    probs = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)
    return probs.astype(np.float32)
