"""RLCD Core Module — Reinforcement Learning for Calibrated Decisions.

Implements:
1. Precomputed directed road-network step connectivity matrices step_conn[t, i, j] for fast GPU RL rollouts.
2. Group-Relative Policy Optimization (GRPO / REINFORCE with group baseline) using:
   - Strictly Proper Scoring Rule Reward (Negative Multi-Class Brier Score + Asymmetric Bluff Penalty)
   - Sequential Directed Road-Network Continuity Reward (rewarding connected arcs, penalizing reverse-arc flips)
   - Differentiable Soft-ECE Calibration Penalty
   - KL Divergence Anchor to Stage 1 NN-GLS Policy (for Model 4: NN-GLS + RLCD)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data_processing"))
from config import GAP_THRESHOLD_S
from io_utils import TripData, build_adjacency


@dataclass
class RLCDConfig:
    group_size: int = 8              # G parallel trajectory rollouts per window (GRPO baseline)
    w_brier_reward: float = 1.0      # weight on strictly proper Brier reward in RL return
    w_cont_reward: float = 0.45      # weight on road-network directed continuity reward
    w_hit_reward: float = 0.60       # weight on calibrated match reward
    pg_weight: float = 0.35          # policy-gradient loss weight
    brier_direct_weight: float = 1.2 # direct differentiable multi-class Brier score loss
    log_score_weight: float = 0.45   # strictly proper log-score (NLL) weight
    soft_ece_weight: float = 0.50    # differentiable Soft-ECE calibration penalty
    kl_anchor_weight: float = 0.12   # KL(pi_theta || pi_NN-GLS) anchor weight (0.0 for standalone RLCD)
    entropy_bonus: float = 0.005     # tiny exploration bonus to avoid policy collapse
    n_ece_bins: int = 10


def precompute_step_connectivity_trip(
    trip: TripData,
    cand_arcs: np.ndarray,
    cand_valid: np.ndarray,
) -> np.ndarray:
    """Precompute step_conn[t, i, j] in {-0.5, 0.0, 0.5, 1.0} for transition cand[t-1, i] -> cand[t, j].

    Values:
      +1.0 : Same directed arc OR direct 1-hop directed road transition (prev.to_node == curr.from_node)
      +0.5 : Reachable within 2 directed hops along the road network
      -0.5 : Immediate reverse-direction flip on a bidirectional road (prev.u == curr.v and prev.v == curr.u)
       0.0 : Disconnected / across GPS gap > GAP_THRESHOLD_S
    """
    T, K = cand_arcs.shape
    conn = np.zeros((T, K, K), dtype=np.float32)
    if T <= 1:
        return conn

    adj, _, _, _ = build_adjacency(trip)
    out_nbrs = [set(v for v, _, _ in adj[u]) for u in range(len(adj))]
    dt = np.diff(trip.track_ts)

    safe_aids = np.maximum(cand_arcs, 0)
    from_nd = trip.arcs[safe_aids, 0]
    to_nd = trip.arcs[safe_aids, 1]

    for t in range(1, T):
        if dt[t - 1] > GAP_THRESHOLD_S:
            continue
        for i in range(K):
            if not cand_valid[t - 1, i]:
                continue
            a1 = int(cand_arcs[t - 1, i])
            u1 = int(from_nd[t - 1, i])
            v1 = int(to_nd[t - 1, i])
            v1_nbrs = out_nbrs[v1]
            for j in range(K):
                if not cand_valid[t, j]:
                    continue
                a2 = int(cand_arcs[t, j])
                u2 = int(from_nd[t, j])
                v2 = int(to_nd[t, j])
                if a1 == a2 or v1 == u2:
                    conn[t, i, j] = 1.0
                elif u2 == v1 and v2 == u1:
                    conn[t, i, j] = -0.5
                elif u2 in v1_nbrs:
                    conn[t, i, j] = 0.5
    return conn


def compute_soft_ece_loss(
    probs: torch.Tensor,
    y_idx: torch.Tensor,
    hit_mask: torch.Tensor,
    n_bins: int = 10,
    sharpness: float = 25.0,
) -> torch.Tensor:
    """Differentiable Soft-ECE penalty using Gaussian soft-binning over top-1 confidence."""
    if not hit_mask.any():
        return probs.sum() * 0.0

    p = probs[hit_mask]                # (M, K)
    y = y_idx[hit_mask].long()         # (M,)
    conf, pred = p.max(dim=-1)         # (M,)
    acc = (pred == y).float()          # (M,)

    centers = torch.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins, device=probs.device)
    # Soft assignment weights (M, n_bins)
    weights = F.softmax(-sharpness * (conf.unsqueeze(-1) - centers.unsqueeze(0)) ** 2, dim=-1)
    bin_mass = weights.sum(dim=0)      # (n_bins,)
    total_m = float(len(conf))

    avg_conf = (weights * conf.unsqueeze(-1)).sum(dim=0) / bin_mass.clamp(min=1e-6)
    avg_acc = (weights * acc.unsqueeze(-1)).sum(dim=0) / bin_mass.clamp(min=1e-6)

    # Smooth L1 / Huber-like calibration gap weighted by bin fraction
    soft_ece = ((bin_mass / max(total_m, 1.0)) * torch.abs(avg_conf - avg_acc)).sum()
    return soft_ece


def compute_rlcd_window_loss(
    logits_win: torch.Tensor,
    y_idx_win: torch.Tensor,
    valid_win: torch.Tensor,
    step_conn_win: torch.Tensor,
    cfg: RLCDConfig,
    ref_logits_win: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute RLCD objective on contiguous trajectory windows (B, W, K).

    Combines:
      1. GRPO Policy Gradient over G sampled trajectory rollouts with strictly proper
         confidence-aware reward + directed road-network continuity reward.
      2. Direct strictly proper Multi-Class Brier Score + Log-Score loss.
      3. Differentiable Soft-ECE calibration regularization.
      4. KL divergence anchor to frozen Stage 1 NN-GLS reference policy (when ref_logits_win is supplied).
    """
    B, W, K = logits_win.shape
    G = cfg.group_size

    log_probs = F.log_softmax(logits_win, dim=-1)
    probs = torch.exp(log_probs) * valid_win.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    hit_mask = y_idx_win >= 0  # (B, W)
    safe_y = y_idx_win.clamp(min=0).long()
    y_oh = F.one_hot(safe_y, num_classes=K).float() * hit_mask.unsqueeze(-1).float()

    # --- 1. GRPO Group-Relative Rollouts (no grad through sampling/rewards) ---
    with torch.no_grad():
        flat_p = probs.reshape(B * W, K).clamp(min=1e-7)
        flat_p = flat_p / flat_p.sum(dim=-1, keepdim=True)
        # Sample G parallel action trajectories: (B, W, G)
        actions = torch.multinomial(flat_p, num_samples=G, replacement=True).reshape(B, W, G)

        # Confidence of each sampled action: p(a_t^(g) | s_t), shape (B, W, G)
        chosen_p = torch.gather(probs.unsqueeze(-1).expand(B, W, K, G), dim=2, index=actions.unsqueeze(2)).squeeze(2)
        is_correct = (actions == safe_y.unsqueeze(-1)).float() * hit_mask.unsqueeze(-1).float()

        # Strictly proper confidence-aware action reward:
        #   Correct at prob p -> +1 - (1 - p)^2  (rewards justified confidence)
        #   Wrong at prob p   -> -p^2 - 0.35     (heavily penalizes overconfident bluffs!)
        action_brier_r = torch.where(
            is_correct > 0.5,
            1.0 - (1.0 - chosen_p) ** 2,
            -(chosen_p ** 2) - 0.35,
        ) * hit_mask.unsqueeze(-1).float()

        # Sequential directed road-network continuity reward between a_{t-1}^(g) and a_t^(g)
        cont_r = torch.zeros((B, W, G), device=logits_win.device)
        if W > 1:
            prev_a = actions[:, :-1, :]  # (B, W-1, G)
            curr_a = actions[:, 1:, :]   # (B, W-1, G)
            # Lookup step_conn_win[b, t, prev_a, curr_a] for t=1..W-1
            b_idx = torch.arange(B, device=logits_win.device)[:, None, None].expand(B, W - 1, G)
            t_idx = torch.arange(1, W, device=logits_win.device)[None, :, None].expand(B, W - 1, G)
            cont_r[:, 1:, :] = step_conn_win[b_idx, t_idx, prev_a, curr_a]

        total_r = (
            cfg.w_brier_reward * action_brier_r
            + cfg.w_hit_reward * is_correct
            + cfg.w_cont_reward * cont_r
        )
        # GRPO group-relative normalized advantage across the G rollouts at each (b, w)
        r_mean = total_r.mean(dim=-1, keepdim=True)
        r_std = total_r.std(dim=-1, keepdim=True).clamp(min=1e-4)
        adv = (total_r - r_mean) / r_std  # (B, W, G)

    # Policy gradient loss: -E[ A_t^(g) * log pi(a_t^(g) | s_t) ]
    chosen_logp = torch.gather(log_probs.unsqueeze(-1).expand(B, W, K, G), dim=2, index=actions.unsqueeze(2)).squeeze(2)
    pg_loss = -(adv * chosen_logp).mean()

    # --- 2. Direct Strictly Proper Scoring Rule Losses (Multi-class Brier + Log-Score) ---
    if hit_mask.any():
        brier_loss = ((probs[hit_mask] - y_oh[hit_mask]) ** 2).sum(dim=-1).mean()
        log_score_loss = F.cross_entropy(logits_win[hit_mask], safe_y[hit_mask])
    else:
        brier_loss = logits_win.sum() * 0.0
        log_score_loss = logits_win.sum() * 0.0

    # --- 3. Differentiable Soft-ECE Calibration Penalty ---
    soft_ece = compute_soft_ece_loss(probs, y_idx_win, hit_mask, n_bins=cfg.n_ece_bins)

    # --- 4. KL Divergence Anchor to Stage 1 NN-GLS Policy ---
    if ref_logits_win is not None and cfg.kl_anchor_weight > 0.0:
        with torch.no_grad():
            ref_logp = F.log_softmax(ref_logits_win, dim=-1)
        # D_KL(pi_theta || pi_ref) = sum_k pi_theta * (log pi_theta - log pi_ref)
        kl_per_step = (probs * (log_probs - ref_logp)).masked_fill(~valid_win, 0.0).sum(dim=-1)
        kl_loss = kl_per_step.mean()
    else:
        kl_loss = logits_win.sum() * 0.0

    # Small entropy regularization
    entropy = -(probs * log_probs).masked_fill(~valid_win, 0.0).sum(dim=-1).mean()

    total_loss = (
        cfg.pg_weight * pg_loss
        + cfg.brier_direct_weight * brier_loss
        + cfg.log_score_weight * log_score_loss
        + cfg.soft_ece_weight * soft_ece
        + cfg.kl_anchor_weight * kl_loss
        - cfg.entropy_bonus * entropy
    )

    return total_loss, {
        "pg_loss": float(pg_loss.detach().cpu()),
        "brier_loss": float(brier_loss.detach().cpu()),
        "log_score": float(log_score_loss.detach().cpu()),
        "soft_ece": float(soft_ece.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "mean_reward": float(total_r.mean().detach().cpu()),
    }
