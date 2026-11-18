"""Shared training & sliding-window inference utilities for Model 4 (NN-GLS + RLCD)."""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data_processing"))
from config import DIR_CACHE, RANDOM_SEED, STUDENT_TEST_DIR, TRAIN_DIR
from io_utils import load_trip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nngls_core import (
    NNGLSSpatioTemporalNet,
    NNGPConfig,
    apply_nngp_kriging_inference,
    compute_nngls_window_loss,
)
from rlcd_core import RLCDConfig, compute_rlcd_window_loss, precompute_step_connectivity_trip


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached_partition(partition: str) -> Dict[str, np.ndarray]:
    path = DIR_CACHE / f"features_{partition}.npz"
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_or_build_step_connectivity(partition: str, part_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Load or compute (N, K, K) directed road-network step connectivity strictly from raw files."""
    cache_path = DIR_CACHE / f"step_conn_{partition}.npz"
    if cache_path.exists():
        return np.load(cache_path)["step_conn"]

    pdir = TRAIN_DIR if partition == "train" else STUDENT_TEST_DIR
    trip_ids = part_data["trip_ids"]
    starts = part_data["trip_starts"]
    ends = part_data["trip_ends"]
    cand_arcs = part_data["cand_arcs"]
    cand_valid = part_data["cand_valid"]
    N, K = cand_arcs.shape
    step_conn = np.zeros((N, K, K), dtype=np.float32)

    for tid, s, e in zip(trip_ids, starts, ends):
        s, e = int(s), int(e)
        trip = load_trip(pdir, str(tid), partition=partition, load_route=False)
        step_conn[s:e] = precompute_step_connectivity_trip(trip, cand_arcs[s:e], cand_valid[s:e])

    np.savez_compressed(cache_path, step_conn=step_conn)
    return step_conn


def fit_feature_scaler(
    X: np.ndarray,
    cand_valid: np.ndarray,
    trip_indices: np.ndarray,
    trip_starts: np.ndarray,
    trip_ends: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler strictly on valid candidate rows of training trips only."""
    chunks = []
    for idx in trip_indices:
        s, e = int(trip_starts[idx]), int(trip_ends[idx])
        v = cand_valid[s:e]
        chunks.append(X[s:e][v])
    flat = np.concatenate(chunks, axis=0).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_feature_scaler(
    X: np.ndarray,
    cand_valid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    Xs = (X.astype(np.float32) - mean[None, None, :]) / std[None, None, :]
    Xs = np.clip(Xs, -10.0, 10.0)
    Xs[~cand_valid] = 0.0
    return Xs


class TrajectoryWindowDataset(Dataset):
    """Contiguous trajectory window dataset that NEVER crosses trip boundaries."""

    def __init__(
        self,
        X_scaled: np.ndarray,
        y_idx: np.ndarray,
        cand_valid: np.ndarray,
        b_weights: np.ndarray,
        d_inv_sqrt: np.ndarray,
        align_k: np.ndarray,
        step_conn: np.ndarray,
        trip_indices: np.ndarray,
        trip_starts: np.ndarray,
        trip_ends: np.ndarray,
        window_len: int = 48,
        stride: int = 24,
        max_windows_per_trip: Optional[int] = None,
    ) -> None:
        self.X = X_scaled
        self.y_idx = y_idx
        self.valid = cand_valid
        self.b_weights = b_weights
        self.d_inv_sqrt = d_inv_sqrt
        self.align_k = align_k
        self.step_conn = step_conn
        self.W = window_len
        self.K = X_scaled.shape[1]
        self.F = X_scaled.shape[2]
        self.m = b_weights.shape[1]

        self.windows: List[Tuple[int, int]] = []
        for idx in trip_indices:
            s, e = int(trip_starts[idx]), int(trip_ends[idx])
            T = e - s
            if T <= window_len:
                trip_wins = [(s, T)]
            else:
                trip_wins = []
                pos = s
                while pos + window_len <= e:
                    trip_wins.append((pos, window_len))
                    pos += stride
                if pos < e and (e - window_len) > trip_wins[-1][0]:
                    trip_wins.append((e - window_len, window_len))
            # Prevent single massive outlier trips (e.g. 19k-pt 00000054) from dominating training gradients
            if max_windows_per_trip is not None and len(trip_wins) > max_windows_per_trip:
                sel = np.linspace(0, len(trip_wins) - 1, num=max_windows_per_trip, dtype=int)
                trip_wins = [trip_wins[k] for k in sel]
            self.windows.extend(trip_wins)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s, L = self.windows[i]
        W = self.W
        if L == W:
            return {
                "x": torch.from_numpy(self.X[s : s + W]),
                "y": torch.from_numpy(self.y_idx[s : s + W]),
                "valid": torch.from_numpy(self.valid[s : s + W]),
                "b_w": torch.from_numpy(self.b_weights[s : s + W]),
                "d_inv": torch.from_numpy(self.d_inv_sqrt[s : s + W]),
                "align_k": torch.from_numpy(self.align_k[s : s + W]),
                "conn": torch.from_numpy(self.step_conn[s : s + W]),
            }

        x = np.zeros((W, self.K, self.F), dtype=np.float32)
        y = np.full(W, -1, dtype=np.int32)
        valid = np.zeros((W, self.K), dtype=np.bool_)
        b_w = np.zeros((W, self.m), dtype=np.float32)
        d_inv = np.ones(W, dtype=np.float32)
        ak = np.full((W, self.m, self.K), -1, dtype=np.int16)
        conn = np.zeros((W, self.K, self.K), dtype=np.float32)

        x[:L] = self.X[s : s + L]
        y[:L] = self.y_idx[s : s + L]
        valid[:L] = self.valid[s : s + L]
        b_w[:L] = self.b_weights[s : s + L]
        d_inv[:L] = self.d_inv_sqrt[s : s + L]
        ak[:L] = self.align_k[s : s + L]
        conn[:L] = self.step_conn[s : s + L]

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "valid": torch.from_numpy(valid),
            "b_w": torch.from_numpy(b_w),
            "d_inv": torch.from_numpy(d_inv),
            "align_k": torch.from_numpy(ak),
            "conn": torch.from_numpy(conn),
        }


@torch.no_grad()
def predict_trip_logits_sliding(
    model: nn.Module,
    X_trip: np.ndarray,
    valid_trip: np.ndarray,
    b_w_trip: np.ndarray,
    align_k_trip: np.ndarray,
    conn_trip: np.ndarray,
    device: torch.device,
    window_len: int = 64,
    stride: int = 32,
    batch_size: int = 64,
) -> np.ndarray:
    """Overlap-averaged sliding-window inference on a single trip (zero boundary artifacts)."""
    model.eval()
    T, K, F = X_trip.shape
    m = b_w_trip.shape[1]

    if T <= window_len:
        xb = torch.from_numpy(X_trip[None]).to(device)
        vb = torch.from_numpy(valid_trip[None]).to(device)
        bb = torch.from_numpy(b_w_trip[None]).to(device)
        ab = torch.from_numpy(align_k_trip[None]).to(device)
        cb = torch.from_numpy(conn_trip[None]).to(device)
        return model(xb, vb, bb, ab, cb)[0].cpu().numpy()

    starts = list(range(0, T - window_len + 1, stride))
    if starts[-1] + window_len < T:
        starts.append(T - window_len)

    # Raised-cosine / triangular interior blending weight favors central context steps
    pos_grid = np.linspace(0.0, 1.0, window_len, dtype=np.float64)
    win_weight = 0.25 + 0.75 * np.sin(np.pi * pos_grid)  # peak 1.0 at center, 0.25 at edges

    logit_acc = np.zeros((T, K), dtype=np.float64)
    weight_acc = np.zeros((T, 1), dtype=np.float64)

    for b0 in range(0, len(starts), batch_size):
        b_starts = starts[b0 : b0 + batch_size]
        B = len(b_starts)
        xs = np.stack([X_trip[s : s + window_len] for s in b_starts], axis=0)
        vs = np.stack([valid_trip[s : s + window_len] for s in b_starts], axis=0)
        bs = np.stack([b_w_trip[s : s + window_len] for s in b_starts], axis=0)
        aks = np.stack([align_k_trip[s : s + window_len] for s in b_starts], axis=0)
        cs = np.stack([conn_trip[s : s + window_len] for s in b_starts], axis=0)

        out = model(
            torch.from_numpy(xs).to(device, non_blocking=True),
            torch.from_numpy(vs).to(device, non_blocking=True),
            torch.from_numpy(bs).to(device, non_blocking=True),
            torch.from_numpy(aks).to(device, non_blocking=True),
            torch.from_numpy(cs).to(device, non_blocking=True),
        ).cpu().numpy()

        for i, s in enumerate(b_starts):
            logit_acc[s : s + window_len] += out[i] * win_weight[:, None]
            weight_acc[s : s + window_len] += win_weight[:, None]

    avg_logits = (logit_acc / np.maximum(weight_acc, 1e-8)).astype(np.float32)
    avg_logits[~valid_trip] = -1e4
    return avg_logits


def predict_partition_probs(
    model: nn.Module,
    X_scaled: np.ndarray,
    cand_valid: np.ndarray,
    whitening: Dict[str, np.ndarray],
    step_conn: np.ndarray,
    trip_indices: np.ndarray,
    trip_starts: np.ndarray,
    trip_ends: np.ndarray,
    device: torch.device,
    krige_strength: float = 0.25,
) -> Tuple[np.ndarray, float]:
    """Run sliding-window neural inference + NNGP spatial residual kriging across selected trips.

    Returns (probs_subset, total_inference_ms).
    """
    t0 = time.perf_counter()
    # NNGP spatial residual kriging + temperature calibration are now integrated directly inside forward(),
    # so we simply apply stable masked softmax with zero train/inference calibration mismatch!
    probs_chunks = []
    for idx in trip_indices:
        s, e = int(trip_starts[idx]), int(trip_ends[idx])
        logits = predict_trip_logits_sliding(
            model,
            X_scaled[s:e],
            cand_valid[s:e],
            whitening["b_weights"][s:e],
            whitening["align_k"][s:e],
            step_conn[s:e],
            device=device,
        )
        v = cand_valid[s:e]
        l = np.where(v, logits.astype(np.float64), -1e9)
        m = l.max(axis=1, keepdims=True)
        ex = np.where(v, np.exp(l - m), 0.0)
        p_trip = (ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
        probs_chunks.append(p_trip)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.concatenate(probs_chunks, axis=0), elapsed_ms


def _eval_validation_metrics(
    model: nn.Module,
    val_loader: Optional[DataLoader],
    device: torch.device,
    mode: str,
    ce_weight: float = 0.55,
    rlcd_cfg: Optional[RLCDConfig] = None,
    ref_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    if val_loader is None:
        return {}
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total_hits = 0
    brier_sum = 0.0
    nll_sum = 0.0
    cont_ok = 0.0
    cont_tot = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            v = batch["valid"].to(device, non_blocking=True)
            bw = batch["b_w"].to(device, non_blocking=True)
            di = batch["d_inv"].to(device, non_blocking=True)
            ak = batch["align_k"].to(device, non_blocking=True)
            cn = batch["conn"].to(device, non_blocking=True)

            logits = model(x, v, bw, ak, cn)

            if mode == "nngls":
                loss, _ = compute_nngls_window_loss(logits, y, v, bw, di, ak, ce_weight=ce_weight)
            else:
                ref_logits = ref_model(x, v, bw, ak, cn) if ref_model is not None else None
                loss, _ = compute_rlcd_window_loss(logits, y, v, cn, cfg=rlcd_cfg or RLCDConfig(), ref_logits_win=ref_logits)

            total_loss += float(loss.item())
            n_batches += 1

            hm = y >= 0
            if hm.any():
                lh = logits[hm]
                yh = y[hm].long()
                ph = F.softmax(lh, dim=-1)
                correct += int((ph.argmax(dim=-1) == yh).sum().item())
                total_hits += int(yh.numel())
                oh = F.one_hot(yh, num_classes=logits.shape[-1]).float()
                brier_sum += float(((ph - oh) ** 2).sum(dim=-1).sum().item())
                nll_sum += float(F.cross_entropy(lh, yh, reduction="sum").item())

            # Quick local step continuity tracking on validation windows
            preds = logits.argmax(dim=-1)  # (B, W)
            valid_step = v.any(dim=-1)
            both_v = valid_step[:, 1:] & valid_step[:, :-1]
            if both_v.any():
                B, W = preds.shape
                b_i = torch.arange(B, device=device)[:, None].expand(B, W - 1)
                t_i = torch.arange(1, W, device=device)[None, :].expand(B, W - 1)
                c_vals = cn[b_i, t_i, preds[:, :-1], preds[:, 1:]]
                # Only count non-gap pairs where at least one connection exists in step_conn
                has_graph = (cn[:, 1:].abs().amax(dim=(-1, -2)) > 0) & both_v
                if has_graph.any():
                    cont_ok += float((c_vals[has_graph] > 0.0).float().sum().item())
                    cont_tot += float(has_graph.float().sum().item())

    return {
        "val_loss": round(total_loss / max(n_batches, 1), 5),
        "val_hit_acc": round(correct / max(total_hits, 1), 5),
        "val_brier": round(brier_sum / max(total_hits, 1), 5),
        "val_nll": round(nll_sum / max(total_hits, 1), 5),
        "val_local_cont": round(cont_ok / max(cont_tot, 1.0), 5),
    }


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    mode: str,
    epochs: int = 20,
    lr: float = 1.2e-3,
    weight_decay: float = 1.5e-4,
    ce_weight: float = 0.55,
    rlcd_cfg: Optional[RLCDConfig] = None,
    ref_model: Optional[nn.Module] = None,
    patience: int = 6,
) -> Tuple[nn.Module, List[Dict], float]:
    """Train Stage 1 (`nngls`) or Stage 2 (`rlcd`) on GPU with cosine LR & best-val selection."""
    t0 = time.perf_counter()
    model = model.to(device)
    if ref_model is not None:
        ref_model = ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Give calibration temperature faster LR during Stage 2 RLCD for surgical calibration
    if mode == "rlcd" and hasattr(model, "log_temp"):
        temp_params = [model.log_temp]
        other_params = [p for n, p in model.named_parameters() if n != "log_temp"]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
                {"params": temp_params, "lr": lr * 4.0, "weight_decay": 0.0},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    history: List[Dict] = []
    best_state = copy.deepcopy(model.state_dict())
    best_monitor = float("inf")
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        comp_accum: Dict[str, float] = {}
        n_b = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            v = batch["valid"].to(device, non_blocking=True)
            bw = batch["b_w"].to(device, non_blocking=True)
            di = batch["d_inv"].to(device, non_blocking=True)
            ak = batch["align_k"].to(device, non_blocking=True)
            cn = batch["conn"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x, v, bw, ak, cn)

            logits = logits.float()
            if mode == "nngls":
                loss, comp = compute_nngls_window_loss(logits, y, v, bw, di, ak, ce_weight=ce_weight)
            elif mode == "rlcd":
                ref_logits = None
                if ref_model is not None:
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        ref_logits = ref_model(x, v, bw, ak, cn).float()
                loss, comp = compute_rlcd_window_loss(logits, y, v, cn, cfg=rlcd_cfg or RLCDConfig(), ref_logits_win=ref_logits)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            ep_loss += float(loss.item())
            for k, val in comp.items():
                comp_accum[k] = comp_accum.get(k, 0.0) + val
            n_b += 1

        scheduler.step()
        rec: Dict = {
            "epoch": ep,
            "stage": mode,
            "train_loss": round(ep_loss / max(n_b, 1), 5),
        }
        for k, val in comp_accum.items():
            rec[f"train_{k}"] = round(val / max(n_b, 1), 5)

        if val_loader is not None:
            vm = _eval_validation_metrics(model, val_loader, device, mode, ce_weight, rlcd_cfg, ref_model)
            rec.update(vm)
            # Proper scoring selection criterion: NLL + 0.5*Brier (lower is strictly better)
            monitor = vm["val_nll"] + 0.5 * vm["val_brier"]
            if monitor < best_monitor - 1e-4:
                best_monitor = monitor
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
        else:
            best_state = copy.deepcopy(model.state_dict())

        history.append(rec)
        if val_loader is not None and no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, history, round(time.perf_counter() - t0, 2)
