"""Stage 03: End-to-End Training & Evaluation of Model 4 — Pure NN-GLS + RLCD (Zero HMM/Bayesian HPO).

Workflow:
  1. Load cached 28-feature candidate tensors (strictly from .track/.nodes/.arcs) for train (80) & student_test (10).
  2. Precompute exact Vecchia/NNGP spatial-temporal whitening operators L^{-1} = D^{-1/2}(I - B)
     and directed road-network step connectivity matrices step_conn[t, i, j].
  3. Run strict 5-fold GroupKFold(trip_id) Cross-Validation on the 80 train trips:
       - Fold-internal StandardScaler (zero leakage)
       - Stage 1: NN-GLS training (Vecchia GLS decorrelated loss + NNGP precision-weighted CE)
       - Stage 2: RLCD post-training (GRPO strictly proper Brier + road-network continuity + Soft-ECE + KL anchor)
       - Log both Stage 1 (pre-RLCD) and Stage 2 (final Model 4) OOF metrics & genuine epoch histories.
  4. Refit final Model 4 (NN-GLS + RLCD) on all 80 train trips and save production artifacts.
  5. Evaluate final Model 4 on the 10 held-out student_test trips (with gradient saliency analysis).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_data_processing"))
from config import (
    DIR_ARTIFACTS,
    DIR_RESULTS,
    N_CV_FOLDS,
    RANDOM_SEED,
    STUDENT_TEST_DIR,
    TRAIN_DIR,
)
from io_utils import build_adjacency, load_trip

sys.path.insert(0, str(ROOT / "02_feature_engineering"))
from feature_extractor import FEATURE_NAMES

sys.path.insert(0, str(ROOT / "03_model_training"))
from nngls_core import (
    NNGLSSpatioTemporalNet,
    NNGPConfig,
    build_nngp_whitening_all,
)
from rlcd_core import RLCDConfig
from train_utils import (
    TrajectoryWindowDataset,
    apply_feature_scaler,
    fit_feature_scaler,
    load_cached_partition,
    load_or_build_step_connectivity,
    predict_partition_probs,
    set_all_seeds,
    train_stage,
)

sys.path.insert(0, str(ROOT / "06_model_evaluation"))
from metrics_model4 import ShortestPathEvaluator, evaluate_predictions


# Model 4: NN-GLS + RLCD theoretically grounded configuration
MODEL4_CONFIG = {
    "model_name": "Model 4: NN-GLS + RLCD",
    "algorithm_family": "Geostatistical Graph Neural Network (NN-GLS) + Calibrated Policy-Gradient RL (RLCD)",
    "architecture": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "window_len": 48,
        "train_stride": 24,
        "eval_window_len": 64,
        "eval_stride": 32,
    },
    "nngp": {
        "m_neighbors": 5,
        "rho_spatial_m": 45.0,
        "rho_temporal_s": 6.0,
        "sigma2": 0.85,
        "tau2": 0.15,
        "gap_threshold_s": 5.0,
        "krige_strength": 0.22,
    },
    "stage1_nngls": {
        "epochs": 20,
        "lr": 1.3e-3,
        "weight_decay": 1.5e-4,
        "ce_weight": 0.60,
        "batch_size": 128,
        "max_windows_per_trip": 140,
        "patience": 5,
    },
    "stage2_rlcd": {
        "epochs": 12,
        "lr": 3.5e-4,
        "weight_decay": 1.0e-4,
        "batch_size": 128,
        "max_windows_per_trip": 140,
        "patience": 5,
        "group_size": 8,
        "w_brier_reward": 1.25,
        "w_cont_reward": 0.55,
        "w_hit_reward": 0.20,
        "pg_weight": 0.18,
        "brier_direct_weight": 1.60,
        "log_score_weight": 1.00,
        "soft_ece_weight": 1.50,
        "kl_anchor_weight": 0.15,
        "entropy_bonus": 0.002,
        "n_ece_bins": 10,
    },
    "random_seed": RANDOM_SEED,
    "cv_folds": N_CV_FOLDS,
}


def compute_gradient_feature_saliency(
    model: NNGLSSpatioTemporalNet,
    X_scaled: np.ndarray,
    cand_valid: np.ndarray,
    whitening: Dict[str, np.ndarray],
    step_conn: np.ndarray,
    trip_starts: np.ndarray,
    trip_ends: np.ndarray,
    device: torch.device,
) -> List[Dict]:
    """Compute mean absolute input-gradient saliency across the 28 features on student_test."""
    model.eval()
    F_dim = X_scaled.shape[2]
    grad_acc = np.zeros(F_dim, dtype=np.float64)
    count = 0

    for s, e in zip(trip_starts, trip_ends):
        s, e = int(s), int(e)
        W = min(64, e - s)
        # Sample up to 8 windows per test trip
        starts = np.linspace(s, e - W, num=min(8, max(1, (e - s) // W)), dtype=int)
        for ws in starts:
            xb = torch.from_numpy(X_scaled[ws : ws + W][None]).to(device).requires_grad_(True)
            vb = torch.from_numpy(cand_valid[ws : ws + W][None]).to(device)
            bb = torch.from_numpy(whitening["b_weights"][ws : ws + W][None]).to(device)
            ab = torch.from_numpy(whitening["align_k"][ws : ws + W][None]).to(device)
            cb = torch.from_numpy(step_conn[ws : ws + W][None]).to(device)

            logits = model(xb, vb, bb, ab, cb)
            top_logit = logits.max(dim=-1).values.sum()
            top_logit.backward()
            g = xb.grad.detach().abs().cpu().numpy()[0]  # (W, K, F)
            v = cand_valid[ws : ws + W]
            grad_acc += g[v].sum(axis=0)
            count += int(v.sum())

    mean_sal = grad_acc / max(count, 1)
    norm_sal = mean_sal / max(mean_sal.sum(), 1e-12)
    rows = [
        {"feature": name, "mean_abs_grad": round(float(ms), 6), "relative_importance": round(float(ns), 5)}
        for name, ms, ns in zip(FEATURE_NAMES, mean_sal, norm_sal)
    ]
    rows.sort(key=lambda r: r["relative_importance"], reverse=True)
    return rows


def main() -> None:
    t_global = time.time()
    set_all_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model 4: NN-GLS + RLCD] Running on device: {device}")

    # 1. Load cached partitions
    train_data = load_cached_partition("train")
    test_data = load_cached_partition("student_test")

    print("[Step 1] Loading/building directed road-network step connectivity...")
    t_conn = time.time()
    train_conn = load_or_build_step_connectivity("train", train_data)
    test_conn = load_or_build_step_connectivity("student_test", test_data)
    print(f"  step_conn ready in {time.time() - t_conn:.2f}s | train={train_conn.shape}, test={test_conn.shape}")

    # 2. Precompute exact Vecchia/NNGP spatial-temporal whitening operators
    print("[Step 2] Building Vecchia/NNGP spatial-temporal covariance whitening operators...")
    t_wh = time.time()
    nngp_cfg = NNGPConfig(**MODEL4_CONFIG["nngp"])
    train_wh = build_nngp_whitening_all(
        train_data["coords"], train_data["cand_arcs"], train_data["cand_valid"],
        train_data["trip_starts"], train_data["trip_ends"], nngp_cfg,
    )
    test_wh = build_nngp_whitening_all(
        test_data["coords"], test_data["cand_arcs"], test_data["cand_valid"],
        test_data["trip_starts"], test_data["trip_ends"], nngp_cfg,
    )
    print(f"  NNGP whitening built in {time.time() - t_wh:.2f}s")

    # Preload trip objects & shortest-path evaluators for route continuity / sequence metrics
    train_trips = [load_trip(TRAIN_DIR, str(tid), "train", load_route=True) for tid in train_data["trip_ids"]]
    test_trips = [load_trip(STUDENT_TEST_DIR, str(tid), "student_test", load_route=True) for tid in test_data["trip_ids"]]
    test_sp_evals = {tr.trip_id: ShortestPathEvaluator(build_adjacency(tr)[0]) for tr in test_trips}

    arch = MODEL4_CONFIG["architecture"]
    s1_cfg = MODEL4_CONFIG["stage1_nngls"]
    s2_cfg = MODEL4_CONFIG["stage2_rlcd"]
    rlcd_cfg = RLCDConfig(
        group_size=s2_cfg["group_size"],
        w_brier_reward=s2_cfg["w_brier_reward"],
        w_cont_reward=s2_cfg["w_cont_reward"],
        w_hit_reward=s2_cfg["w_hit_reward"],
        pg_weight=s2_cfg["pg_weight"],
        brier_direct_weight=s2_cfg["brier_direct_weight"],
        log_score_weight=s2_cfg["log_score_weight"],
        soft_ece_weight=s2_cfg["soft_ece_weight"],
        kl_anchor_weight=s2_cfg["kl_anchor_weight"],
        entropy_bonus=s2_cfg["entropy_bonus"],
        n_ece_bins=s2_cfg["n_ece_bins"],
    )

    X_tr = train_data["X"]
    y_tr = train_data["y_idx"]
    ref_tr = train_data["ref_arc"]
    arcs_tr = train_data["cand_arcs"]
    val_tr = train_data["cand_valid"]
    starts_tr = train_data["trip_starts"]
    ends_tr = train_data["trip_ends"]
    n_train_trips = len(train_data["trip_ids"])
    n_features = X_tr.shape[2]

    # 3. Strict 5-Fold GroupKFold CV on the 80 train trips
    print(f"\n[Step 3] Running {N_CV_FOLDS}-Fold GroupKFold(trip_id) CV on {n_train_trips} train trips...")
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    oof_probs_stage1 = np.zeros((len(y_tr), arcs_tr.shape[1]), dtype=np.float32)
    oof_probs_stage2 = np.zeros((len(y_tr), arcs_tr.shape[1]), dtype=np.float32)

    fold_rows: List[Dict] = []
    all_histories: Dict[str, List[Dict]] = {}
    best_ep_s1_list: List[int] = []
    best_ep_s2_list: List[int] = []
    cv_train_sec_total = 0.0

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.arange(n_train_trips), groups=np.arange(n_train_trips))):
        set_all_seeds(RANDOM_SEED + fold * 17)
        mean_f, std_f = fit_feature_scaler(X_tr, val_tr, tr_idx, starts_tr, ends_tr)
        Xs_tr = apply_feature_scaler(X_tr, val_tr, mean_f, std_f)

        tr_ds = TrajectoryWindowDataset(
            Xs_tr, y_tr, val_tr, train_wh["b_weights"], train_wh["d_inv_sqrt"], train_wh["align_k"],
            train_conn, tr_idx, starts_tr, ends_tr, window_len=arch["window_len"], stride=arch["train_stride"],
            max_windows_per_trip=s1_cfg["max_windows_per_trip"],
        )
        va_ds = TrajectoryWindowDataset(
            Xs_tr, y_tr, val_tr, train_wh["b_weights"], train_wh["d_inv_sqrt"], train_wh["align_k"],
            train_conn, va_idx, starts_tr, ends_tr, window_len=arch["window_len"], stride=arch["window_len"],
        )
        tr_loader = DataLoader(tr_ds, batch_size=s1_cfg["batch_size"], shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        # Stage 1: NN-GLS
        model = NNGLSSpatioTemporalNet(
            in_features=n_features,
            hidden_dim=arch["hidden_dim"],
            num_layers=arch["num_layers"],
            num_heads=arch["num_heads"],
            dropout=arch["dropout"],
        )
        model, hist_s1, t_s1 = train_stage(
            model, tr_loader, va_loader, device, mode="nngls",
            epochs=s1_cfg["epochs"], lr=s1_cfg["lr"], weight_decay=s1_cfg["weight_decay"],
            ce_weight=s1_cfg["ce_weight"], patience=s1_cfg["patience"],
        )
        best_s1_ep = int(min(hist_s1, key=lambda r: r["val_nll"] + 0.5 * r["val_brier"])["epoch"])
        best_ep_s1_list.append(best_s1_ep)

        # Evaluate Stage 1 OOF on held-out fold
        p_va_s1, _ = predict_partition_probs(
            model, Xs_tr, val_tr, train_wh, train_conn, va_idx, starts_tr, ends_tr,
            device=device, krige_strength=nngp_cfg.krige_strength,
        )
        va_obs = np.concatenate([np.arange(int(starts_tr[i]), int(ends_tr[i])) for i in va_idx])
        oof_probs_stage1[va_obs] = p_va_s1

        # Stage 2: RLCD post-training anchored to Stage 1 NN-GLS
        ref_nngls = copy.deepcopy(model)
        model, hist_s2, t_s2 = train_stage(
            model, tr_loader, va_loader, device, mode="rlcd",
            epochs=s2_cfg["epochs"], lr=s2_cfg["lr"], weight_decay=s2_cfg["weight_decay"],
            rlcd_cfg=rlcd_cfg, ref_model=ref_nngls, patience=s2_cfg["patience"],
        )
        best_s2_ep = int(min(hist_s2, key=lambda r: r["val_nll"] + 0.5 * r["val_brier"])["epoch"])
        best_ep_s2_list.append(best_s2_ep)
        cv_train_sec_total += t_s1 + t_s2

        p_va_s2, va_inf_ms = predict_partition_probs(
            model, Xs_tr, val_tr, train_wh, train_conn, va_idx, starts_tr, ends_tr,
            device=device, krige_strength=nngp_cfg.krige_strength,
        )
        oof_probs_stage2[va_obs] = p_va_s2

        # Compute fold metrics (Stage 1 vs Stage 2)
        va_trips = [train_trips[i] for i in va_idx]
        va_slices = []
        pos = 0
        for i in va_idx:
            L = int(ends_tr[i] - starts_tr[i])
            va_slices.append((pos, pos + L))
            pos += L
        va_sp = {tr.trip_id: ShortestPathEvaluator(build_adjacency(tr)[0]) for tr in va_trips}

        m_s1 = evaluate_predictions(
            p_va_s1, arcs_tr[va_obs], val_tr[va_obs], y_tr[va_obs], ref_tr[va_obs],
            trips=va_trips, trip_slices=va_slices, sp_evals=va_sp,
        )
        m_s2 = evaluate_predictions(
            p_va_s2, arcs_tr[va_obs], val_tr[va_obs], y_tr[va_obs], ref_tr[va_obs],
            trips=va_trips, trip_slices=va_slices, sp_evals=va_sp,
        )

        # Combine contiguous epoch timeline for plots
        combined_hist = []
        for r in hist_s1:
            combined_hist.append({**r, "global_epoch": r["epoch"]})
        offset_ep = len(hist_s1)
        for r in hist_s2:
            combined_hist.append({**r, "global_epoch": offset_ep + r["epoch"]})
        all_histories[f"fold_{fold}"] = combined_hist

        fold_rec = {
            "fold": fold,
            "n_val_trips": len(va_idx),
            "n_val_obs": len(va_obs),
            "stage1_nngls_macro_f1": m_s1["macro_f1"],
            "stage1_nngls_top1_acc": m_s1["obs_top1_accuracy"],
            "stage1_nngls_hit_acc": m_s1["obs_top1_accuracy_given_hit"],
            "stage1_nngls_log_loss": m_s1["log_loss"],
            "stage1_nngls_brier": m_s1["brier_score"],
            "stage1_nngls_ece": m_s1["ece"],
            "stage1_nngls_continuity": m_s1["route_continuity"],
            "model4_macro_f1": m_s2["macro_f1"],
            "model4_top1_acc": m_s2["obs_top1_accuracy"],
            "model4_hit_acc": m_s2["obs_top1_accuracy_given_hit"],
            "model4_log_loss": m_s2["log_loss"],
            "model4_brier": m_s2["brier_score"],
            "model4_ece": m_s2["ece"],
            "model4_continuity": m_s2["route_continuity"],
            "model4_continuity_1hop": m_s2["route_continuity_1hop"],
            "model4_seq_arc_f1": m_s2["seq_arc_f1"],
            "best_ep_s1": best_s1_ep,
            "best_ep_s2": best_s2_ep,
            "train_sec": round(t_s1 + t_s2, 2),
            "val_inf_ms_per_trip": round(va_inf_ms / len(va_idx), 2),
        }
        fold_rows.append(fold_rec)
        print(
            f"  Fold {fold} | S1(NN-GLS): F1={m_s1['macro_f1']:.4f} Top1={m_s1['obs_top1_accuracy']:.4f} "
            f"(Hit={m_s1['obs_top1_accuracy_given_hit']:.4f}) LL={m_s1['log_loss']:.4f} Brier={m_s1['brier_score']:.4f} "
            f"ECE={m_s1['ece']:.4f} Cont={m_s1['route_continuity']:.4f} -> "
            f"M4(NN-GLS+RLCD): F1={m_s2['macro_f1']:.4f} Top1={m_s2['obs_top1_accuracy']:.4f} "
            f"(Hit={m_s2['obs_top1_accuracy_given_hit']:.4f}) LL={m_s2['log_loss']:.4f} Brier={m_s2['brier_score']:.4f} "
            f"ECE={m_s2['ece']:.4f} Cont={m_s2['route_continuity']:.4f} ({fold_rec['train_sec']}s)",
            flush=True,
        )

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(DIR_RESULTS / "model4_cv_folds.csv", index=False)

    # Overall OOF evaluation across all 80 training trips (208,417 obs)
    tr_slices = [(int(s), int(e)) for s, e in zip(starts_tr, ends_tr)]
    oof_eval_s1 = evaluate_predictions(oof_probs_stage1, arcs_tr, val_tr, y_tr, ref_tr)
    oof_eval_s2 = evaluate_predictions(oof_probs_stage2, arcs_tr, val_tr, y_tr, ref_tr)
    oof_eval_s1["route_continuity"] = round(float(df_folds["stage1_nngls_continuity"].mean()), 5)
    oof_eval_s2["route_continuity"] = round(float(df_folds["model4_continuity"].mean()), 5)
    oof_eval_s2["route_continuity_1hop"] = round(float(df_folds["model4_continuity_1hop"].mean()), 5)
    oof_eval_s2["seq_arc_f1"] = round(float(df_folds["model4_seq_arc_f1"].mean()), 5)

    # 4. Final Refit on ALL 80 Training Trips (zero student_test leakage!)
    refit_ep_s1 = max(12, int(round(float(np.median(best_ep_s1_list)))))
    refit_ep_s2 = max(8, int(round(float(np.median(best_ep_s2_list)))))
    print(f"\n[Step 4] Refitting final Model 4 on all {n_train_trips} train trips (S1={refit_ep_s1} ep, S2={refit_ep_s2} ep)...")

    set_all_seeds(RANDOM_SEED)
    all_tr_idx = np.arange(n_train_trips)
    final_mean, final_std = fit_feature_scaler(X_tr, val_tr, all_tr_idx, starts_tr, ends_tr)
    Xs_tr_full = apply_feature_scaler(X_tr, val_tr, final_mean, final_std)
    Xs_te = apply_feature_scaler(test_data["X"], test_data["cand_valid"], final_mean, final_std)

    full_ds = TrajectoryWindowDataset(
        Xs_tr_full, y_tr, val_tr, train_wh["b_weights"], train_wh["d_inv_sqrt"], train_wh["align_k"],
        train_conn, all_tr_idx, starts_tr, ends_tr, window_len=arch["window_len"], stride=arch["train_stride"],
        max_windows_per_trip=s1_cfg["max_windows_per_trip"],
    )
    full_loader = DataLoader(full_ds, batch_size=s1_cfg["batch_size"], shuffle=True, drop_last=False, num_workers=2, pin_memory=True)

    final_model = NNGLSSpatioTemporalNet(
        in_features=n_features,
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        dropout=arch["dropout"],
    )
    final_model, full_hist_s1, t_refit_s1 = train_stage(
        final_model, full_loader, None, device, mode="nngls",
        epochs=refit_ep_s1, lr=s1_cfg["lr"], weight_decay=s1_cfg["weight_decay"],
        ce_weight=s1_cfg["ce_weight"],
    )
    stage1_snapshot = copy.deepcopy(final_model)

    final_model, full_hist_s2, t_refit_s2 = train_stage(
        final_model, full_loader, None, device, mode="rlcd",
        epochs=refit_ep_s2, lr=s2_cfg["lr"], weight_decay=s2_cfg["weight_decay"],
        rlcd_cfg=rlcd_cfg, ref_model=stage1_snapshot,
    )
    final_train_sec = round(t_refit_s1 + t_refit_s2, 2)
    learned_temp = round(float(torch.exp(final_model.log_temp.detach().clamp(-1.2, 1.2)).cpu().item()), 5)
    print(f"  Final refit complete in {final_train_sec}s | learned calibration temperature = {learned_temp}")

    # Save scaler & model artifacts
    scaler_payload = {
        "feature_names": FEATURE_NAMES,
        "mean": final_mean.tolist(),
        "std": final_std.tolist(),
        "clip_range": [-10.0, 10.0],
        "fit_partition": "train (80 trips only)",
    }
    with open(DIR_ARTIFACTS / "scaler_model4_nngls_rlcd.json", "w") as f:
        json.dump(scaler_payload, f, indent=2)

    ckpt_payload = {
        "state_dict": final_model.cpu().state_dict(),
        "config": MODEL4_CONFIG,
        "in_features": n_features,
        "feature_names": FEATURE_NAMES,
        "learned_temperature": learned_temp,
        "refit_epochs": {"stage1_nngls": refit_ep_s1, "stage2_rlcd": refit_ep_s2},
    }
    torch.save(ckpt_payload, DIR_ARTIFACTS / "model4_nngls_rlcd.pt")
    final_model = final_model.to(device)

    with open(DIR_ARTIFACTS / "model4_nngls_rlcd_config.json", "w") as f:
        json.dump(
            {
                **MODEL4_CONFIG,
                "in_features": n_features,
                "n_parameters": int(sum(p.numel() for p in final_model.parameters())),
                "learned_temperature": learned_temp,
                "refit_epochs": {"stage1_nngls": refit_ep_s1, "stage2_rlcd": refit_ep_s2},
                "final_train_sec": final_train_sec,
            },
            f,
            indent=2,
        )

    # 5. Evaluate on ALL 10 Held-Out Student-Test Trips
    print("\n[Step 5] Evaluating final Model 4 (NN-GLS + RLCD) on 10 held-out student_test trips...")
    te_idx = np.arange(len(test_data["trip_ids"]))
    te_slices = [(int(s), int(e)) for s, e in zip(test_data["trip_starts"], test_data["trip_ends"])]

    # Also evaluate Stage 1 (pre-RLCD) on student_test to quantify RLCD's exact post-training lift!
    probs_te_s1, inf_ms_s1 = predict_partition_probs(
        stage1_snapshot.to(device), Xs_te, test_data["cand_valid"], test_wh, test_conn,
        te_idx, test_data["trip_starts"], test_data["trip_ends"],
        device=device, krige_strength=nngp_cfg.krige_strength,
    )
    probs_te_m4, inf_ms_m4 = predict_partition_probs(
        final_model, Xs_te, test_data["cand_valid"], test_wh, test_conn,
        te_idx, test_data["trip_starts"], test_data["trip_ends"],
        device=device, krige_strength=nngp_cfg.krige_strength,
    )

    test_eval_s1 = evaluate_predictions(
        probs_te_s1, test_data["cand_arcs"], test_data["cand_valid"], test_data["y_idx"], test_data["ref_arc"],
        trips=test_trips, trip_slices=te_slices, sp_evals=test_sp_evals,
    )
    test_eval_m4 = evaluate_predictions(
        probs_te_m4, test_data["cand_arcs"], test_data["cand_valid"], test_data["y_idx"], test_data["ref_arc"],
        trips=test_trips, trip_slices=te_slices, sp_evals=test_sp_evals,
    )

    # Gradient feature saliency on student_test
    saliency = compute_gradient_feature_saliency(
        final_model, Xs_te, test_data["cand_valid"], test_wh, test_conn,
        test_data["trip_starts"], test_data["trip_ends"], device,
    )

    # Save per-trip student_test metrics & raw predictions for maps/plots
    df_per_trip = pd.DataFrame(test_eval_m4["per_trip"])
    df_per_trip.to_csv(DIR_RESULTS / "model4_student_test_per_trip.csv", index=False)

    pred_k_m4 = np.argmax(np.where(test_data["cand_valid"], probs_te_m4, -1.0), axis=1)
    pred_arcs_m4 = test_data["cand_arcs"][np.arange(len(pred_k_m4)), pred_k_m4]
    pred_k_s1 = np.argmax(np.where(test_data["cand_valid"], probs_te_s1, -1.0), axis=1)
    pred_arcs_s1 = test_data["cand_arcs"][np.arange(len(pred_k_s1)), pred_k_s1]

    np.savez_compressed(
        DIR_RESULTS / "model4_student_test_predictions.npz",
        trip_ids=test_data["trip_ids"],
        trip_starts=test_data["trip_starts"],
        trip_ends=test_data["trip_ends"],
        probs_stage1_nngls=probs_te_s1,
        probs_model4_nngls_rlcd=probs_te_m4,
        pred_arcs_stage1=pred_arcs_s1,
        pred_arcs_model4=pred_arcs_m4,
        ref_arc=test_data["ref_arc"],
        y_idx=test_data["y_idx"],
        cand_arcs=test_data["cand_arcs"],
        cand_valid=test_data["cand_valid"],
    )

    with open(DIR_RESULTS / "model4_training_histories.json", "w") as f:
        json.dump(
            {
                "cv_folds": all_histories,
                "final_refit_stage1": full_hist_s1,
                "final_refit_stage2": full_hist_s2,
            },
            f,
            indent=2,
        )

    # Save concise Deliverable Section 3.1 row table (Stage 1 NN-GLS intermediate vs Model 4 NN-GLS+RLCD)
    table_rows = [
        {
            "Model": "Stage 1: NN-GLS (Pre-RLCD Checkpoint)",
            "Algorithm Family": "Geostatistical Graph NN (Vecchia NNGP GLS)",
            "Key Configuration": f"H={arch['hidden_dim']}, L={arch['num_layers']}, m={nngp_cfg.m_neighbors}, rho_s={nngp_cfg.rho_spatial_m}m, tau2={nngp_cfg.tau2}",
            "CV Macro F1-score": f"{df_folds['stage1_nngls_macro_f1'].mean():.4f} ± {df_folds['stage1_nngls_macro_f1'].std():.4f}",
            "Top-1 Accuracy": round(test_eval_s1["obs_top1_accuracy"], 5),
            "Route Continuity": round(test_eval_s1["route_continuity"], 5),
            "Student-Test Macro F1-score": round(test_eval_s1["macro_f1"], 5),
            "Log Loss": round(test_eval_s1["log_loss"], 5),
            "Brier Score": round(test_eval_s1["brier_score"], 5),
            "ECE": round(test_eval_s1["ece"], 5),
            "Seq Arc F1": round(test_eval_s1["seq_arc_f1"], 5),
            "Inference Time (ms/trip)": round(inf_ms_s1 / len(test_trips), 2),
        },
        {
            "Model": "Model 4: NN-GLS + RLCD (Final)",
            "Algorithm Family": "Geostatistical GNN (NN-GLS) + Calibrated RL (RLCD)",
            "Key Configuration": f"Stage1 NN-GLS + GRPO(G={rlcd_cfg.group_size}) Brier+Cont+SoftECE+KL({rlcd_cfg.kl_anchor_weight}), T={learned_temp}",
            "CV Macro F1-score": f"{df_folds['model4_macro_f1'].mean():.4f} ± {df_folds['model4_macro_f1'].std():.4f}",
            "Top-1 Accuracy": round(test_eval_m4["obs_top1_accuracy"], 5),
            "Route Continuity": round(test_eval_m4["route_continuity"], 5),
            "Student-Test Macro F1-score": round(test_eval_m4["macro_f1"], 5),
            "Log Loss": round(test_eval_m4["log_loss"], 5),
            "Brier Score": round(test_eval_m4["brier_score"], 5),
            "ECE": round(test_eval_m4["ece"], 5),
            "Seq Arc F1": round(test_eval_m4["seq_arc_f1"], 5),
            "Inference Time (ms/trip)": round(inf_ms_m4 / len(test_trips), 2),
        },
    ]
    pd.DataFrame(table_rows).to_csv(DIR_RESULTS / "model4_results_table.csv", index=False)

    summary = {
        "model": "Model 4: NN-GLS + RLCD",
        "device": str(device),
        "n_parameters": int(sum(p.numel() for p in final_model.parameters())),
        "learned_temperature": learned_temp,
        "timing": {
            "cv_5fold_total_sec": round(cv_train_sec_total, 2),
            "final_refit_sec": final_train_sec,
            "student_test_total_inf_ms": round(inf_ms_m4, 2),
            "student_test_ms_per_trip": round(inf_ms_m4 / len(test_trips), 2),
            "total_pipeline_sec": round(time.time() - t_global, 2),
        },
        "cv_summary_5fold": {
            "stage1_nngls_macro_f1_mean": round(float(df_folds["stage1_nngls_macro_f1"].mean()), 5),
            "stage1_nngls_macro_f1_std": round(float(df_folds["stage1_nngls_macro_f1"].std()), 5),
            "stage1_nngls_top1_mean": round(float(df_folds["stage1_nngls_top1_acc"].mean()), 5),
            "stage1_nngls_log_loss_mean": round(float(df_folds["stage1_nngls_log_loss"].mean()), 5),
            "stage1_nngls_brier_mean": round(float(df_folds["stage1_nngls_brier"].mean()), 5),
            "stage1_nngls_ece_mean": round(float(df_folds["stage1_nngls_ece"].mean()), 5),
            "stage1_nngls_continuity_mean": round(float(df_folds["stage1_nngls_continuity"].mean()), 5),
            "model4_macro_f1_mean": round(float(df_folds["model4_macro_f1"].mean()), 5),
            "model4_macro_f1_std": round(float(df_folds["model4_macro_f1"].std()), 5),
            "model4_top1_mean": round(float(df_folds["model4_top1_acc"].mean()), 5),
            "model4_top1_std": round(float(df_folds["model4_top1_acc"].std()), 5),
            "model4_hit_acc_mean": round(float(df_folds["model4_hit_acc"].mean()), 5),
            "model4_log_loss_mean": round(float(df_folds["model4_log_loss"].mean()), 5),
            "model4_brier_mean": round(float(df_folds["model4_brier"].mean()), 5),
            "model4_ece_mean": round(float(df_folds["model4_ece"].mean()), 5),
            "model4_continuity_mean": round(float(df_folds["model4_continuity"].mean()), 5),
            "model4_continuity_1hop_mean": round(float(df_folds["model4_continuity_1hop"].mean()), 5),
            "model4_seq_arc_f1_mean": round(float(df_folds["model4_seq_arc_f1"].mean()), 5),
        },
        "oof_all_80_train_stage1": oof_eval_s1,
        "oof_all_80_train_model4": oof_eval_s2,
        "student_test_stage1_nngls": test_eval_s1,
        "student_test_model4_nngls_rlcd": test_eval_m4,
        "feature_saliency": saliency,
    }
    with open(DIR_RESULTS / "model4_nngls_rlcd_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 82)
    print("FINAL MODEL 4 (NN-GLS + RLCD) RESULTS ON HELD-OUT STUDENT_TEST (10 TRIPS, 19,160 OBS)")
    print("=" * 82)
    for label, ev in [("Stage 1: NN-GLS (pre-RLCD)", test_eval_s1), ("Model 4: NN-GLS + RLCD    ", test_eval_m4)]:
        print(
            f"  {label} | Top-1 Acc={ev['obs_top1_accuracy']*100:.2f}% | "
            f"Macro F1={ev['macro_f1']:.4f} | Route Cont={ev['route_continuity']*100:.2f}% "
            f"(1-hop {ev['route_continuity_1hop']*100:.2f}%) | Seq Arc F1={ev['seq_arc_f1']:.4f} | "
            f"LogLoss={ev['log_loss']:.4f} | Brier={ev['brier_score']:.4f} | ECE={ev['ece']:.4f}"
        )
    print(f"  Candidate Miss Rate = {test_eval_m4['candidate_miss_rate']*100:.2f}% | "
          f"Inference = {inf_ms_m4/len(test_trips):.2f} ms/trip | Total elapsed = {time.time()-t_global:.1f}s")
    print("=" * 82)


if __name__ == "__main__":
    main()
