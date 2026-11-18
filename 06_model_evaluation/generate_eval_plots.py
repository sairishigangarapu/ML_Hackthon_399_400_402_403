"""Stage 06B: Generate genuine diagnostic visualisations for Model 4 (NN-GLS + RLCD).

Reads ONLY verified experimental outputs from results/ — zero fabricated numbers or curves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_data_processing"))
from config import DIR_PLOTS, DIR_RESULTS


def plot_training_curves(histories: dict) -> None:
    """Plot genuine epoch-wise Stage 1 (NN-GLS) -> Stage 2 (RLCD) train & validation trajectories."""
    fold0 = histories["cv_folds"]["fold_0"]
    df = pd.DataFrame(fold0)
    s1_end = int(df[df["stage"] == "nngls"]["global_epoch"].max())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # 1. Stage-specific train & val loss
    ax = axes[0, 0]
    ax.plot(df["global_epoch"], df["train_loss"], marker="o", ms=3.5, lw=2.0, color="#2563eb", label="Train Loss")
    ax.plot(df["global_epoch"], df["val_loss"], marker="s", ms=3.5, lw=2.0, color="#dc2626", label="Validation Loss")
    ax.axvline(s1_end + 0.5, color="#475569", linestyle="--", lw=1.4, label="Stage 1→2 (RLCD start)")
    ax.set_title("Epoch-wise Loss (Fold 0: Stage 1 NN-GLS → Stage 2 RLCD)", fontweight="bold")
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Objective Loss")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.28)

    # 2. Validation Hit Top-1 Accuracy across all 5 folds
    ax = axes[0, 1]
    for fk, hist in histories["cv_folds"].items():
        hdf = pd.DataFrame(hist)
        ax.plot(hdf["global_epoch"], hdf["val_hit_acc"] * 100.0, lw=1.7, alpha=0.82, label=fk.replace("_", " ").title())
    ax.axvline(s1_end + 0.5, color="#475569", linestyle="--", lw=1.2)
    ax.set_title("Validation Hit Top-1 Accuracy Across 5 GroupKFold Folds", fontweight="bold")
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Validation Hit Top-1 Accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.28)

    # 3. Validation NLL & Multi-Class Brier Score (Fold 0)
    ax = axes[1, 0]
    ax.plot(df["global_epoch"], df["val_nll"], marker="o", ms=3.5, lw=2.0, color="#7c3aed", label="Val Log-Loss (NLL)")
    ax.plot(df["global_epoch"], df["val_brier"], marker="^", ms=3.5, lw=2.0, color="#0d9488", label="Val Multi-Class Brier")
    ax.axvline(s1_end + 0.5, color="#475569", linestyle="--", lw=1.4, label="RLCD Post-Training Start")
    ax.set_title("Strictly Proper Scoring Rules on Validation (Fold 0)", fontweight="bold")
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Score (Lower is Better)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.28)

    # 4. Validation Local Road-Network Continuity (Fold 0)
    ax = axes[1, 1]
    ax.plot(df["global_epoch"], df["val_local_cont"] * 100.0, marker="D", ms=3.5, lw=2.0, color="#ea580c",
            label="Val Local Graph Continuity (%)")
    ax.axvline(s1_end + 0.5, color="#475569", linestyle="--", lw=1.4, label="RLCD Continuity Reward Start")
    ax.set_title("Validation Directed Road-Network Continuity (Fold 0)", fontweight="bold")
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Connected Step Transitions (%)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.28)

    plt.suptitle("Model 4 (NN-GLS + RLCD): Genuine Epoch-Wise Training & Validation Histories",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "model4_training_curves.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_calibration_and_stage_lift(summary: dict) -> None:
    """Plot Reliability Diagram & Calibration comparison: Stage 1 (NN-GLS) vs Model 4 (NN-GLS + RLCD)."""
    s1 = summary["student_test_stage1_nngls"]
    m4 = summary["student_test_model4_nngls_rlcd"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # 1. Reliability Diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.3, label="Perfect Calibration")
    for label, ev, col, mk in [
        (f"Stage 1: NN-GLS (ECE={s1['ece']:.4f})", s1, "#64748b", "s"),
        (f"Model 4: NN-GLS + RLCD (ECE={m4['ece']:.4f})", m4, "#2563eb", "o"),
    ]:
        rb = ev["reliability_bins"]
        xs = [c for c, a in zip(rb["avg_conf"], rb["avg_acc"]) if c is not None and a is not None]
        ys = [a for c, a in zip(rb["avg_conf"], rb["avg_acc"]) if c is not None and a is not None]
        ax.plot(xs, ys, marker=mk, lw=2.2, ms=6, color=col, label=label)
    ax.set_title("Reliability Diagram (Student-Test)", fontweight="bold")
    ax.set_xlabel("Mean Predicted Top-1 Confidence")
    ax.set_ylabel("Empirical Top-1 Accuracy")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.28)

    # 2. Proper Scoring & Calibration Metrics (Lower is Better)
    ax = axes[1]
    metrics_low = ["Log-Loss", "Brier Score", "ECE (×10)"]
    s1_vals = [s1["log_loss"], s1["brier_score"], s1["ece"] * 10.0]
    m4_vals = [m4["log_loss"], m4["brier_score"], m4["ece"] * 10.0]
    x = np.arange(len(metrics_low))
    w = 0.34
    b1 = ax.bar(x - w / 2, s1_vals, w, color="#94a3b8", label="Stage 1: NN-GLS")
    b2 = ax.bar(x + w / 2, m4_vals, w, color="#2563eb", label="Model 4: NN-GLS + RLCD")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_low)
    ax.set_title("Calibration & Proper Scoring Comparison (Lower = Better)", fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.28)

    # 3. Accuracy, F1, Continuity & Sequence Arc F1 (Higher is Better)
    ax = axes[2]
    metrics_hi = ["Top-1 Acc", "Macro F1", "Route Cont.", "Seq Arc F1"]
    s1_hi = [s1["obs_top1_accuracy"] * 100, s1["macro_f1"] * 100, s1["route_continuity"] * 100, s1["seq_arc_f1"] * 100]
    m4_hi = [m4["obs_top1_accuracy"] * 100, m4["macro_f1"] * 100, m4["route_continuity"] * 100, m4["seq_arc_f1"] * 100]
    x2 = np.arange(len(metrics_hi))
    b1 = ax.bar(x2 - w / 2, s1_hi, w, color="#94a3b8", label="Stage 1: NN-GLS")
    b2 = ax.bar(x2 + w / 2, m4_hi, w, color="#0d9488", label="Model 4: NN-GLS + RLCD")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics_hi)
    ax.set_ylim(max(0.0, min(s1_hi + m4_hi) - 10.0), 103.5)
    ax.set_title("Route & Candidate Performance (Higher = Better)", fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.28)

    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "model4_calibration_and_stage_lift.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(summary: dict) -> None:
    """Plot Candidate-Level Confusion Matrix (raw counts + row-normalized %) on Student-Test."""
    cm = np.array(summary["student_test_model4_nngls_rlcd"]["confusion_matrix"], dtype=np.int64)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    labels = np.array([
        [f"{cm[i, j]:,}\n({cm_norm[i, j]:.2f}%)" for j in range(2)]
        for i in range(2)
    ])

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Row-Normalized Percentage (%)"},
        xticklabels=["Pred: Non-Match (0)", "Pred: Match (1)"],
        yticklabels=["True: Non-Match (0)", "True: Match (1)"],
        linewidths=1.0,
        ax=ax,
    )
    m4 = summary["student_test_model4_nngls_rlcd"]
    ax.set_title(
        f"Model 4 (NN-GLS + RLCD) Candidate Confusion Matrix\n"
        f"Student-Test (10 trips, {m4['n_observations']:,} GPS obs | Macro F1={m4['macro_f1']:.4f})",
        fontweight="bold",
        fontsize=10.5,
    )
    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "model4_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_per_trip_and_cv_stability(df_folds: pd.DataFrame, df_trips: pd.DataFrame) -> None:
    """Plot 5-fold GroupKFold CV stability and per-trip Student-Test breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

    # Left: 5-fold CV stability
    ax = axes[0]
    x = np.arange(len(df_folds))
    w = 0.26
    ax.bar(x - w, df_folds["model4_macro_f1"] * 100, w, color="#2563eb", label="CV Macro F1")
    ax.bar(x, df_folds["model4_top1_acc"] * 100, w, color="#0d9488", label="CV Top-1 Acc (All Obs)")
    ax.bar(x + w, df_folds["model4_continuity"] * 100, w, color="#ea580c", label="CV Route Continuity")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in df_folds["fold"]])
    ax.set_ylim(50, 104)
    ax.set_title("5-Fold GroupKFold CV Stability (80 Train Trips)", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.28)

    # Right: Per-trip Student-Test breakdown
    ax = axes[1]
    x2 = np.arange(len(df_trips))
    ax.bar(x2 - w, df_trips["obs_top1_accuracy"] * 100, w, color="#2563eb", label="Obs Top-1 Acc")
    ax.bar(x2, df_trips["route_continuity"] * 100, w, color="#0d9488", label="Route Continuity")
    ax.bar(x2 + w, df_trips["seq_arc_f1"] * 100, w, color="#7c3aed", label="Seq Arc F1")
    ax.set_xticks(x2)
    ax.set_xticklabels(df_trips["trip_id"].tolist(), rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(50, 104)
    ax.set_title("Per-Trip Performance Across All 10 Held-Out Student-Test Trips", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.28)

    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "model4_cv_and_per_trip.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_feature_saliency(summary: dict) -> None:
    """Plot Top-16 engineered features ranked by Model 4 gradient saliency."""
    sal = summary["feature_saliency"][:16]
    df_s = pd.DataFrame(sal).iloc[::-1]

    group_colors = {}
    for f in df_s["feature"]:
        if any(k in f for k in ["dist_m", "proj_", "lateral", "arc_len"]):
            group_colors[f] = "#2563eb"
        elif any(k in f for k in ["heading", "speed", "stationary"]):
            group_colors[f] = "#0d9488"
        elif any(k in f for k in ["rank", "diff_best", "ratio_best", "reverse", "density"]):
            group_colors[f] = "#7c3aed"
        elif any(k in f for k in ["node_", "step_"]):
            group_colors[f] = "#ea580c"
        else:
            group_colors[f] = "#64748b"

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = [group_colors[f] for f in df_s["feature"]]
    ax.barh(df_s["feature"], df_s["relative_importance"] * 100.0, color=colors, alpha=0.88)
    for i, val in enumerate(df_s["relative_importance"] * 100.0):
        ax.text(val + 0.15, i, f"{val:.2f}%", va="center", fontsize=8.5)
    ax.set_xlabel("Relative Gradient Saliency (%)")
    ax.set_title("Model 4 (NN-GLS + RLCD): Top-16 Feature Importances on Student-Test", fontweight="bold")
    ax.grid(axis="x", alpha=0.28)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2563eb", label="Geometric / Proximity"),
        Patch(facecolor="#0d9488", label="Kinematic / Heading"),
        Patch(facecolor="#7c3aed", label="Listwise Competition"),
        Patch(facecolor="#ea580c", label="Road Topology / Context"),
        Patch(facecolor="#64748b", label="Temporal / GPS Gaps"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "model4_feature_saliency.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    with open(DIR_RESULTS / "model4_nngls_rlcd_summary.json") as f:
        summary = json.load(f)
    with open(DIR_RESULTS / "model4_training_histories.json") as f:
        histories = json.load(f)
    df_folds = pd.read_csv(DIR_RESULTS / "model4_cv_folds.csv")
    df_trips = pd.read_csv(DIR_RESULTS / "model4_student_test_per_trip.csv")

    plot_training_curves(histories)
    plot_calibration_and_stage_lift(summary)
    plot_confusion_matrix(summary)
    plot_per_trip_and_cv_stability(df_folds, df_trips)
    plot_feature_saliency(summary)
    print(f"[Stage 06B Evaluation Plots] Saved 5 publication-grade figures to {DIR_PLOTS}")


if __name__ == "__main__":
    main()
