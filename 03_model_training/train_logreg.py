"""Model 1 - Logistic Regression (linear / probabilistic family) for PathSnap candidate scoring.

Task: for each GPS observation we have K=16 candidate road arcs; the model outputs P(candidate is the reference arc).
Observation prediction = candidate with the highest probability (Top-1).

Protocol
  * 5-fold GroupKFold by TRIP on the 80 training trips (no trip is ever split across train/validation)
  * StandardScaler is inside the sklearn Pipeline, so it is fitted on the training folds only
  * Training rows = candidates of observations whose reference arc is among the candidates (missed observations
    have no positive candidate and would only teach the model that every candidate is negative).
    EVALUATION uses all label-valid observations, so candidate misses count as errors.
  * Final model: refit on all 80 training trips, evaluated on the 10 student_test trips (development comparison only).
  * evaluator_test is never touched.

Usage (repo root, venv active):
    python 03_model_training/train_logreg.py
    python 03_model_training/train_logreg.py --data "C:\\Student folder" --c 1.0
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _d in ("01_data_processing", "02_feature_engineering", "06_model_evaluation"):
    sys.path.insert(0, str(ROOT / _d))

from data_io import data_root, load_arcs_only, load_route_only      # noqa: E402
from build_dataset import build_partition, load_cache, cache_path    # noqa: E402
from candidates import K_DEFAULT                                     # noqa: E402
from metrics import evaluate_all, TripGraph, HOPS, MAX_DT_S          # noqa: E402

SEED = 42
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
PLOTS = ROOT / "07_visualisations" / "plots"
MODEL_NAME = "logreg"


def make_pipeline(c: float, max_iter: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter, random_state=SEED)),
    ])


def fit_predict(ds_fit_mask_rows, ds, c, max_iter):
    """Fit on given training rows; return fitted pipeline and fit seconds."""
    X, y = ds["row_X"][ds_fit_mask_rows], ds["row_y"][ds_fit_mask_rows]
    pipe = make_pipeline(c, max_iter)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(X, y)
    secs = time.perf_counter() - t0
    converged = not any("converge" in str(x.message).lower() for x in w)
    return pipe, secs, converged


def proba(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_plots(oof_cm, test_cm, coefs, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cm, title in ((axes[0], oof_cm, "5-fold out-of-fold (train trips)"),
                          (axes[1], test_cm, "student_test")):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(title, fontsize=10)
    fig.suptitle("Logistic Regression - candidate-level confusion matrix (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(PLOTS / "logreg_confusion_matrix.png", dpi=150)
    plt.close(fig)

    order = np.argsort(np.abs(coefs))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], np.array(coefs)[order], color="#3b6ea5")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("standardised coefficient (log-odds per 1 SD)")
    ax.set_title("Logistic Regression coefficients (final model)")
    fig.tight_layout()
    fig.savefig(PLOTS / "logreg_coefficients.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to 'Student folder'")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--c", type=float, default=1.0, help="inverse L2 regularisation strength")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature caches")
    a = ap.parse_args()

    root = Path(a.data) if a.data else data_root()
    for part in ("train", "student_test"):
        if a.rebuild or not cache_path(part, a.k).exists():
            print(f"Building feature cache for {part} ...", flush=True)
            build_partition(part, root, a.k)
    tr = load_cache("train", a.k)
    te = load_cache("student_test", a.k)
    names = [str(s) for s in tr["feature_names"]]

    graph_cache, route_cache = {}, {}

    def graphs_for(part):
        def f(tid):
            key = (part, tid)
            if key not in graph_cache:
                graph_cache[key] = TripGraph(*load_arcs_only(part, tid, root))
            return graph_cache[key]
        return f

    def routes_for(part):
        def f(tid):
            key = (part, tid)
            if key not in route_cache:
                route_cache[key] = load_route_only(part, tid, root)
            return route_cache[key]
        return f

    g_tr, r_tr = graphs_for("train"), routes_for("train")
    g_te, r_te = graphs_for("student_test"), routes_for("student_test")

    # rows usable for fitting: observation has a reference arc AND it is among the candidates
    def fit_rows(ds, obs_mask=None):
        ok = ds["obs_label_valid"] & ds["obs_hit"]
        if obs_mask is not None:
            ok = ok & obs_mask
        return ok[ds["row_obs"]]

    # ---------------------------------------------------------------- 5-fold group CV on train trips
    n_obs_tr = len(tr["obs_trip"])
    fold_of_obs = np.full(n_obs_tr, -1, dtype=np.int64)
    gkf = GroupKFold(n_splits=a.folds)
    for f, (_, va) in enumerate(gkf.split(np.zeros(n_obs_tr), groups=tr["obs_trip"])):
        fold_of_obs[va] = f
    oof = np.full(len(tr["row_y"]), np.nan)
    folds = []
    for f in range(a.folds):
        va_obs = fold_of_obs == f
        tr_obs = ~va_obs
        pipe, fit_s, conv = fit_predict(fit_rows(tr, tr_obs), tr, a.c, a.max_iter)
        rows_va = va_obs[tr["row_obs"]]
        t0 = time.perf_counter()
        oof[rows_va] = proba(pipe, tr["row_X"][rows_va])
        inf_s = time.perf_counter() - t0
        summ, _, _ = evaluate_all(tr, oof, g_tr, r_tr, obs_sel=va_obs)
        trips_f = sorted(int(t) for t in np.unique(tr["obs_trip"][va_obs]))
        folds.append({"fold": f, "trips": trips_f, "fit_seconds": fit_s, "inference_seconds": inf_s,
                      "converged": conv, "n_iter": int(pipe.named_steps["clf"].n_iter_[0]), "metrics": summ})
        print(f"[CV fold {f}] trips={len(trips_f)} top1={summ['observation_top1']:.4f} "
              f"logloss={summ['candidate_level']['log_loss']:.4f} fit={fit_s:.1f}s converged={conv}", flush=True)
    assert not np.isnan(oof).any(), "every training row must receive an out-of-fold prediction"
    oof_summary, oof_per_trip, _ = evaluate_all(tr, oof, g_tr, r_tr)

    # ---------------------------------------------------------------- final fit + evaluation
    pipe, fit_s, conv = fit_predict(fit_rows(tr), tr, a.c, a.max_iter)
    p_train = proba(pipe, tr["row_X"])
    train_summary, _, _ = evaluate_all(tr, p_train, g_tr, r_tr)
    t0 = time.perf_counter()
    p_test = proba(pipe, te["row_X"])
    inf_s = time.perf_counter() - t0
    test_summary, test_per_trip, _ = evaluate_all(te, p_test, g_te, r_te)

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]
    coef_table = sorted(zip(names, coefs.tolist()), key=lambda kv: -abs(kv[1]))

    # ---------------------------------------------------------------- save
    ARTIFACTS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    (ARTIFACTS / "cache").mkdir(exist_ok=True)
    joblib.dump(pipe, ARTIFACTS / "logreg_pipeline.joblib")
    schema = {
        "feature_names": names, "K": a.k, "projection": "per-trip AEQD (pyproj), metres",
        "candidate_sample_step_m": 20.0, "gap_seconds": 5.0, "probability_column": "P(candidate is reference arc)",
        "model": "StandardScaler + LogisticRegression(L2, lbfgs)", "C": a.c, "seed": SEED,
    }
    with open(ARTIFACTS / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    np.savez(ARTIFACTS / "cache" / "logreg_scores.npz", oof_train=oof, student_test=p_test,
             train_row_obs=tr["row_obs"], test_row_obs=te["row_obs"])

    result = {
        "model": MODEL_NAME, "family": "linear / probabilistic",
        "config": {"C": a.c, "penalty": "l2", "solver": "lbfgs", "max_iter": a.max_iter, "class_weight": None,
                   "scaler": "StandardScaler (fit on training folds only)", "K_candidates": a.k,
                   "cv": f"GroupKFold({a.folds}) by trip", "seed": SEED, "threshold": 0.5,
                   "route_continuity": {"hops": HOPS, "max_dt_s": MAX_DT_S}},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__},
        "primary_selection_metric_proposal": "observation_top1 (candidate misses counted as wrong)",
        "train_rows_used_for_fit": int(fit_rows(tr).sum()),
        "final_fit": {"seconds": fit_s, "converged": conv, "n_iter": int(clf.n_iter_[0])},
        "inference": {"student_test_rows": int(len(p_test)), "seconds": inf_s,
                      "microseconds_per_observation": inf_s / len(te["obs_trip"]) * 1e6},
        "cv_folds": folds,
        "cv_stability": {
            "top1_mean": float(np.mean([x["metrics"]["observation_top1"] for x in folds])),
            "top1_std": float(np.std([x["metrics"]["observation_top1"] for x in folds])),
            "log_loss_mean": float(np.mean([x["metrics"]["candidate_level"]["log_loss"] for x in folds])),
            "log_loss_std": float(np.std([x["metrics"]["candidate_level"]["log_loss"] for x in folds])),
        },
        "train_in_sample": train_summary, "cv_out_of_fold_pooled": oof_summary, "student_test": test_summary,
        "coefficients_standardised": coef_table, "intercept": float(clf.intercept_[0]),
    }
    with open(RESULTS / "logreg_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)

    import csv
    for tag, rows in (("cv_oof_train", oof_per_trip), ("student_test", test_per_trip)):
        with open(RESULTS / f"logreg_per_trip_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(jsonable(rows))
    with open(RESULTS / "logreg_coefficients.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["feature", "standardised_coefficient"]); w.writerows(coef_table)
    save_plots(oof_summary["candidate_level"]["confusion_matrix"],
               test_summary["candidate_level"]["confusion_matrix"], coefs, names)

    # ---------------------------------------------------------------- console summary (ASCII only)
    def line(tag, s):
        c = s["candidate_level"]
        return (f"{tag:<18} top1={s['observation_top1']:.4f} nearest={s['nearest_arc_top1']:.4f} "
                f"miss={s['candidate_miss_rate']:.4f} P={c['precision']:.4f} R={c['recall']:.4f} "
                f"F1={c['f1']:.4f} logloss={c['log_loss']:.4f} cont={s['route_continuity']:.4f} "
                f"seqF1={s['sequence_f1_macro']:.4f}")
    print("\n=== Logistic Regression summary ===")
    print(line("train (in-sample)", train_summary))
    print(line("CV out-of-fold", oof_summary))
    print(line("student_test", test_summary))
    print(f"CV top1 mean+-std: {result['cv_stability']['top1_mean']:.4f} +- {result['cv_stability']['top1_std']:.4f}")
    print(f"reference-label continuity upper bound (student_test): {test_summary['route_continuity_reference_labels']:.4f}")
    print(f"final fit {fit_s:.1f}s, inference {inf_s:.3f}s for {len(te['obs_trip'])} observations")
    print("Saved: artifacts/logreg_pipeline.joblib, artifacts/feature_schema.json, results/logreg_*.{json,csv}, "
          "07_visualisations/plots/logreg_*.png")


if __name__ == "__main__":
    main()
