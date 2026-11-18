"""Build and cache the (observation x candidate) dataset for one partition.

Usage (from the repo root):
    python 02_feature_engineering/build_dataset.py --partition train
    python 02_feature_engineering/build_dataset.py --partition student_test
    python 02_feature_engineering/build_dataset.py --partition all
Options: --data "<path to Student folder>"  --k 16

Writes artifacts/cache/<partition>_k<K>.npz (git-ignored, reproducible) and results/dataset_build_<partition>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _d in ("01_data_processing", "02_feature_engineering", "06_model_evaluation"):
    sys.path.insert(0, str(ROOT / _d))

from data_io import data_root, list_trips, load_trip          # noqa: E402
from features import build_trip_table, FEATURE_NAMES           # noqa: E402
from labeling import align_route, CAP                          # noqa: E402
from candidates import K_DEFAULT                               # noqa: E402

CACHE_DIR = ROOT / "artifacts" / "cache"
RESULTS_DIR = ROOT / "results"


def cache_path(partition: str, k: int = K_DEFAULT) -> Path:
    return CACHE_DIR / f"{partition}_k{k}.npz"


def build_partition(partition: str, root=None, k: int = K_DEFAULT, verbose=True):
    root = Path(root) if root else data_root()
    trips = list_trips(partition, root)
    t0 = time.perf_counter()
    rows = {"X": [], "y": [], "obs": [], "cand_arc": [], "cand_rank": []}
    obs = {"trip": [], "idx": [], "t": [], "label_arc": [], "label_dist": [], "label_valid": [],
           "hit": [], "n_cand": []}
    obs_offset = 0
    per_trip = []
    for ti, tid in enumerate(trips):
        t1 = time.perf_counter()
        trip = load_trip(partition, tid, root)
        tab = build_trip_table(trip, K=k)
        lab = align_route(trip)
        n = trip.n_obs
        row_obs = tab["row_obs"]
        y = (tab["cand_arc"] == lab["arc"][row_obs]) & lab["valid"][row_obs]
        hit = np.zeros(n, dtype=bool)
        hit[row_obs[y]] = True
        rows["X"].append(tab["X"])
        rows["y"].append(y.astype(np.int8))
        rows["obs"].append(row_obs + obs_offset)
        rows["cand_arc"].append(tab["cand_arc"])
        rows["cand_rank"].append(tab["cand_rank"])
        obs["trip"].append(np.full(n, tid, dtype=np.int32))
        obs["idx"].append(np.arange(n, dtype=np.int32))
        obs["t"].append(trip.t)
        obs["label_arc"].append(lab["arc"].astype(np.int32))
        obs["label_dist"].append(lab["dist"])
        obs["label_valid"].append(lab["valid"])
        obs["hit"].append(hit)
        obs["n_cand"].append(tab["n_cand_per_obs"].astype(np.int8))
        obs_offset += n
        v = lab["valid"]
        per_trip.append({
            "trip": tid, "n_obs": int(n), "n_label_valid": int(v.sum()),
            "miss_rate": float(1.0 - hit[v].mean()) if v.any() else None,
            "n_nonmonotone": lab["n_nonmonotone"], "n_back_moves": lab["n_back_moves"],
            "seconds": round(time.perf_counter() - t1, 3),
        })
        if verbose:
            print(f"[{partition}] {ti + 1:3d}/{len(trips)} trip {tid:8d} obs={n:6d} "
                  f"valid={int(v.sum()):6d} miss={(per_trip[-1]['miss_rate'] if per_trip[-1]['miss_rate'] is not None else float('nan')):.3f} "
                  f"{per_trip[-1]['seconds']:.1f}s", flush=True)

    out = {f"row_{k_}": np.concatenate(v_) for k_, v_ in rows.items()}
    out.update({f"obs_{k_}": np.concatenate(v_) for k_, v_ in obs.items()})
    out["trips"] = np.array(trips, dtype=np.int32)
    out["feature_names"] = np.array(FEATURE_NAMES)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path(partition, k), **out)

    valid = out["obs_label_valid"]
    summary = {
        "partition": partition, "K": k, "label_cap_m": CAP,
        "n_trips": len(trips), "n_obs": int(len(valid)), "n_obs_label_valid": int(valid.sum()),
        "n_obs_label_invalid": int((~valid).sum()), "n_rows": int(len(out["row_y"])),
        "candidate_miss_rate_valid_obs": float(1.0 - out["obs_hit"][valid].mean()),
        "n_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
        "build_seconds": round(time.perf_counter() - t0, 2),
        "per_trip": per_trip,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"dataset_build_{partition}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"[{partition}] rows={summary['n_rows']} obs={summary['n_obs']} "
              f"invalid_labels={summary['n_obs_label_invalid']} "
              f"miss_rate={summary['candidate_miss_rate_valid_obs']:.4f} "
              f"time={summary['build_seconds']}s -> {cache_path(partition, k)}")
    return summary


def load_cache(partition: str, k: int = K_DEFAULT) -> dict:
    p = cache_path(partition, k)
    with np.load(p, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="all", choices=["train", "student_test", "all"])
    ap.add_argument("--data", default=None, help="path to 'Student folder'")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    a = ap.parse_args()
    parts = ["train", "student_test"] if a.partition == "all" else [a.partition]
    for p in parts:
        build_partition(p, a.data, a.k)


if __name__ == "__main__":
    main()
