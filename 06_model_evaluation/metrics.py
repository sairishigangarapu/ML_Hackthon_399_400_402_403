"""Evaluation metrics for PathSnap (candidate-level, observation-level, route continuity, sequence-level).

Dataset dict `ds` comes from build_dataset.load_cache(). Row arrays (row_*) are candidate rows,
obs arrays (obs_*) are GPS observations. `p` is a score/probability per candidate row.

Conventions
  * Only observations with obs_label_valid are scored for accuracy-type metrics (reference alignment exists).
  * Observation-level Top-1: highest-score candidate == reference arc. A candidate MISS (reference arc not among
    the K candidates) always counts as wrong.
  * Candidate miss rate = fraction of valid observations whose reference arc is not in the candidate set.
  * Candidate-level metrics (acc/precision/recall/F1/log loss/confusion matrix, threshold 0.5) are computed on the
    candidate rows of valid observations. Rows of missed observations are all negatives.
  * Route continuity = share of consecutive observation pairs (time step <= MAX_DT_S) whose predicted arcs are the
    same arc or joined by a directed path of <= HOPS arc-to-arc links. (Own definition; the reference labels get the
    same score as an upper bound.)
  * Sequence-level = set precision/recall/F1/Jaccard between the set of predicted arcs of a trip and the arcs in
    the reference .route file.
"""
from __future__ import annotations

import numpy as np

HOPS = 4
MAX_DT_S = 5.0
EPS = 1e-15


# ----------------------------------------------------------------------------- candidate level
def candidate_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
    pc = np.clip(p, EPS, 1 - EPS)
    ll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    return {
        "n_rows": n, "positives": tp + fn, "threshold": thr,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": prec, "recall": rec, "f1": f1, "macro_f1": (f1 + f10) / 2,
        "log_loss": ll, "brier": float(((p - y) ** 2).mean()),
        "confusion_matrix": [[tn, fp], [fn, tp]],      # rows = true (0,1), cols = predicted (0,1)
    }


# ----------------------------------------------------------------------------- observation level
def top1_predictions(row_obs, cand_arc, p, n_obs):
    """Predicted arc per observation (-1 if it has no candidates). Ties -> earlier (nearer) candidate row."""
    pred = np.full(n_obs, -1, dtype=np.int64)
    if len(p) == 0:
        return pred
    order = np.lexsort((-p, row_obs))
    first = np.unique(row_obs[order], return_index=True)
    pred[first[0]] = cand_arc[order[first[1]]]
    return pred


def nearest_arc_predictions(row_obs, cand_arc, cand_rank, n_obs):
    return top1_predictions(row_obs, cand_arc, -cand_rank.astype(np.float64), n_obs)


# ----------------------------------------------------------------------------- continuity
class TripGraph:
    def __init__(self, arc_from, arc_to):
        self.f = np.asarray(arc_from)
        self.t = np.asarray(arc_to)
        order = np.argsort(self.f, kind="stable")
        self._sorted_arcs = order
        n_nodes = int(max(self.f.max(), self.t.max())) + 1
        self._start = np.searchsorted(self.f[order], np.arange(n_nodes + 1))
        self._cache = {}

    def _heads_from(self, node):
        s, e = self._start[node], self._start[node + 1]
        return self.t[self._sorted_arcs[s:e]]

    def connected(self, a, b, hops=HOPS):
        """True if arc b follows arc a (same arc, or a directed path with <= hops arc-to-arc links)."""
        if a == b:
            return True
        if a < 0 or b < 0:
            return False
        key = (a, b)
        r = self._cache.get(key)
        if r is not None:
            return r
        target = int(self.f[b])
        frontier = {int(self.t[a])}
        seen = set(frontier)
        ok = target in frontier
        h = 1
        while not ok and h < hops and frontier:
            nxt = set()
            for node in frontier:
                for v in self._heads_from(node):
                    v = int(v)
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            ok = target in frontier
            h += 1
        self._cache[key] = ok
        return ok


def continuity(pred_arc, obs_t, graph, valid=None, hops=HOPS, max_dt=MAX_DT_S):
    """(n_continuous, n_evaluated) over consecutive pairs of ONE trip. `valid` optionally restricts pairs."""
    n = len(pred_arc)
    if n < 2:
        return 0, 0
    dt = np.diff(obs_t)
    use = dt <= max_dt
    if valid is not None:
        use &= valid[:-1] & valid[1:]
    idx = np.flatnonzero(use)
    good = 0
    for i in idx:
        a, b = int(pred_arc[i]), int(pred_arc[i + 1])
        if a == b or graph.connected(a, b, hops):
            good += 1
    return good, len(idx)


# ----------------------------------------------------------------------------- sequence level
def set_prf(pred_set, ref_set):
    inter = len(pred_set & ref_set)
    p = inter / len(pred_set) if pred_set else 0.0
    r = inter / len(ref_set) if ref_set else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    j = inter / len(pred_set | ref_set) if (pred_set | ref_set) else 0.0
    return p, r, f, j, inter


# ----------------------------------------------------------------------------- everything together
def evaluate_all(ds, p, graphs, routes, obs_sel=None, thr=0.5):
    """Full metric set on the observations in `obs_sel` (bool mask over ds obs arrays; default all).

    graphs : callable trip_id -> TripGraph
    routes : callable trip_id -> reference route array
    Returns (summary dict, per-trip list of dicts, predicted arc per observation).
    """
    n_obs = len(ds["obs_trip"])
    sel = np.ones(n_obs, bool) if obs_sel is None else obs_sel
    valid = ds["obs_label_valid"]
    row_obs = ds["row_obs"]
    row_sel = sel[row_obs]

    pred = np.full(n_obs, -1, dtype=np.int64)
    pred_near = np.full(n_obs, -1, dtype=np.int64)
    rs = np.flatnonzero(row_sel)
    pred_all = top1_predictions(row_obs[rs], ds["row_cand_arc"][rs], p[rs], n_obs)
    near_all = nearest_arc_predictions(row_obs[rs], ds["row_cand_arc"][rs], ds["row_cand_rank"][rs], n_obs)
    pred[sel] = pred_all[sel]
    pred_near[sel] = near_all[sel]

    ov = sel & valid
    label = ds["obs_label_arc"]
    hit = ds["obs_hit"]
    n_v = int(ov.sum())
    top1 = float((pred[ov] == label[ov]).mean()) if n_v else float("nan")
    near1 = float((pred_near[ov] == label[ov]).mean()) if n_v else float("nan")
    miss = float(1.0 - hit[ov].mean()) if n_v else float("nan")
    top1_hit = float((pred[ov & hit] == label[ov & hit]).mean()) if (ov & hit).any() else float("nan")

    rmask = row_sel & valid[row_obs]
    cm = candidate_metrics(ds["row_y"][rmask], p[rmask], thr)

    per_trip, cont_g, cont_n, ref_g, ref_n = [], 0, 0, 0, 0
    micro_i = micro_p = micro_r = 0
    prfs = []
    for tid in np.unique(ds["obs_trip"][sel]):
        m = sel & (ds["obs_trip"] == tid)
        g = graphs(int(tid))
        t = ds["obs_t"][m]
        cg, cn = continuity(pred[m], t, g)
        rg, rn = continuity(label[m], t, g, valid=valid[m])
        ref_route = set(int(a) for a in routes(int(tid)))
        pset = set(int(a) for a in pred[m] if a >= 0)
        pr, rc, f1, jac, inter = set_prf(pset, ref_route)
        mv = m & valid
        per_trip.append({
            "trip": int(tid), "n_obs": int(m.sum()), "n_valid": int(mv.sum()),
            "miss_rate": float(1 - hit[mv].mean()) if mv.any() else None,
            "top1": float((pred[mv] == label[mv]).mean()) if mv.any() else None,
            "nearest_arc_top1": float((pred_near[mv] == label[mv]).mean()) if mv.any() else None,
            "continuity": cg / cn if cn else None,
            "seq_precision": pr, "seq_recall": rc, "seq_f1": f1, "seq_jaccard": jac,
        })
        cont_g += cg; cont_n += cn; ref_g += rg; ref_n += rn
        micro_i += inter; micro_p += len(pset); micro_r += len(ref_route)
        prfs.append((pr, rc, f1, jac))
    prfs = np.array(prfs) if prfs else np.zeros((0, 4))
    mp = micro_i / micro_p if micro_p else 0.0
    mr = micro_i / micro_r if micro_r else 0.0
    summary = {
        "n_trips": len(per_trip), "n_obs": int(sel.sum()), "n_obs_scored": n_v,
        "observation_top1": top1, "observation_top1_given_candidate_hit": top1_hit,
        "nearest_arc_top1": near1, "candidate_miss_rate": miss,
        "route_continuity": cont_g / cont_n if cont_n else float("nan"),
        "route_continuity_reference_labels": ref_g / ref_n if ref_n else float("nan"),
        "continuity_pairs": cont_n,
        "sequence_precision_macro": float(prfs[:, 0].mean()) if len(prfs) else float("nan"),
        "sequence_recall_macro": float(prfs[:, 1].mean()) if len(prfs) else float("nan"),
        "sequence_f1_macro": float(prfs[:, 2].mean()) if len(prfs) else float("nan"),
        "sequence_jaccard_macro": float(prfs[:, 3].mean()) if len(prfs) else float("nan"),
        "sequence_precision_micro": mp, "sequence_recall_micro": mr,
        "sequence_f1_micro": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "candidate_level": cm,
    }
    return summary, per_trip, pred
