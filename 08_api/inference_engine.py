"""Laptop 1 Inference Engine for Model 4 (Pure NN-GLS + RLCD — Zero HMM).

STRICT EVALUATION INTEGRITY:
  - `predict_trip()` loads ONLY .track, .nodes, .arcs (`load_route=False`).
    Ground-truth .route files NEVER enter the prediction pipeline.
  - `evaluate_server_side()` independently reads .route strictly on Laptop 1
    to compute quantitative metrics and NEVER returns ground-truth arc IDs or
    ground-truth coordinates to the client UI.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_data_processing"))
from config import DIR_ARTIFACTS, EVAL_DIR
from io_utils import TripData, build_adjacency, list_trips, load_route_only, load_trip

sys.path.insert(0, str(ROOT / "02_feature_engineering"))
from candidate_generator import CandidateSet
from feature_extractor import extract_trip_features
from route_aligner import align_track_to_route

sys.path.insert(0, str(ROOT / "03_model_training"))
from nngls_core import (
    NNGLSSpatioTemporalNet,
    NNGPConfig,
    apply_nngp_kriging_inference,
    build_nngp_whitening_for_trip,
)
from rlcd_core import precompute_step_connectivity_trip
from train_utils import apply_feature_scaler, predict_trip_logits_sliding

sys.path.insert(0, str(ROOT / "06_model_evaluation"))
from metrics import (
    ShortestPathEvaluator,
    collapse_predicted_route_arcs,
    evaluate_predictions,
)


class Model4InferenceEngine:
    """Production runtime for Model 4 (NN-GLS + RLCD) on Laptop 1."""

    def __init__(
        self,
        checkpoint_path: Path = DIR_ARTIFACTS / "model4_nngls_rlcd.pt",
        scaler_path: Path = DIR_ARTIFACTS / "scaler_model4_nngls_rlcd.json",
        eval_dir: Path = EVAL_DIR,
    ) -> None:
        self.eval_dir = Path(eval_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing Model 4 checkpoint: {checkpoint_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing Model 4 scaler: {scaler_path}")

        with open(scaler_path) as f:
            sc = json.load(f)
        self.scaler_mean = np.asarray(sc["mean"], dtype=np.float32)
        self.scaler_std = np.asarray(sc["std"], dtype=np.float32)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.in_features = int(ckpt["in_features"])
        self.learned_temperature = float(ckpt.get("learned_temperature", 1.0))

        arch = self.config["architecture"]
        self.nngp_cfg = NNGPConfig(**self.config["nngp"])
        self.model = NNGLSSpatioTemporalNet(
            in_features=self.in_features,
            hidden_dim=arch["hidden_dim"],
            num_layers=arch["num_layers"],
            num_heads=arch["num_heads"],
            dropout=arch["dropout"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.n_parameters = int(sum(p.numel() for p in self.model.parameters()))

    def list_evaluation_trips(self) -> List[Dict]:
        """List available evaluation trips with non-label metadata ONLY."""
        out = []
        for tid in list_trips(self.eval_dir):
            trip = load_trip(self.eval_dir, tid, partition="eval", load_route=False)
            out.append({
                "trip_id": tid,
                "n_gps": int(len(trip.track_ts)),
                "n_nodes": int(len(trip.node_x)),
                "n_arcs": int(len(trip.arcs)),
                "duration_s": round(float(trip.track_ts[-1] - trip.track_ts[0]), 1) if len(trip.track_ts) > 1 else 0.0,
                "center_lat": round(float(np.mean(trip.track_lat)), 5),
                "center_lon": round(float(np.mean(trip.track_lon)), 5),
            })
        return out

    def predict_and_evaluate(self, trip_id: str) -> Dict:
        """Run strict isolated Model 4 prediction, then separate server-side metric computation."""
        t_total = time.perf_counter()

        # ------------------------------------------------------------------
        # PHASE 1: STRICT PREDICTION PIPELINE (load_route=False — ZERO GT!)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        trip = load_trip(self.eval_dir, trip_id, partition="eval", load_route=False)
        feats, cands = extract_trip_features(trip)
        X_scaled = apply_feature_scaler(feats, cands.cand_valid, self.scaler_mean, self.scaler_std)
        feat_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        coords = np.column_stack([trip.track_x, trip.track_y, trip.track_ts]).astype(np.float32)
        wh = build_nngp_whitening_for_trip(coords, cands.cand_arc_ids, cands.cand_valid, self.nngp_cfg)
        step_conn = precompute_step_connectivity_trip(trip, cands.cand_arc_ids, cands.cand_valid)
        nngp_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        raw_logits = predict_trip_logits_sliding(
            self.model,
            X_scaled,
            cands.cand_valid,
            wh["b_weights"],
            wh["align_k"],
            step_conn,
            device=self.device,
        )
        T = len(trip.track_ts)
        # NNGP spatial residual kriging + temperature scaling are inside model.forward()
        l = np.where(cands.cand_valid, raw_logits.astype(np.float64), -1e9)
        m_max = l.max(axis=1, keepdims=True)
        ex = np.where(cands.cand_valid, np.exp(l - m_max), 0.0)
        cand_probs = (ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
        neural_ms = (time.perf_counter() - t2) * 1000.0

        # Direct calibrated argmax decision (Pure NN-GLS + RLCD — Zero HMM!)
        pred_k = np.argmax(np.where(cands.cand_valid, cand_probs, -1.0), axis=1)
        pred_arcs = cands.cand_arc_ids[np.arange(T), pred_k]
        pred_conf = cand_probs[np.arange(T), pred_k]

        adj, _, _, _ = build_adjacency(trip)
        sp_eval = ShortestPathEvaluator(adj)
        collapsed_pred_arcs = collapse_predicted_route_arcs(trip, pred_arcs, sp_eval=sp_eval)

        # Build predicted route geometry for Leaflet / OpenStreetMap UI
        pred_segments = []
        for aid in collapsed_pred_arcs:
            u, v = int(trip.arcs[aid, 0]), int(trip.arcs[aid, 1])
            pred_segments.append([
                [round(float(trip.node_lat[u]), 6), round(float(trip.node_lon[u]), 6)],
                [round(float(trip.node_lat[v]), 6), round(float(trip.node_lon[v]), 6)],
            ])

        # Nearby road network sample (purely from .nodes/.arcs for map context)
        pad_lon = max(0.003, (trip.track_lon.max() - trip.track_lon.min()) * 0.10)
        pad_lat = max(0.003, (trip.track_lat.max() - trip.track_lat.min()) * 0.10)
        u_lon, u_lat = trip.node_lon[trip.arcs[:, 0]], trip.node_lat[trip.arcs[:, 0]]
        v_lon, v_lat = trip.node_lon[trip.arcs[:, 1]], trip.node_lat[trip.arcs[:, 1]]
        in_box = (
            (u_lon >= trip.track_lon.min() - pad_lon) & (u_lon <= trip.track_lon.max() + pad_lon)
            & (u_lat >= trip.track_lat.min() - pad_lat) & (u_lat <= trip.track_lat.max() + pad_lat)
        )
        box_idx = np.where(in_box)[0]
        if len(box_idx) > 1400:
            box_idx = box_idx[:: max(1, len(box_idx) // 1400)]
        road_context = [
            [
                [round(float(u_lat[i]), 6), round(float(u_lon[i]), 6)],
                [round(float(v_lat[i]), 6), round(float(v_lon[i]), 6)],
            ]
            for i in box_idx
        ]

        # Raw GPS observations + RLCD calibrated confidence (downsampled only for very long >2500 UI render)
        gps_stride = max(1, T // 1600)
        raw_gps = [
            {
                "lat": round(float(trip.track_lat[i]), 6),
                "lon": round(float(trip.track_lon[i]), 6),
                "ts": round(float(trip.track_ts[i]), 1),
                "pred_arc": int(pred_arcs[i]),
                "confidence": round(float(pred_conf[i]), 4),
            }
            for i in range(0, T, gps_stride)
        ]

        # ------------------------------------------------------------------
        # PHASE 2: STRICT SERVER-SIDE EVALUATION (GT NEVER EXPOSED TO UI!)
        # ------------------------------------------------------------------
        metrics: Dict = {"reference_available": False, "ground_truth_exposed": False}
        ref_route = load_route_only(self.eval_dir, trip_id)
        if ref_route is not None:
            ref_arcs_obs, _ = align_track_to_route(trip, ref_route)
            y_idx = np.full(T, -1, dtype=np.int32)
            matches = (cands.cand_arc_ids == ref_arcs_obs[:, None]) & cands.cand_valid
            ht, hk = np.where(matches)
            y_idx[ht] = hk

            # Temporarily attach route ONLY to an internal evaluator copy, never sent out
            eval_trip = TripData(**{**trip.__dict__, "route": ref_route})
            ev = evaluate_predictions(
                cand_probs,
                cands.cand_arc_ids,
                cands.cand_valid,
                y_idx,
                ref_arcs_obs,
                trips=[eval_trip],
                trip_slices=[(0, T)],
                sp_evals={trip_id: sp_eval},
            )
            metrics = {
                "reference_available": True,
                "ground_truth_exposed": False,
                "obs_top1_accuracy": ev["obs_top1_accuracy"],
                "obs_top1_accuracy_given_hit": ev["obs_top1_accuracy_given_hit"],
                "macro_f1": ev["macro_f1"],
                "precision": ev["precision"],
                "recall": ev["recall"],
                "candidate_miss_rate": ev["candidate_miss_rate"],
                "route_continuity": ev["route_continuity"],
                "route_continuity_1hop": ev["route_continuity_1hop"],
                "seq_arc_f1": ev.get("seq_arc_f1", 0.0),
                "seq_arc_precision": ev.get("seq_arc_precision", 0.0),
                "seq_arc_recall": ev.get("seq_arc_recall", 0.0),
                "log_loss": ev["log_loss"],
                "brier_score": ev["brier_score"],
                "ece": ev["ece"],
                "confusion_matrix": ev["confusion_matrix"],
            }

        total_ms = (time.perf_counter() - t_total) * 1000.0

        return {
            "trip_id": trip_id,
            "model_name": "Model 4: NN-GLS + RLCD",
            "algorithm_family": self.config["algorithm_family"],
            "device": str(self.device),
            "n_observations": int(T),
            "n_predicted_route_arcs": len(collapsed_pred_arcs),
            "predicted_arc_sequence": collapsed_pred_arcs,
            "confidence_summary": {
                "mean_confidence": round(float(np.mean(pred_conf)), 4),
                "median_confidence": round(float(np.median(pred_conf)), 4),
                "p10_confidence": round(float(np.percentile(pred_conf, 10)), 4),
                "learned_temperature": round(self.learned_temperature, 4),
            },
            "latency_ms": {
                "feature_extraction_ms": round(feat_ms, 2),
                "nngp_graph_ms": round(nngp_ms, 2),
                "neural_inference_ms": round(neural_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "bounds": {
                "min_lat": float(trip.track_lat.min()),
                "max_lat": float(trip.track_lat.max()),
                "min_lon": float(trip.track_lon.min()),
                "max_lon": float(trip.track_lon.max()),
            },
            "raw_gps": raw_gps,
            "predicted_route_segments": pred_segments,
            "road_network_context": road_context,
            "metrics": metrics,
        }
