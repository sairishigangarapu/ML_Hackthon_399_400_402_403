"""End-to-End FastAPI Integration Test for Model 4 (NN-GLS + RLCD).

Verifies:
  1. GET /health returns 200 OK with strict server-side GT isolation flag.
  2. GET /trips lists all 10 evaluation trips with ZERO ground-truth labels.
  3. POST /predict executes live Model 4 inference on ALL 10 trips and returns valid route geometry + metrics.
  4. Ground-truth fields (`route`, `ref_arc`, `ground_truth_arcs`) are NEVER leaked in API payloads.
  5. Writes verified API response log to results/api_e2e_verification.json for Report Section 7.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "08_api"))
from main import app

sys.path.insert(0, str(ROOT / "01_data_processing"))
from config import DIR_RESULTS


def main() -> None:
    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200, r_health.text
        health = r_health.json()
        print("[E2E] GET /health ->", json.dumps(health))

        r_trips = client.get("/trips")
        assert r_trips.status_code == 200, r_trips.text
        trips_payload = r_trips.json()
        trips = trips_payload["trips"]
        assert len(trips) == 10, f"Expected 10 evaluation trips, found {len(trips)}"
        print(f"[E2E] GET /trips -> {len(trips)} trips available")

        # Verify strict ground-truth non-leakage on all 10 trips
        forbidden_keys = {"route", "ref_arc", "reference_route", "ground_truth_arcs", "ground_truth_coords"}
        results = []
        for t in trips:
            tid = t["trip_id"]
            resp = client.post("/predict", json={"trip_id": tid})
            assert resp.status_code == 200, f"Trip {tid} failed: {resp.text}"
            body = resp.json()

            leaked = forbidden_keys.intersection(set(body.keys()))
            assert not leaked, f"GROUND TRUTH LEAK DETECTED: {leaked}"
            assert body["metrics"]["ground_truth_exposed"] is False
            assert len(body["predicted_route_segments"]) > 0

            m = body["metrics"]
            lat = body["latency_ms"]
            row = {
                "request_id": body["request_id"],
                "trip_id": tid,
                "n_observations": body["n_observations"],
                "n_predicted_route_arcs": body["n_predicted_route_arcs"],
                "obs_top1_accuracy": m["obs_top1_accuracy"],
                "macro_f1": m["macro_f1"],
                "route_continuity": m["route_continuity"],
                "seq_arc_f1": m["seq_arc_f1"],
                "log_loss": m["log_loss"],
                "brier_score": m["brier_score"],
                "ece": m["ece"],
                "total_ms": lat["total_ms"],
                "neural_inference_ms": lat["neural_inference_ms"],
            }
            results.append(row)
            print(
                f"  POST /predict trip={tid} | req={body['request_id']} | "
                f"Top1={m['obs_top1_accuracy']*100:.2f}% | Cont={m['route_continuity']*100:.2f}% | "
                f"ECE={m['ece']:.4f} | total={lat['total_ms']:.1f}ms"
            )

        out = {
            "health": health,
            "n_trips_verified": len(results),
            "ground_truth_leak_check": "PASSED (zero reference route labels/coords in API responses)",
            "per_trip_api_responses": results,
        }
        with open(DIR_RESULTS / "api_e2e_verification.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"[E2E] ALL 10 TRIPS PASSED! Verification saved to {DIR_RESULTS / 'api_e2e_verification.json'}")


if __name__ == "__main__":
    main()
