"""Laptop 1 FastAPI Backend for PathSnap — Model 4 (NN-GLS + RLCD).

Start with:
  uvicorn main:app --host 0.0.0.0 --port 8000

Switch evaluation directory for final evaluation via:
  PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_engine import Model4InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pathsnap.api")

engine: Optional[Model4InferenceEngine] = None
request_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("=" * 76)
    logger.info("[PATHSNAP-API] Booting Laptop 1 Inference Server — Model 4 (NN-GLS + RLCD)")
    engine = Model4InferenceEngine()
    trips = engine.list_evaluation_trips()
    logger.info(
        "[PATHSNAP-API] Loaded Model 4 (NN-GLS + RLCD) | params=%d | device=%s | temp=%.4f",
        engine.n_parameters,
        engine.device,
        engine.learned_temperature,
    )
    logger.info(
        "[PATHSNAP-API] Evaluation dir: %s (%d trips available) | Ground truth: SERVER-SIDE ONLY",
        engine.eval_dir,
        len(trips),
    )
    logger.info("=" * 76)
    yield
    logger.info("[PATHSNAP-API] Shutting down Laptop 1 Inference Server.")


app = FastAPI(
    title="PathSnap Model 4 API (NN-GLS + RLCD)",
    description="Distributed Map-Matching & Route Reconstruction API powered by NN-GLS + RLCD.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable cross-laptop CORS so Laptop 2 Web UI can call Laptop 1 API over LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Evaluation trip identifier (e.g., '00000000')")


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    trips = engine.list_evaluation_trips()
    return {
        "status": "ok",
        "model_name": "Model 4: NN-GLS + RLCD",
        "device": str(engine.device),
        "n_parameters": engine.n_parameters,
        "learned_temperature": engine.learned_temperature,
        "eval_dir": str(engine.eval_dir),
        "n_trips": len(trips),
        "ground_truth_isolation": "strict_server_side_only",
    }


@app.get("/trips")
def get_trips():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    trips = engine.list_evaluation_trips()
    logger.info("[PATHSNAP-API] GET /trips -> returned %d evaluation trips", len(trips))
    return {"eval_dir": str(engine.eval_dir), "count": len(trips), "trips": trips}


@app.post("/predict")
def predict_trip(req: PredictRequest):
    global request_counter
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    avail = {t["trip_id"] for t in engine.list_evaluation_trips()}
    if req.trip_id not in avail:
        raise HTTPException(status_code=404, detail=f"Trip '{req.trip_id}' not found in {engine.eval_dir}")

    request_counter += 1
    req_id = f"PS-{request_counter:04d}-{uuid.uuid4().hex[:6].upper()}"
    ts_utc = datetime.now(timezone.utc).isoformat()

    logger.info("-" * 76)
    logger.info("[PATHSNAP-API] POST /predict | req_id=%s | trip_id=%s | model=Model 4 (NN-GLS + RLCD)", req_id, req.trip_id)

    result = engine.predict_and_evaluate(req.trip_id)
    result["request_id"] = req_id
    result["timestamp_utc"] = ts_utc

    m = result["metrics"]
    lat = result["latency_ms"]
    if m.get("reference_available"):
        logger.info(
            "[PATHSNAP-API] DONE req_id=%s | trip=%s | obs=%d | Top1=%.2f%% | MacroF1=%.4f | "
            "Cont=%.2f%% | SeqF1=%.4f | LogLoss=%.4f | Brier=%.4f | ECE=%.4f | total=%.1fms",
            req_id,
            req.trip_id,
            result["n_observations"],
            m["obs_top1_accuracy"] * 100.0,
            m["macro_f1"],
            m["route_continuity"] * 100.0,
            m["seq_arc_f1"],
            m["log_loss"],
            m["brier_score"],
            m["ece"],
            lat["total_ms"],
        )
    else:
        logger.info(
            "[PATHSNAP-API] DONE req_id=%s | trip=%s | obs=%d | pred_arcs=%d | total=%.1fms (no ref .route)",
            req_id,
            req.trip_id,
            result["n_observations"],
            result["n_predicted_route_arcs"],
            lat["total_ms"],
        )
    logger.info("-" * 76)
    return result
