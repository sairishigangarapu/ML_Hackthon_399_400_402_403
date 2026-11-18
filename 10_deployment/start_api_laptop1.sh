#!/usr/bin/env bash
# Launch Laptop 1: FastAPI Inference & Server-Side Evaluation Backend (Model 4: NN-GLS + RLCD)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"

export PATHSNAP_EVAL_DIR="${PATHSNAP_EVAL_DIR:-$PROJECT_ROOT/../Student folder/student_test}"
HOST="${PATHSNAP_API_HOST:-0.0.0.0}"
PORT="${PATHSNAP_API_PORT:-8000}"

echo "========================================================================"
echo " PathSnap — Laptop 1 FastAPI Backend (Model 4: NN-GLS + RLCD)"
echo " Eval Directory : $PATHSNAP_EVAL_DIR"
echo " Binding        : http://$HOST:$PORT"
echo " Health Check   : http://localhost:$PORT/health"
echo " OpenAPI Docs   : http://localhost:$PORT/docs"
echo "========================================================================"

cd "$PROJECT_ROOT/08_api"
exec "$VENV_PY" -m uvicorn main:app --host "$HOST" --port "$PORT" --log-level info
