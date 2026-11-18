#!/usr/bin/env bash
# Launch Laptop 2: Evaluator-Facing Web UI (Leaflet + OpenStreetMap Free Tier)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${PATHSNAP_UI_PORT:-3000}"

echo "========================================================================"
echo " PathSnap — Laptop 2 Evaluator Web UI (OpenStreetMap Free Tier)"
echo " Open in browser: http://localhost:$PORT"
echo " Point the API URL bar to: http://<LAPTOP_1_LAN_IP>:8000"
echo "========================================================================"

cd "$PROJECT_ROOT/09_ui"
exec python3 -m http.server "$PORT" --bind 0.0.0.0
