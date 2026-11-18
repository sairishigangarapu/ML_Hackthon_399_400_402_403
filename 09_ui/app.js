// PathSnap Laptop 2 Evaluator UI — Model 4 (NN-GLS + RLCD) + OpenStreetMap Free Tier

const state = {
  apiUrl: "http://localhost:8000",
  activeTripId: null,
  lastPayload: null,
  map: null,
  layers: {
    road: null,
    pred: null,
    gps: null,
  },
};

function el(id) {
  return document.getElementById(id);
}

function appendLog(msg, cls = "") {
  const box = el("auditLog");
  const entry = document.createElement("div");
  entry.className = `log-entry ${cls}`.trim();
  const ts = new Date().toLocaleTimeString();
  entry.textContent = `[${ts}] ${msg}`;
  box.prepend(entry);
}

function initMap() {
  state.map = L.map("map", { zoomControl: true }).setView([48.85, 11.5], 5);

  const osmFree = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  });

  const openTopo = L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
    maxZoom: 17,
    attribution: "Map data: &copy; OpenStreetMap contributors | OpenTopoMap (CC-BY-SA)",
  });

  osmFree.addTo(state.map);
  L.control.layers(
    {
      "OpenStreetMap (Free Tier)": osmFree,
      "OpenTopoMap (Free OSM)": openTopo,
    },
    {},
    { collapsed: true }
  ).addTo(state.map);

  state.layers.road = L.layerGroup().addTo(state.map);
  state.layers.pred = L.layerGroup().addTo(state.map);
  state.layers.gps = L.layerGroup().addTo(state.map);
}

function confidenceColor(c) {
  if (c >= 0.85) return "#22c55e";
  if (c >= 0.55) return "#f59e0b";
  return "#ef4444";
}

function renderTripPayload(data) {
  state.lastPayload = data;
  state.layers.road.clearLayers();
  state.layers.pred.clearLayers();
  state.layers.gps.clearLayers();

  const showRoad = el("toggleRoad").checked;
  const showPred = el("togglePred").checked;
  const showGps = el("toggleGps").checked;
  const colorByConf = el("toggleConfColor").checked;

  // 1. Road network context
  if (showRoad && Array.isArray(data.road_network_context)) {
    for (const seg of data.road_network_context) {
      L.polyline(seg, { color: "#64748b", weight: 1.4, opacity: 0.45 }).addTo(state.layers.road);
    }
  }

  // 2. Predicted route (Model 4: NN-GLS + RLCD)
  if (showPred && Array.isArray(data.predicted_route_segments)) {
    for (const seg of data.predicted_route_segments) {
      L.polyline(seg, { color: "#ef4444", weight: 4.2, opacity: 0.92 }).addTo(state.layers.pred);
    }
  }

  // 3. Raw GPS observations with RLCD confidence popups + free KartaView/OSM street context link
  if (showGps && Array.isArray(data.raw_gps)) {
    for (const pt of data.raw_gps) {
      const col = colorByConf ? confidenceColor(pt.confidence) : "#38bdf8";
      const osmStreetUrl = `https://www.openstreetmap.org/?mlat=${pt.lat}&mlon=${pt.lon}#map=18/${pt.lat}/${pt.lon}`;
      const marker = L.circleMarker([pt.lat, pt.lon], {
        radius: 3.2,
        color: col,
        fillColor: col,
        fillOpacity: 0.85,
        weight: 1,
      });
      marker.bindPopup(
        `<strong>Trip ${data.trip_id}</strong><br>` +
        `Timestamp: ${pt.ts}s<br>` +
        `Predicted Arc: <code>${pt.pred_arc}</code><br>` +
        `RLCD Calibrated Conf: <strong>${(pt.confidence * 100).toFixed(1)}%</strong><br>` +
        `<a href="${osmStreetUrl}" target="_blank" rel="noopener noreferrer">Inspect on OpenStreetMap &nearr;</a>`
      );
      marker.addTo(state.layers.gps);
    }
    if (data.raw_gps.length > 0) {
      const first = data.raw_gps[0];
      const last = data.raw_gps[data.raw_gps.length - 1];
      L.circleMarker([first.lat, first.lon], { radius: 7, color: "#fff", fillColor: "#16a34a", fillOpacity: 1, weight: 2 })
        .bindTooltip("Trip Start").addTo(state.layers.gps);
      L.circleMarker([last.lat, last.lon], { radius: 7, color: "#fff", fillColor: "#dc2626", fillOpacity: 1, weight: 2 })
        .bindTooltip("Trip End").addTo(state.layers.gps);
    }
  }

  // Fit bounds
  if (data.bounds) {
    const b = data.bounds;
    state.map.fitBounds([[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]], { padding: [26, 26] });
    const cLat = ((b.min_lat + b.max_lat) / 2).toFixed(5);
    const cLon = ((b.min_lon + b.max_lon) / 2).toFixed(5);
    el("osmExternalLink").href = `https://www.openstreetmap.org/#map=14/${cLat}/${cLon}`;
  }

  // Update KPI cards
  const m = data.metrics || {};
  const lat = data.latency_ms || {};
  el("activeTripTitle").textContent = `Trip ${data.trip_id} — ${data.model_name}`;
  el("activeTripMeta").textContent = `${data.n_observations.toLocaleString()} GPS obs | ${data.n_predicted_route_arcs} route arcs | Device: ${data.device} | Req: ${data.request_id}`;

  if (m.reference_available) {
    el("kpiTop1").textContent = `${(m.obs_top1_accuracy * 100).toFixed(2)}%`;
    el("kpiHitAcc").textContent = `Hit-cond: ${(m.obs_top1_accuracy_given_hit * 100).toFixed(2)}%`;
    el("kpiF1").textContent = m.macro_f1.toFixed(4);
    el("kpiPrecRec").textContent = `P: ${m.precision.toFixed(3)} / R: ${m.recall.toFixed(3)}`;
    el("kpiCont").textContent = `${(m.route_continuity * 100).toFixed(2)}%`;
    el("kpiCont1Hop").textContent = `1-hop: ${(m.route_continuity_1hop * 100).toFixed(2)}%`;
    el("kpiSeqF1").textContent = m.seq_arc_f1.toFixed(4);
    el("kpiArcCount").textContent = `Pred arcs: ${data.n_predicted_route_arcs}`;
    el("kpiEce").textContent = m.ece.toFixed(4);
    el("kpiBrierLL").textContent = `Brier: ${m.brier_score.toFixed(4)} | LL: ${m.log_loss.toFixed(4)}`;
    el("kpiLatencySub").textContent = `Miss rate: ${(m.candidate_miss_rate * 100).toFixed(2)}%`;
  } else {
    el("kpiTop1").textContent = "N/A";
    el("kpiHitAcc").textContent = "No server .route";
    el("kpiF1").textContent = "N/A";
    el("kpiPrecRec").textContent = "Unlabeled trip";
    el("kpiCont").textContent = "N/A";
    el("kpiCont1Hop").textContent = "--";
    el("kpiSeqF1").textContent = "N/A";
    el("kpiArcCount").textContent = `Pred arcs: ${data.n_predicted_route_arcs}`;
    el("kpiEce").textContent = `Conf ${(data.confidence_summary.mean_confidence * 100).toFixed(1)}%`;
    el("kpiBrierLL").textContent = `Temp: ${data.confidence_summary.learned_temperature}`;
    el("kpiLatencySub").textContent = "Live inference";
  }
  el("kpiLatency").textContent = `${lat.total_ms.toFixed(0)} ms`;
}

async function selectTrip(tripId) {
  state.activeTripId = tripId;
  document.querySelectorAll(".trip-card").forEach((c) => {
    c.classList.toggle("active", c.dataset.tripId === tripId);
  });

  el("activeTripTitle").textContent = `Running Model 4 (NN-GLS + RLCD) on Trip ${tripId}...`;
  el("activeTripMeta").textContent = `POST ${state.apiUrl}/predict`;
  appendLog(`POST /predict trip=${tripId} -> Laptop 1`);

  try {
    const resp = await fetch(`${state.apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${txt}`);
    }
    const data = await resp.json();
    renderTripPayload(data);
    const accStr = data.metrics?.reference_available
      ? `Top1=${(data.metrics.obs_top1_accuracy * 100).toFixed(2)}% Cont=${(data.metrics.route_continuity * 100).toFixed(2)}% ECE=${data.metrics.ece.toFixed(4)}`
      : `pred_arcs=${data.n_predicted_route_arcs}`;
    appendLog(`200 OK ${data.request_id} | ${tripId} | ${accStr} | ${data.latency_ms.total_ms.toFixed(0)}ms`, "ok");
  } catch (err) {
    appendLog(`ERROR ${tripId}: ${err.message}`, "err");
    el("activeTripTitle").textContent = `Prediction failed for ${tripId}`;
    el("activeTripMeta").textContent = err.message;
  }
}

async function connectApi() {
  state.apiUrl = el("apiUrlInput").value.trim().replace(/\/+$/, "");
  const badge = el("healthBadge");
  badge.className = "status-pill status-idle";
  badge.textContent = "Connecting...";

  try {
    const hResp = await fetch(`${state.apiUrl}/health`);
    if (!hResp.ok) throw new Error(`HTTP ${hResp.status}`);
    const health = await hResp.json();
    badge.className = "status-pill status-ok";
    badge.textContent = `Connected (${health.device})`;
    appendLog(`Connected to ${state.apiUrl} | ${health.model_name} | ${health.n_trips} trips`, "ok");

    const tResp = await fetch(`${state.apiUrl}/trips`);
    const tData = await tResp.json();
    const list = el("tripList");
    list.innerHTML = "";
    el("tripCountBadge").textContent = `${tData.count} trips`;

    for (const t of tData.trips) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "trip-card";
      btn.dataset.tripId = t.trip_id;
      btn.innerHTML = `
        <div class="trip-card-top">
          <span class="trip-id">Trip ${t.trip_id}</span>
          <span class="trip-tag">${t.n_gps.toLocaleString()} GPS pts</span>
        </div>
        <div class="trip-meta">${t.n_arcs.toLocaleString()} arcs &middot; ${t.duration_s}s &middot; (${t.center_lat.toFixed(2)}, ${t.center_lon.toFixed(2)})</div>
      `;
      btn.addEventListener("click", () => selectTrip(t.trip_id));
      list.appendChild(btn);
    }
  } catch (err) {
    badge.className = "status-pill status-err";
    badge.textContent = "API Unreachable";
    appendLog(`Failed connecting to ${state.apiUrl}: ${err.message}`, "err");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  el("connectBtn").addEventListener("click", connectApi);
  ["toggleGps", "togglePred", "toggleRoad", "toggleConfColor"].forEach((id) => {
    el(id).addEventListener("change", () => {
      if (state.lastPayload) renderTripPayload(state.lastPayload);
    });
  });
  connectApi();
});
