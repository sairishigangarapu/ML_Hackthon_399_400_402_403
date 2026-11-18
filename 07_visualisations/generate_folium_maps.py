"""Stage 07: Interactive Folium (OpenStreetMap Free Tier) Maps + High-Res Map Visualisations.

Visualises ALL mandatory layers from Deliverables.pdf Section 6:
  - Original GPS observations (.track)
  - Relevant road-network geometry (.nodes / .arcs)
  - Reference route (.route)
  - Model 4 (NN-GLS + RLCD) Predicted route
with distinguishable colors, legends, fit-bounds zoom, and LayerControl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_data_processing"))
from config import DIR_MAPS, DIR_PLOTS, DIR_RESULTS, STUDENT_TEST_DIR
from io_utils import TripData, build_adjacency, load_trip

sys.path.insert(0, str(ROOT / "06_model_evaluation"))
from metrics_model4 import ShortestPathEvaluator, collapse_predicted_route_arcs


def build_route_comparison_folium_map(
    trip: TripData,
    pred_arcs_obs: np.ndarray,
    ref_arcs_obs: np.ndarray,
    probs_obs: np.ndarray,
    trip_metrics: Dict,
    out_html: Path,
    max_bg_arcs: int = 2500,
) -> None:
    """Create interactive OpenStreetMap Folium map comparing GPS, road network, reference & Model 4 predicted route."""
    center_lat = float(np.mean(trip.track_lat))
    center_lon = float(np.mean(trip.track_lon))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=None)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (Free Tier)").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors | Style: OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap (Free OSM Tier)",
    ).add_to(m)

    # 1. Road-network context within padded bounding box
    pad_lon = max(0.004, (trip.track_lon.max() - trip.track_lon.min()) * 0.12)
    pad_lat = max(0.004, (trip.track_lat.max() - trip.track_lat.min()) * 0.12)
    lon_lo, lon_hi = trip.track_lon.min() - pad_lon, trip.track_lon.max() + pad_lon
    lat_lo, lat_hi = trip.track_lat.min() - pad_lat, trip.track_lat.max() + pad_lat

    u_lon, u_lat = trip.node_lon[trip.arcs[:, 0]], trip.node_lat[trip.arcs[:, 0]]
    v_lon, v_lat = trip.node_lon[trip.arcs[:, 1]], trip.node_lat[trip.arcs[:, 1]]
    in_box = (u_lon >= lon_lo) & (u_lon <= lon_hi) & (u_lat >= lat_lo) & (u_lat <= lat_hi)
    box_idx = np.where(in_box)[0]
    if len(box_idx) > max_bg_arcs:
        box_idx = box_idx[:: max(1, len(box_idx) // max_bg_arcs)]

    fg_net = folium.FeatureGroup(name="Road Network Geometry", show=True)
    for idx in box_idx:
        folium.PolyLine(
            [[float(u_lat[idx]), float(u_lon[idx])], [float(v_lat[idx]), float(v_lon[idx])]],
            color="#64748b",
            weight=1.3,
            opacity=0.45,
        ).add_to(fg_net)
    fg_net.add_to(m)

    # 2. Reference Route (.route)
    fg_ref = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    if trip.route is not None:
        for aid in trip.route:
            u, v = int(trip.arcs[aid, 0]), int(trip.arcs[aid, 1])
            folium.PolyLine(
                [[float(trip.node_lat[u]), float(trip.node_lon[u])],
                 [float(trip.node_lat[v]), float(trip.node_lon[v])]],
                color="#16a34a",
                weight=6.5,
                opacity=0.65,
            ).add_to(fg_ref)
    fg_ref.add_to(m)

    # 3. Model 4 (NN-GLS + RLCD) Predicted Route
    sp_eval = ShortestPathEvaluator(build_adjacency(trip)[0])
    collapsed_pred = collapse_predicted_route_arcs(trip, pred_arcs_obs, sp_eval=sp_eval)
    fg_pred = folium.FeatureGroup(name="Model 4 (NN-GLS + RLCD) Predicted Route", show=True)
    for aid in collapsed_pred:
        u, v = int(trip.arcs[aid, 0]), int(trip.arcs[aid, 1])
        folium.PolyLine(
            [[float(trip.node_lat[u]), float(trip.node_lon[u])],
             [float(trip.node_lat[v]), float(trip.node_lon[v])]],
            color="#dc2626",
            weight=3.2,
            opacity=0.92,
            dash_array="7,4",
        ).add_to(fg_pred)
    fg_pred.add_to(m)

    # 4. Raw GPS Observations (with RLCD confidence tooltips) + Mismatch Highlights
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    fg_err = folium.FeatureGroup(name="Mismatched GPS Points (Diagnostic)", show=False)
    top_conf = probs_obs.max(axis=1)
    stride = max(1, len(trip.track_lat) // 1000)

    for i in range(0, len(trip.track_lat), stride):
        ok = bool(pred_arcs_obs[i] == ref_arcs_obs[i])
        folium.CircleMarker(
            [float(trip.track_lat[i]), float(trip.track_lon[i])],
            radius=2.6,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.72,
            tooltip=f"t={trip.track_ts[i]:.0f}s | pred_arc={int(pred_arcs_obs[i])} | RLCD conf={top_conf[i]:.2f} | match={ok}",
        ).add_to(fg_gps)

    err_idx = np.where(pred_arcs_obs != ref_arcs_obs)[0]
    err_stride = max(1, len(err_idx) // 300) if len(err_idx) > 0 else 1
    for i in err_idx[::err_stride]:
        folium.CircleMarker(
            [float(trip.track_lat[i]), float(trip.track_lon[i])],
            radius=4.0,
            color="#b91c1c",
            fill=True,
            fill_color="#f59e0b",
            fill_opacity=0.9,
            tooltip=f"MISMATCH t={trip.track_ts[i]:.0f}s | pred={int(pred_arcs_obs[i])} vs ref={int(ref_arcs_obs[i])} | conf={top_conf[i]:.2f}",
        ).add_to(fg_err)

    folium.Marker(
        [float(trip.track_lat[0]), float(trip.track_lon[0])],
        tooltip=f"Start (Trip {trip.trip_id})",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fg_gps)
    folium.Marker(
        [float(trip.track_lat[-1]), float(trip.track_lon[-1])],
        tooltip=f"End (Trip {trip.trip_id})",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fg_gps)

    fg_gps.add_to(m)
    fg_err.add_to(m)

    m.fit_bounds([[float(trip.track_lat.min()), float(trip.track_lon.min())],
                  [float(trip.track_lat.max()), float(trip.track_lon.max())]])

    legend_html = f"""
    <div style="position:fixed; bottom:24px; left:24px; z-index:9999; background:rgba(255,255,255,0.96);
                padding:12px 16px; border-radius:10px; border:1px solid #cbd5e1;
                box-shadow:0 4px 12px rgba(0,0,0,0.12); font-family:sans-serif; font-size:12px; line-height:1.55;">
      <div style="font-weight:700; font-size:13px; margin-bottom:4px;">
        PathSnap Model 4 (NN-GLS + RLCD) — Trip {trip.trip_id}
      </div>
      <div><span style="color:#2563eb; font-size:14px;">&#9679;</span> Raw GPS Observations ({len(trip.track_ts):,} pts)</div>
      <div><span style="color:#16a34a; font-weight:900;">&#9473;&#9473;</span> Reference Route ({len(trip.route) if trip.route is not None else 0} arcs)</div>
      <div><span style="color:#dc2626; font-weight:900;">- - -</span> Model 4 Predicted Route ({len(collapsed_pred)} arcs)</div>
      <div><span style="color:#64748b;">&#9473;&#9473;</span> Road Network Context</div>
      <hr style="margin:6px 0; border:none; border-top:1px solid #e2e8f0;">
      <div>
        <b>Top-1 Acc:</b> {trip_metrics['obs_top1_accuracy']*100:.2f}% &nbsp;|&nbsp;
        <b>Continuity:</b> {trip_metrics['route_continuity']*100:.2f}%<br>
        <b>Seq Arc F1:</b> {trip_metrics['seq_arc_f1']:.4f} &nbsp;|&nbsp;
        <b>ECE:</b> {trip_metrics['ece']:.4f} &nbsp;|&nbsp;
        <b>LogLoss:</b> {trip_metrics['log_loss']:.4f}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def plot_static_route_comparison_panel(
    showcase_items: List[tuple],
    out_png: Path,
) -> None:
    """Generate clean high-resolution 2x2 geographic comparison panel for the PDF report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
    for ax, (trip, pred_arcs, ref_arcs, tm) in zip(axes.ravel(), showcase_items):
        xmin, xmax = trip.track_x.min() - 200, trip.track_x.max() + 200
        ymin, ymax = trip.track_y.min() - 200, trip.track_y.max() + 200
        mask = (trip.arc_x1 >= xmin) & (trip.arc_x1 <= xmax) & (trip.arc_y1 >= ymin) & (trip.arc_y1 <= ymax)
        idx = np.where(mask)[0]
        if len(idx) > 4500:
            idx = idx[:: max(1, len(idx) // 4500)]
        for i in idx:
            ax.plot([trip.arc_x1[i], trip.arc_x2[i]], [trip.arc_y1[i], trip.arc_y2[i]],
                    color="#cbd5e1", lw=0.55, alpha=0.65, zorder=1)
        # Reference route
        for aid in trip.route:
            ax.plot([trip.arc_x1[aid], trip.arc_x2[aid]], [trip.arc_y1[aid], trip.arc_y2[aid]],
                    color="#16a34a", lw=3.8, alpha=0.65, zorder=2)
        # Model 4 predicted unique arcs
        uniq_pred = np.unique(pred_arcs[pred_arcs >= 0])
        for aid in uniq_pred:
            ax.plot([trip.arc_x1[aid], trip.arc_x2[aid]], [trip.arc_y1[aid], trip.arc_y2[aid]],
                    color="#dc2626", lw=1.8, linestyle="--", alpha=0.95, zorder=3)
        # Raw GPS
        ax.scatter(trip.track_x, trip.track_y, s=3.0, c="#2563eb", alpha=0.45, zorder=4)
        ax.plot([], [], color="#94a3b8", lw=0.8, label="Road Network")
        ax.plot([], [], color="#16a34a", lw=3.5, alpha=0.7, label="Reference Route")
        ax.plot([], [], color="#dc2626", lw=2.0, ls="--", label="Model 4 (NN-GLS+RLCD)")
        ax.scatter([], [], s=18, c="#2563eb", label="Raw GPS")
        ax.set_title(
            f"Student-Test Trip {trip.trip_id} | Top-1={tm['obs_top1_accuracy']*100:.2f}% | "
            f"Cont={tm['route_continuity']*100:.2f}% | ECE={tm['ece']:.4f}",
            fontsize=9.8, fontweight="bold",
        )
        ax.set_xlabel("Local East (m)")
        ax.set_ylabel("Local North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=7.8)
    plt.suptitle("Model 4 (NN-GLS + RLCD): Route Reconstruction vs. Reference Route (Student-Test)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    preds = np.load(DIR_RESULTS / "model4_student_test_predictions.npz", allow_pickle=True)
    df_trips = pd.read_csv(DIR_RESULTS / "model4_student_test_per_trip.csv")
    metrics_by_id = {str(r["trip_id"]).zfill(8): r for r in df_trips.to_dict(orient="records")}

    trip_ids = [str(t) for t in preds["trip_ids"]]
    starts = preds["trip_starts"]
    ends = preds["trip_ends"]
    pred_arcs_all = preds["pred_arcs_model4"]
    ref_arcs_all = preds["ref_arc"]
    probs_all = preds["probs_model4_nngls_rlcd"]

    # Generate interactive Folium OpenStreetMap HTML maps for ALL 10 student_test trips!
    showcase = []
    showcase_ids = {"00000000", "00000009", "00000040", "00000056"}
    for tid, s, e in zip(trip_ids, starts, ends):
        s, e = int(s), int(e)
        trip = load_trip(STUDENT_TEST_DIR, tid, "student_test", load_route=True)
        tm = metrics_by_id[tid]
        out_html = DIR_MAPS / f"model4_map_student_test_{tid}.html"
        build_route_comparison_folium_map(
            trip,
            pred_arcs_all[s:e],
            ref_arcs_all[s:e],
            probs_all[s:e],
            tm,
            out_html,
        )
        if tid in showcase_ids:
            showcase.append((trip, pred_arcs_all[s:e], ref_arcs_all[s:e], tm))

    plot_static_route_comparison_panel(showcase, DIR_PLOTS / "model4_route_reconstruction_panel.png")
    print(f"[Stage 07 Folium Maps] Generated {len(trip_ids)} interactive OpenStreetMap Folium HTML maps in {DIR_MAPS}")


if __name__ == "__main__":
    main()
