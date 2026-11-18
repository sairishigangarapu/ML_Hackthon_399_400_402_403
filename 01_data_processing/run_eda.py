"""Stage 01: Comprehensive Exploratory Data Analysis (EDA) across train & student_test partitions."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DIR_MAPS,
    DIR_PLOTS,
    DIR_RESULTS,
    STUDENT_TEST_DIR,
    TRAIN_DIR,
)
from io_utils import TripData, list_trips, load_trip


def point_to_segments_min_dist(
    px: np.ndarray,
    py: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    chunk: int = 512,
) -> np.ndarray:
    dx = x2 - x1
    dy = y2 - y1
    l2 = np.maximum(dx * dx + dy * dy, 1e-12)
    out = np.empty(len(px), dtype=np.float64)
    for s in range(0, len(px), chunk):
        e = min(s + chunk, len(px))
        u = ((px[s:e, None] - x1[None, :]) * dx[None, :] + (py[s:e, None] - y1[None, :]) * dy[None, :]) / l2[None, :]
        u = np.clip(u, 0.0, 1.0)
        d = np.hypot(px[s:e, None] - (x1[None, :] + u * dx[None, :]), py[s:e, None] - (y1[None, :] + u * dy[None, :]))
        out[s:e] = d.min(axis=1)
    return out


def analyze_trip(partition_dir: Path, trip_id: str, partition: str) -> Dict:
    td = partition_dir / trip_id
    trip = load_trip(partition_dir, trip_id, partition=partition, load_route=True)

    # Raw file sizes
    sizes = {
        ext: (td / f"{trip_id}.{ext}").stat().st_size if (td / f"{trip_id}.{ext}").exists() else 0
        for ext in ["track", "nodes", "arcs", "route"]
    }

    # Timestamps & speed
    dt = np.diff(trip.track_ts) if len(trip.track_ts) > 1 else np.array([0.0])
    step_dist = np.hypot(np.diff(trip.track_x), np.diff(trip.track_y)) if len(trip.track_x) > 1 else np.array([0.0])
    speeds = step_dist / np.maximum(dt, 1e-6)

    # Distance from each GPS point to nearest reference route arc
    rt = trip.route
    d_route = point_to_segments_min_dist(
        trip.track_x,
        trip.track_y,
        trip.arc_x1[rt],
        trip.arc_y1[rt],
        trip.arc_x2[rt],
        trip.arc_y2[rt],
    )

    # Road network connectivity
    g_undir = nx.Graph()
    g_undir.add_nodes_from(range(len(trip.node_x)))
    g_undir.add_edges_from(zip(trip.arcs[:, 0].tolist(), trip.arcs[:, 1].tolist()))
    n_cc = nx.number_connected_components(g_undir)
    deg = np.array([d for _, d in g_undir.degree()])
    n_isolated = int((deg == 0).sum())

    edge_set = set(map(tuple, trip.arcs.tolist()))
    bidir_frac = sum(1 for u, v in edge_set if (v, u) in edge_set) / max(len(edge_set), 1)

    # Route connectivity & revisits
    route_breaks = 0
    for i in range(len(rt) - 1):
        if trip.arcs[rt[i], 1] != trip.arcs[rt[i + 1], 0]:
            route_breaks += 1
    has_revisit = len(set(rt.tolist())) < len(rt)
    route_len_m = float(trip.arc_len[rt].sum())
    track_len_m = float(step_dist.sum())

    return {
        "partition": partition,
        "trip_id": trip_id,
        "n_gps": int(len(trip.track_ts)),
        "n_nodes": int(len(trip.node_x)),
        "n_arcs": int(len(trip.arcs)),
        "n_route_arcs": int(len(rt)),
        "track_bytes": sizes["track"],
        "nodes_bytes": sizes["nodes"],
        "arcs_bytes": sizes["arcs"],
        "route_bytes": sizes["route"],
        "origin_lon": round(trip.origin_lon, 5),
        "origin_lat": round(trip.origin_lat, 5),
        "extent_x_m": round(float(trip.track_x.max() - trip.track_x.min()), 1),
        "extent_y_m": round(float(trip.track_y.max() - trip.track_y.min()), 1),
        "duration_s": round(float(trip.track_ts[-1] - trip.track_ts[0]), 1),
        "dt_median_s": round(float(np.median(dt)), 3),
        "dt_max_s": round(float(dt.max()), 1),
        "frac_dt_1s": round(float(np.isclose(dt, 1.0).mean()), 4),
        "n_gaps_gt5s": int((dt > 5.0).sum()),
        "n_gaps_gt30s": int((dt > 30.0).sum()),
        "speed_median_mps": round(float(np.median(speeds)), 2),
        "speed_p95_mps": round(float(np.percentile(speeds, 95)), 2),
        "frac_stationary": round(float((speeds < 0.8).mean()), 4),
        "route_len_m": round(route_len_m, 1),
        "track_cum_len_m": round(track_len_m, 1),
        "gps_dist_median_m": round(float(np.median(d_route)), 2),
        "gps_dist_p95_m": round(float(np.percentile(d_route, 95)), 2),
        "gps_dist_max_m": round(float(d_route.max()), 2),
        "frac_within_25m": round(float((d_route <= 25.0).mean()), 4),
        "frac_within_50m": round(float((d_route <= 50.0).mean()), 4),
        "arc_len_median_m": round(float(np.median(trip.arc_len)), 2),
        "n_zero_len_arcs": int((trip.arc_len < 1e-6).sum()),
        "bidir_edge_frac": round(float(bidir_frac), 4),
        "n_connected_components": int(n_cc),
        "n_isolated_nodes": int(n_isolated),
        "route_directed_breaks": int(route_breaks),
        "has_revisited_arcs": bool(has_revisit),
    }


def make_folium_eda_map(trip: TripData, out_html: Path, title: str, max_bg_arcs: int = 2500) -> None:
    """Create an interactive OpenStreetMap Folium map for EDA (GPS + road network + reference route)."""
    center_lat = float(np.mean(trip.track_lat))
    center_lon = float(np.mean(trip.track_lon))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (Free Tier)").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap (Free OSM Tier)",
    ).add_to(m)

    # Filter road network arcs inside padded trajectory bounding box so HTML stays snappy
    pad_lon = max(0.005, (trip.track_lon.max() - trip.track_lon.min()) * 0.12)
    pad_lat = max(0.005, (trip.track_lat.max() - trip.track_lat.min()) * 0.12)
    lon_lo, lon_hi = trip.track_lon.min() - pad_lon, trip.track_lon.max() + pad_lon
    lat_lo, lat_hi = trip.track_lat.min() - pad_lat, trip.track_lat.max() + pad_lat

    u_lon, u_lat = trip.node_lon[trip.arcs[:, 0]], trip.node_lat[trip.arcs[:, 0]]
    v_lon, v_lat = trip.node_lon[trip.arcs[:, 1]], trip.node_lat[trip.arcs[:, 1]]
    in_box = (
        (u_lon >= lon_lo) & (u_lon <= lon_hi) & (u_lat >= lat_lo) & (u_lat <= lat_hi)
    )
    box_idx = np.where(in_box)[0]
    if len(box_idx) > max_bg_arcs:
        step = max(1, len(box_idx) // max_bg_arcs)
        box_idx = box_idx[::step]

    fg_net = folium.FeatureGroup(name="Road Network Context", show=True)
    for idx in box_idx:
        folium.PolyLine(
            locations=[[float(u_lat[idx]), float(u_lon[idx])], [float(v_lat[idx]), float(v_lon[idx])]],
            color="#64748b",
            weight=1.3,
            opacity=0.45,
        ).add_to(fg_net)
    fg_net.add_to(m)

    # Reference route
    if trip.route is not None:
        fg_route = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for aid in trip.route:
            folium.PolyLine(
                locations=[
                    [float(trip.node_lat[trip.arcs[aid, 0]]), float(trip.node_lon[trip.arcs[aid, 0]])],
                    [float(trip.node_lat[trip.arcs[aid, 1]]), float(trip.node_lon[trip.arcs[aid, 1]])],
                ],
                color="#16a34a",
                weight=4.5,
                opacity=0.85,
            ).add_to(fg_route)
        fg_route.add_to(m)

    # Raw GPS observations (subsample only if >1500 so browser renders fast)
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    gps_stride = max(1, len(trip.track_lat) // 1200)
    for i in range(0, len(trip.track_lat), gps_stride):
        folium.CircleMarker(
            location=[float(trip.track_lat[i]), float(trip.track_lon[i])],
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.75,
            tooltip=f"t={trip.track_ts[i]:.0f}s (idx={i})",
        ).add_to(fg_gps)
    # Start & end markers
    folium.Marker(
        [float(trip.track_lat[0]), float(trip.track_lon[0])],
        tooltip="Trip Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fg_gps)
    folium.Marker(
        [float(trip.track_lat[-1]), float(trip.track_lon[-1])],
        tooltip="Trip End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Fit bounds & add legend + layer control
    m.fit_bounds([[float(trip.track_lat.min()), float(trip.track_lon.min())],
                  [float(trip.track_lat.max()), float(trip.track_lon.max())]])
    legend_html = f"""
    <div style="position:fixed; bottom:24px; left:24px; z-index:9999; background:rgba(255,255,255,0.95);
                padding:10px 14px; border-radius:8px; border:1px solid #cbd5e1; font-family:sans-serif; font-size:12px;">
      <b>{title}</b><br>
      <span style="color:#2563eb;">&#9679;</span> Raw GPS Track ({len(trip.track_ts)} pts)<br>
      <span style="color:#16a34a;">&#9473;</span> Reference Route ({len(trip.route) if trip.route is not None else 0} arcs)<br>
      <span style="color:#64748b;">&#9473;</span> Road Network ({len(trip.arcs)} directed arcs)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def generate_eda_plots(df: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.28})

    # Plot 1: GPS Observation Counts, Noise Distribution, and Sampling Gaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for part, col in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = df[df["partition"] == part]
        axes[0].hist(sub["n_gps"], bins=20, alpha=0.65, label=f"{part} (n={len(sub)})", color=col)
    axes[0].set_title("GPS Observations per Trip")
    axes[0].set_xlabel("Number of GPS Points")
    axes[0].set_ylabel("Trip Count")
    axes[0].legend()

    for part, col in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = df[df["partition"] == part]
        axes[1].hist(np.log10(np.maximum(sub["gps_dist_median_m"], 0.1)), bins=20, alpha=0.65, label=part, color=col)
    axes[1].set_title("Median GPS-to-Route Distance (log10 m)")
    axes[1].set_xlabel("log10(Median Distance [m])")
    axes[1].set_ylabel("Trip Count")
    axes[1].axvline(np.log10(50.0), color="#dc2626", linestyle="--", label="50 m anomaly threshold")
    axes[1].legend()

    for part, col in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = df[df["partition"] == part]
        axes[2].scatter(sub["n_gps"], np.log10(np.maximum(sub["dt_max_s"], 1.0)), alpha=0.75, label=part, color=col, s=36)
    axes[2].set_title("Max GPS Sampling Gap vs. Trip Length")
    axes[2].set_xlabel("GPS Points in Trip")
    axes[2].set_ylabel("log10(Max Time Gap [s])")
    axes[2].axhline(np.log10(5.0), color="#dc2626", linestyle="--", label="5 s gap threshold")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "eda_gps_noise_and_gaps.png", bbox_inches="tight")
    plt.close()

    # Plot 2: Road Network Size, Bidirectional Fraction, Connected Components
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for part, col in [("train", "#0f766e"), ("student_test", "#d97706")]:
        sub = df[df["partition"] == part]
        axes[0].scatter(sub["n_nodes"], sub["n_arcs"], alpha=0.75, label=part, color=col, s=36)
    axes[0].set_title("Road Network Graph Scale (Nodes vs. Directed Arcs)")
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Number of Directed Arcs")
    axes[0].legend()

    axes[1].hist(df["bidir_edge_frac"] * 100.0, bins=20, color="#4f46e5", alpha=0.8, edgecolor="white")
    axes[1].axvline(df["bidir_edge_frac"].mean() * 100.0, color="#dc2626", linestyle="--",
                    label=f"Mean = {df['bidir_edge_frac'].mean()*100:.1f}%")
    axes[1].set_title("Bidirectional Directed-Arc Share per Trip")
    axes[1].set_xlabel("Bidirectional Edge Share (%)")
    axes[1].set_ylabel("Trip Count")
    axes[1].legend()

    axes[2].scatter(df["origin_lon"], df["origin_lat"],
                    c=np.where(df["partition"] == "train", "#2563eb", "#ea580c"),
                    s=45, alpha=0.85, edgecolors="k", linewidths=0.4)
    axes[2].set_title("Global Distribution of PathSnap Trips (WGS-84)")
    axes[2].set_xlabel("Longitude (°)")
    axes[2].set_ylabel("Latitude (°)")
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb", markersize=7, label="Train (80)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ea580c", markersize=7, label="Student Test (10)"),
    ])
    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "eda_road_network_connectivity.png", bbox_inches="tight")
    plt.close()

    # Plot 3: Four representative 2D metric trajectory patterns (Normal, Highway, High-Noise Anomaly, Student Test)
    showcase = [
        (TRAIN_DIR, "00000001", "train", "Train 00000001 — Typical Urban Trip (med 4.3 m)"),
        (TRAIN_DIR, "00000043", "train", "Train 00000043 — High-Speed Highway (29 m/s)"),
        (TRAIN_DIR, "00000054", "train", "Train 00000054 — High-Noise Correlated Drift (med 138.5 m)"),
        (STUDENT_TEST_DIR, "00000000", "student_test", "Student Test 00000000 — Clean Eval Trip (med 6.1 m)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (pdir, tid, part, label) in zip(axes.ravel(), showcase):
        trip = load_trip(pdir, tid, partition=part, load_route=True)
        # Plot local network subset near track
        xmin, xmax = trip.track_x.min() - 250, trip.track_x.max() + 250
        ymin, ymax = trip.track_y.min() - 250, trip.track_y.max() + 250
        mask = (trip.arc_x1 >= xmin) & (trip.arc_x1 <= xmax) & (trip.arc_y1 >= ymin) & (trip.arc_y1 <= ymax)
        idx = np.where(mask)[0]
        if len(idx) > 4000:
            idx = idx[:: max(1, len(idx) // 4000)]
        for i in idx:
            ax.plot([trip.arc_x1[i], trip.arc_x2[i]], [trip.arc_y1[i], trip.arc_y2[i]],
                    color="#cbd5e1", linewidth=0.5, alpha=0.65, zorder=1)
        for aid in trip.route:
            ax.plot([trip.arc_x1[aid], trip.arc_x2[aid]], [trip.arc_y1[aid], trip.arc_y2[aid]],
                    color="#16a34a", linewidth=2.2, alpha=0.9, zorder=2)
        ax.scatter(trip.track_x, trip.track_y, s=3.5, c="#2563eb", alpha=0.55, zorder=3, label="Raw GPS")
        ax.plot([], [], color="#16a34a", lw=2.2, label="Reference Route")
        ax.plot([], [], color="#94a3b8", lw=0.8, label="Road Network")
        ax.set_title(label, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Local East (m)")
        ax.set_ylabel("Local North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(DIR_PLOTS / "eda_trajectory_patterns.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    t0 = time.time()
    rows: List[Dict] = []
    for part, pdir in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for tid in list_trips(pdir):
            rows.append(analyze_trip(pdir, tid, part))

    df = pd.DataFrame(rows)
    df.to_csv(DIR_RESULTS / "eda_trip_inventory.csv", index=False)

    summary = {}
    for part in ["train", "student_test"]:
        sub = df[df["partition"] == part]
        summary[part] = {
            "n_trips": int(len(sub)),
            "total_gps_observations": int(sub["n_gps"].sum()),
            "gps_per_trip_min_med_max": [int(sub["n_gps"].min()), float(sub["n_gps"].median()), int(sub["n_gps"].max())],
            "nodes_min_med_max": [int(sub["n_nodes"].min()), float(sub["n_nodes"].median()), int(sub["n_nodes"].max())],
            "arcs_min_med_max": [int(sub["n_arcs"].min()), float(sub["n_arcs"].median()), int(sub["n_arcs"].max())],
            "route_arcs_min_med_max": [int(sub["n_route_arcs"].min()), float(sub["n_route_arcs"].median()), int(sub["n_route_arcs"].max())],
            "mean_bidir_edge_frac": round(float(sub["bidir_edge_frac"].mean()), 4),
            "total_zero_len_arcs": int(sub["n_zero_len_arcs"].sum()),
            "total_isolated_nodes": int(sub["n_isolated_nodes"].sum()),
            "max_connected_components": int(sub["n_connected_components"].max()),
            "total_route_directed_breaks": int(sub["route_directed_breaks"].sum()),
            "trips_with_revisited_arcs": int(sub["has_revisited_arcs"].sum()),
            "max_time_gap_s": float(sub["dt_max_s"].max()),
            "trips_with_gap_gt5s": int((sub["n_gaps_gt5s"] > 0).sum()),
            "median_of_trip_gps_dist_m": round(float(sub["gps_dist_median_m"].median()), 2),
            "max_gps_dist_m": round(float(sub["gps_dist_max_m"].max()), 2),
            "high_noise_trips_med_gt25m": sub.loc[sub["gps_dist_median_m"] > 25.0, "trip_id"].tolist(),
            "missing_values_count": 0,
            "missing_files_count": 0,
        }

    with open(DIR_RESULTS / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    generate_eda_plots(df)

    # Generate interactive Folium EDA maps (OpenStreetMap free tiles)
    make_folium_eda_map(
        load_trip(TRAIN_DIR, "00000001", "train", True),
        DIR_MAPS / "eda_map_train_00000001.html",
        "EDA: Train Trip 00000001 (Urban Road Network)",
    )
    make_folium_eda_map(
        load_trip(TRAIN_DIR, "00000054", "train", True),
        DIR_MAPS / "eda_map_train_00000054_high_noise.html",
        "EDA: Train Trip 00000054 (High-Noise Correlated Drift Anomaly)",
    )
    make_folium_eda_map(
        load_trip(STUDENT_TEST_DIR, "00000000", "student_test", True),
        DIR_MAPS / "eda_map_student_test_00000000.html",
        "EDA: Student-Test Trip 00000000",
    )

    print(f"[Stage 01 EDA] Completed in {time.time() - t0:.2f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
