"""Raw trip I/O, validation, local AEQD metric projection, and graph structures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pyproj import CRS, Transformer


@dataclass
class TripData:
    trip_id: str
    partition: str
    # Raw geographic coordinates
    track_lon: np.ndarray        # (T,)
    track_lat: np.ndarray        # (T,)
    track_ts: np.ndarray         # (T,) seconds
    node_lon: np.ndarray         # (N_nodes,)
    node_lat: np.ndarray         # (N_nodes,)
    arcs: np.ndarray             # (N_arcs, 2) int32 [from_node_id, to_node_id]
    route: Optional[np.ndarray]  # (M,) int32 arc_ids, or None in strict inference mode
    # Projected local East-North-Up (AEQD) coordinates in meters
    origin_lon: float
    origin_lat: float
    track_x: np.ndarray          # (T,) meters
    track_y: np.ndarray          # (T,) meters
    node_x: np.ndarray           # (N_nodes,) meters
    node_y: np.ndarray           # (N_nodes,) meters
    # Precomputed arc geometry in meters
    arc_x1: np.ndarray           # (N_arcs,)
    arc_y1: np.ndarray           # (N_arcs,)
    arc_x2: np.ndarray           # (N_arcs,)
    arc_y2: np.ndarray           # (N_arcs,)
    arc_len: np.ndarray          # (N_arcs,) meters
    arc_ux: np.ndarray           # (N_arcs,) unit direction x
    arc_uy: np.ndarray           # (N_arcs,) unit direction y
    arc_bearing: np.ndarray      # (N_arcs,) radians in [-pi, pi]


def list_trips(partition_dir: Path) -> List[str]:
    """Return sorted list of valid trip IDs inside a partition directory."""
    if not partition_dir.exists():
        return []
    trips = []
    for p in sorted(partition_dir.iterdir()):
        if p.is_dir() and (p / f"{p.name}.track").exists():
            trips.append(p.name)
    return trips


def build_aeqd_projector(lon0: float, lat0: float) -> Transformer:
    """Build a WGS-84 Azimuthal Equidistant (AEQD) local metric projector centered at (lon0, lat0)."""
    crs_wgs84 = CRS.from_epsg(4326)
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)


def load_trip(
    partition_dir: Path,
    trip_id: str,
    partition: str = "unknown",
    load_route: bool = True,
) -> TripData:
    """Load raw trip files, validate integrity, and project to local AEQD metric coordinates.

    Parameters
    ----------
    load_route : bool
        Set to False in prediction pipelines to guarantee strict isolation from ground truth.
    """
    td = partition_dir / trip_id
    track_path = td / f"{trip_id}.track"
    nodes_path = td / f"{trip_id}.nodes"
    arcs_path = td / f"{trip_id}.arcs"
    route_path = td / f"{trip_id}.route"

    track = np.loadtxt(track_path, dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)
    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    arcs = np.loadtxt(arcs_path, dtype=np.int32)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route = None
    if load_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int32)
        if route.ndim == 0:
            route = route.reshape(1)

    # Sort track chronologically if ever unsorted (defensive check)
    if len(track) > 1 and np.any(np.diff(track[:, 2]) < 0):
        order = np.argsort(track[:, 2], kind="stable")
        track = track[order]

    # Filter any NaN / non-finite GPS rows defensively
    finite_mask = np.all(np.isfinite(track), axis=1)
    if not np.all(finite_mask):
        track = track[finite_mask]

    track_lon = track[:, 0]
    track_lat = track[:, 1]
    track_ts = track[:, 2]
    node_lon = nodes[:, 0]
    node_lat = nodes[:, 1]

    # Center local AEQD projection on road network centroid
    origin_lon = float(np.mean(node_lon))
    origin_lat = float(np.mean(node_lat))
    transformer = build_aeqd_projector(origin_lon, origin_lat)

    track_x, track_y = transformer.transform(track_lon, track_lat)
    node_x, node_y = transformer.transform(node_lon, node_lat)
    track_x = np.asarray(track_x, dtype=np.float64)
    track_y = np.asarray(track_y, dtype=np.float64)
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)

    arc_x1 = node_x[arcs[:, 0]]
    arc_y1 = node_y[arcs[:, 0]]
    arc_x2 = node_x[arcs[:, 1]]
    arc_y2 = node_y[arcs[:, 1]]
    dx = arc_x2 - arc_x1
    dy = arc_y2 - arc_y1
    arc_len = np.hypot(dx, dy)

    # Guard against zero-length arcs (100 in train, 6 in student_test)
    safe_len = np.where(arc_len > 1e-6, arc_len, 1.0)
    arc_ux = np.where(arc_len > 1e-6, dx / safe_len, 0.0)
    arc_uy = np.where(arc_len > 1e-6, dy / safe_len, 0.0)
    arc_bearing = np.arctan2(dy, dx)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track_lon=track_lon,
        track_lat=track_lat,
        track_ts=track_ts,
        node_lon=node_lon,
        node_lat=node_lat,
        arcs=arcs,
        route=route,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        track_x=track_x,
        track_y=track_y,
        node_x=node_x,
        node_y=node_y,
        arc_x1=arc_x1,
        arc_y1=arc_y1,
        arc_x2=arc_x2,
        arc_y2=arc_y2,
        arc_len=arc_len,
        arc_ux=arc_ux,
        arc_uy=arc_uy,
        arc_bearing=arc_bearing,
    )


def load_route_only(partition_dir: Path, trip_id: str) -> Optional[np.ndarray]:
    """Load ONLY the ground-truth .route file (used exclusively by server-side evaluator)."""
    route_path = partition_dir / trip_id / f"{trip_id}.route"
    if not route_path.exists():
        return None
    route = np.loadtxt(route_path, dtype=np.int32)
    if route.ndim == 0:
        route = route.reshape(1)
    return route


def build_adjacency(trip: TripData) -> Tuple[List[List[Tuple[int, int, float]]], np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    """Build directed adjacency list: adj[u] = [(v, arc_id, length_m), ...], plus degrees and reverse-edge lookup."""
    n_nodes = len(trip.node_x)
    adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(n_nodes)]
    out_deg = np.zeros(n_nodes, dtype=np.int32)
    in_deg = np.zeros(n_nodes, dtype=np.int32)
    edge_map: Dict[Tuple[int, int], int] = {}

    u_arr = trip.arcs[:, 0]
    v_arr = trip.arcs[:, 1]
    lens = trip.arc_len
    for arc_id in range(len(u_arr)):
        u = int(u_arr[arc_id])
        v = int(v_arr[arc_id])
        w = float(lens[arc_id])
        adj[u].append((v, arc_id, w))
        out_deg[u] += 1
        in_deg[v] += 1
        edge_map[(u, v)] = arc_id

    return adj, out_deg, in_deg, edge_map
