"""PathSnap data loading and local metric projection.

Raw files per trip (headerless, whitespace separated):
  <id>.track : lon lat t      (GPS observations, t in seconds)
  <id>.nodes : lon lat        (row index = node id)
  <id>.arcs  : from_node to_node  (row index = arc id, directed)
  <id>.route : arc_id         (reference route, chronological; train / student_test only)

Every trip has its OWN road network, so node ids and arc ids are only valid inside one trip.

Distances are computed in a per-trip Azimuthal Equidistant (AEQD) projection centred on the
trip's mean GPS position. Within a city-sized extent AEQD distances are accurate to well under
1 m, and it is a plain metric (metre) plane, so Euclidean maths is valid.

Data root: --data argument, else env var PATHSNAP_DATA, else <repo>/../Student folder
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "student_test", "evaluator_test")


def data_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("PATHSNAP_DATA")
    if env:
        return Path(env)
    return REPO_ROOT.parent / "Student folder"


def trip_dir(partition: str, trip_id: int, root: Optional[Path] = None) -> Path:
    root = Path(root) if root else data_root()
    return root / partition / f"{int(trip_id):08d}"


def list_trips(partition: str, root: Optional[Path] = None) -> list:
    root = Path(root) if root else data_root()
    pdir = root / partition
    if not pdir.is_dir():
        raise FileNotFoundError(
            f"Partition folder not found: {pdir}\n"
            "Put 'Student folder' next to the repo folder, or set PATHSNAP_DATA / pass --data."
        )
    return sorted(int(p.name) for p in pdir.iterdir() if p.is_dir() and p.name.isdigit())


def _read_table(path: Path, dtype) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


@dataclass
class Trip:
    trip_id: int
    partition: str
    lon: np.ndarray          # GPS longitude
    lat: np.ndarray
    t: np.ndarray            # seconds
    x: np.ndarray            # GPS projected (m)
    y: np.ndarray
    node_x: np.ndarray       # nodes projected (m)
    node_y: np.ndarray
    arc_from: np.ndarray     # int64, node ids
    arc_to: np.ndarray
    route: Optional[np.ndarray]   # reference route arc ids, or None
    center_lon: float
    center_lat: float

    @property
    def n_obs(self) -> int:
        return len(self.t)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_from)


def load_trip(partition: str, trip_id: int, root: Optional[Path] = None,
              with_route: bool = True) -> Trip:
    d = trip_dir(partition, trip_id, root)
    stem = f"{int(trip_id):08d}"
    trk = _read_table(d / f"{stem}.track", np.float64)
    nodes = _read_table(d / f"{stem}.nodes", np.float64)
    arcs = _read_table(d / f"{stem}.arcs", np.int64)
    route = None
    rpath = d / f"{stem}.route"
    if with_route and rpath.exists():
        route = _read_table(rpath, np.int64).reshape(-1)

    lon, lat, t = trk[:, 0], trk[:, 1], trk[:, 2]
    if len(t) == 0:
        raise ValueError(f"trip {trip_id}: empty track")
    if not (np.isfinite(trk).all() and np.isfinite(nodes).all()):
        raise ValueError(f"trip {trip_id}: non-finite values in raw data")
    if np.any(np.diff(t) < 0):
        raise ValueError(f"trip {trip_id}: timestamps are not sorted")
    n_nodes = len(nodes)
    if arcs.min() < 0 or arcs.max() >= n_nodes:
        raise ValueError(f"trip {trip_id}: arc node index out of range")
    if route is not None and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"trip {trip_id}: route arc id out of range")

    c_lon, c_lat = float(lon.mean()), float(lat.mean())
    crs = (f"+proj=aeqd +lat_0={c_lat:.8f} +lon_0={c_lon:.8f} "
           "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
    return Trip(
        trip_id=int(trip_id), partition=partition,
        lon=lon, lat=lat, t=t,
        x=np.asarray(x), y=np.asarray(y),
        node_x=np.asarray(nx), node_y=np.asarray(ny),
        arc_from=arcs[:, 0], arc_to=arcs[:, 1], route=route,
        center_lon=c_lon, center_lat=c_lat,
    )


def load_arcs_only(partition: str, trip_id: int, root: Optional[Path] = None):
    """Light loader used by evaluation (graph connectivity only)."""
    d = trip_dir(partition, trip_id, root)
    arcs = _read_table(d / f"{int(trip_id):08d}.arcs", np.int64)
    return arcs[:, 0].astype(np.int32), arcs[:, 1].astype(np.int32)


def load_route_only(partition: str, trip_id: int, root: Optional[Path] = None) -> np.ndarray:
    d = trip_dir(partition, trip_id, root)
    return _read_table(d / f"{int(trip_id):08d}.route", np.int64).reshape(-1)
