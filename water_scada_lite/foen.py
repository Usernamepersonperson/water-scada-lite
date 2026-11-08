"""Ingestion of FOEN/BAFU hydrological station data (level + discharge + official flood bands).

Source: the public FOEN hydrology portal publishes, per station, a JSON document that backs its
"water level / discharge, last 7 days" chart. It holds the raw time series (5 or 10 min) and the
official flood-danger bands drawn on the discharge axis. Source: FOEN/BAFU, https://www.hydrodaten.admin.ch
The endpoint is a portal endpoint, not a versioned API; the parser is therefore strict and tested
against a cached snapshot (``fixtures/``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

PLOT_URL = "https://www.hydrodaten.admin.ch/plots/p_q_7days/{sid}_p_q_7days_de.json"

# Colour used by the portal for each official danger level (level 1 = no band drawn).
_COLOR_TO_LEVEL = {"#FFFF00": 2, "#FF9900": 3, "#F7001D": 4, "#800000": 5}


class ParseError(ValueError):
    """The portal document does not have the expected structure."""


@dataclass
class FloodBand:
    level: int  # official danger level 2..5
    lower: float  # discharge in m3/s at which the level starts
    upper: float


@dataclass
class StationData:
    station_id: str
    name: str
    frame: pd.DataFrame  # UTC index; columns level_m, discharge_m3s (NaN = missing)
    bands: list[FloodBand] = field(default_factory=list)
    fetched_at: pd.Timestamp | None = None

    def threshold(self, level: int) -> float | None:
        for b in self.bands:
            if b.level == level:
                return b.lower
        return None


def parse_plot(doc: dict, station_id: str) -> StationData:
    """Parse a portal plot document into a :class:`StationData`."""
    try:
        plot = doc["plot"]
        traces = plot["data"]
        layout = plot["layout"]
    except (KeyError, TypeError) as exc:
        raise ParseError(f"station {station_id}: missing plot/data/layout") from exc
    cols: dict[str, pd.Series] = {}
    axis_of: dict[str, str] = {}
    for tr in traces:
        name = tr.get("name")
        key = {"Wasserstand": "level_m", "Abfluss": "discharge_m3s"}.get(name)
        if key is None:
            continue
        xs, ys = tr.get("x"), tr.get("y")
        if xs is None or ys is None or len(xs) != len(ys):
            raise ParseError(f"station {station_id}: trace {name} has mismatched x/y")
        idx = pd.to_datetime(pd.Series(xs), utc=True, errors="coerce", format="ISO8601")
        if idx.isna().any():
            raise ParseError(f"station {station_id}: unparseable timestamp in {name}")
        vals = pd.to_numeric(pd.Series(ys, dtype="object"), errors="coerce")
        cols[key] = pd.Series(vals.to_numpy(float), index=pd.DatetimeIndex(idx))
        axis_of[key] = tr.get("yaxis", "y")
    if "discharge_m3s" not in cols and "level_m" not in cols:
        raise ParseError(f"station {station_id}: no level or discharge trace")
    frame = pd.DataFrame(cols).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for c in ("level_m", "discharge_m3s"):
        if c not in frame:
            frame[c] = float("nan")
    frame = frame[["level_m", "discharge_m3s"]]
    frame.index.name = "ts"

    q_axis = axis_of.get("discharge_m3s", "y2").replace("y", "y")  # e.g. "y2"
    bands: dict[int, FloodBand] = {}
    for sh in layout.get("shapes", []):
        if sh.get("type") != "rect" or sh.get("yref") != q_axis:
            continue
        lvl = _COLOR_TO_LEVEL.get(str(sh.get("fillcolor", "")).upper())
        if lvl is None:
            continue
        bands[lvl] = FloodBand(lvl, float(sh["y0"]), float(sh["y1"]))
    title = str(layout.get("title", station_id))
    name = re.sub(r"\s+PQPlot.*$", "", title).strip()
    return StationData(station_id, name, frame, sorted(bands.values(), key=lambda b: b.level))


def fetch_station(station_id: str, client: httpx.Client | None = None, timeout: float = 30.0) -> StationData:
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        r = client.get(PLOT_URL.format(sid=station_id))
        r.raise_for_status()
        sd = parse_plot(r.json(), station_id)
        sd.fetched_at = pd.Timestamp.now(tz="UTC").floor("s")
        return sd
    finally:
        if own:
            client.close()


def save_snapshot(stations: list[StationData], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {}
    for s in stations:
        s.frame.to_csv(directory / f"{s.station_id}.csv", float_format="%.4f")
        meta[s.station_id] = {
            "name": s.name,
            "bands": [vars(b) for b in s.bands],
            "fetched_at": s.fetched_at.isoformat() if s.fetched_at is not None else None,
        }
    (directory / "stations.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), "utf-8")


def load_snapshot(directory: Path, ids: list[str] | None = None) -> list[StationData]:
    meta = json.loads((directory / "stations.json").read_text("utf-8"))
    out = []
    for sid, m in meta.items():
        if ids and sid not in ids:
            continue
        df = pd.read_csv(directory / f"{sid}.csv", index_col="ts", parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
        fa = pd.Timestamp(m["fetched_at"]) if m.get("fetched_at") else None
        out.append(StationData(sid, m["name"], df, [FloodBand(**b) for b in m["bands"]], fa))
    return out
