"""Configuration loading (TOML)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .residual import Intake

DEFAULT_CONFIG = Path(__file__).parent / "config" / "stations.toml"


@dataclass
class AlarmCfg:
    rise_pct_per_h: float = 25.0
    rise_window_min: int = 60
    elevated_factor: float = 3.0
    baseline_hours: int = 48
    hysteresis: float = 0.05
    min_samples: int = 2


@dataclass
class Config:
    station_ids: list[str]
    intakes: dict[str, Intake] = field(default_factory=dict)
    latency_min: int = 30
    stuck_samples: int = 24
    spike_k: float = 8.0
    alarms: AlarmCfg = field(default_factory=AlarmCfg)
    horizons_min: list[int] = field(default_factory=lambda: [60, 180])
    train_frac: float = 0.6


def load_config(path: Path | str | None = None) -> Config:
    raw = tomllib.loads(Path(path or DEFAULT_CONFIG).read_text("utf-8"))
    stations = raw.get("stations", [])
    if not stations:
        raise ValueError("config has no [[stations]]")
    ids = [str(s["id"]) for s in stations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate station id in config")
    intakes = {}
    for s in stations:
        if "intake" in s:
            i = s["intake"]
            intakes[str(s["id"])] = Intake(float(i["min_flow_m3s"]), float(i["capacity_m3s"]), int(i.get("min_duration_min", 15)))
    qa, al, fc = raw.get("qa", {}), raw.get("alarms", {}), raw.get("forecast", {})
    return Config(
        station_ids=ids,
        intakes=intakes,
        latency_min=int(qa.get("latency_min", 30)),
        stuck_samples=int(qa.get("stuck_samples", 24)),
        spike_k=float(qa.get("spike_k", 8.0)),
        alarms=AlarmCfg(**{k: v for k, v in al.items() if k in AlarmCfg.__dataclass_fields__}),
        horizons_min=[int(h) for h in fc.get("horizons_min", [60, 180])],
        train_frac=float(fc.get("train_frac", 0.6)),
    )
