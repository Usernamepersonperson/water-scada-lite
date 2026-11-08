"""End-to-end run: QA -> alarms -> flood comparison -> residual flow -> forecast."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import forecast as fc
from .alarms import AlarmEngine, Rule, compare_flood_levels
from .foen import StationData
from .qa import QAConfig, clean, qa_flags, summarise
from .residual import check_residual_flow
from .settings import Config

SEVERITY_BY_LEVEL = {2: "warning", 3: "critical", 4: "critical", 5: "critical"}


@dataclass
class StationResult:
    data: StationData
    qa_q: pd.DataFrame
    qa_h: pd.DataFrame
    rules: list[Rule]
    flood: dict
    residual: dict | None
    forecast: list[dict]
    baseline_q: float | None


@dataclass
class Result:
    stations: dict[str, StationResult] = field(default_factory=dict)
    engine: AlarmEngine = field(default_factory=AlarmEngine)
    period: tuple[str, str] | None = None
    fetched_at: str | None = None

    def summary(self) -> dict:
        """JSON-serialisable measured results."""
        st = {}
        for sid, r in self.stations.items():
            alarms = [a for a in self.engine.alarms if a.station == sid]
            by_rule: dict[str, int] = {}
            for a in alarms:
                by_rule[a.rule] = by_rule.get(a.rule, 0) + 1
            st[sid] = {
                "name": r.data.name,
                "qa_discharge": summarise(r.qa_q),
                "qa_level": summarise(r.qa_h),
                "alarms_by_rule": by_rule,
                "alarms_total": len(alarms),
                "flood": r.flood,
                "residual": r.residual,
                "forecast": r.forecast,
            }
        return {
            "period_utc": self.period,
            "fetched_at": self.fetched_at,
            "alarms_total": len(self.engine.alarms),
            "alarms_by_state": _count(a.state for a in self.engine.alarms),
            "stations": st,
        }


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def build_rules(sd: StationData, cfg: Config, baseline_q: float | None) -> list[Rule]:
    a = cfg.alarms
    rules = [
        Rule(
            "rise_rate", "rise", a.rise_pct_per_h, "warning",
            window_min=a.rise_window_min, min_samples=a.min_samples, unit="%/h",
        )
    ]
    if baseline_q is not None and baseline_q > 0:
        rules.append(
            Rule("elevated_flow", "high", a.elevated_factor * baseline_q, "warning",
                 hysteresis=a.hysteresis, min_samples=a.min_samples)
        )
    for b in sd.bands:
        rules.append(
            Rule(f"official_level_{b.level}", "high", b.lower, SEVERITY_BY_LEVEL.get(b.level, "critical"),
                 hysteresis=a.hysteresis, min_samples=a.min_samples)
        )
    return rules


def run(stations: list[StationData], cfg: Config) -> Result:
    res = Result()
    starts, ends = [], []
    for sd in stations:
        end = None
        if sd.fetched_at is not None:
            end = sd.fetched_at - pd.Timedelta(minutes=cfg.latency_min)
        qcfg = QAConfig(stuck_samples=cfg.stuck_samples, spike_k=cfg.spike_k)
        if not sd.frame["discharge_m3s"].notna().any():
            continue  # alarms and compliance are discharge based
        qa_q = qa_flags(sd.frame["discharge_m3s"], qcfg, end)
        hcfg = QAConfig(stuck_samples=cfg.stuck_samples, spike_k=cfg.spike_k, min_value=-1e9)
        qa_h = qa_flags(sd.frame["level_m"], hcfg, end)
        q = clean(qa_q)
        h = clean(qa_h)
        step = qa_q.attrs["step_min"]
        n_base = int(cfg.alarms.baseline_hours * 60 / step)
        base = q.iloc[:n_base].median()
        baseline = None if pd.isna(base) else float(base)
        rules = build_rules(sd, cfg, baseline)
        res.engine.process_series(sd.station_id, q, rules)
        flood = compare_flood_levels(sd.station_id, q, {b.level: b.lower for b in sd.bands}).to_dict()
        residual = None
        if sd.station_id in cfg.intakes:
            residual = check_residual_flow(q, cfg.intakes[sd.station_id])
        fcs = []
        if h.notna().sum() > 200:
            for hm in cfg.horizons_min:
                steps = max(1, round(hm / step))
                try:
                    r = fc.evaluate(h, q, steps, cfg.train_frac)
                except ValueError:
                    continue
                r["horizon_min"] = hm
                fcs.append(r)
        res.stations[sd.station_id] = StationResult(sd, qa_q, qa_h, rules, flood, residual, fcs, baseline)
        starts.append(qa_q.index[0])
        ends.append(qa_q.index[-1])
    if starts:
        res.period = (min(starts).isoformat(), max(ends).isoformat())
    fa = [s.fetched_at for s in stations if s.fetched_at is not None]
    res.fetched_at = max(fa).isoformat() if fa else None
    return res


def simulate_acks(result: Result, by: str = "demo-operator (simulated)") -> None:
    """Demo only: acknowledge every other alarm so both acknowledged and open states are visible."""
    for a in result.engine.alarms:
        if a.id % 2 == 0:
            result.engine.ack(a.id, by, a.t_raised + pd.Timedelta(minutes=10))
