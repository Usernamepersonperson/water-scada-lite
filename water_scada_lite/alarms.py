"""Threshold and rate-of-rise alarms with hysteresis, debounce and operator acknowledgement.

State machine per alarm instance::

    ACTIVE_UNACK --ack--> ACTIVE_ACK
    ACTIVE_UNACK --condition clears--> CLEARED_UNACK --ack--> CLEARED
    ACTIVE_ACK   --condition clears--> CLEARED

A new breach after a clear opens a new alarm instance. Missing (NaN) samples hold the state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

ACTIVE_UNACK, ACTIVE_ACK, CLEARED_UNACK, CLEARED = "active_unack", "active_ack", "cleared_unack", "cleared"


@dataclass
class Rule:
    name: str
    kind: str  # "high" | "low" | "rise"
    trigger: float  # value (high/low) or percent per hour (rise)
    severity: str = "warning"  # info | warning | critical
    hysteresis: float = 0.05  # high/low: fraction of trigger; rise: clear at trigger*(1-hysteresis*..)
    clear: float | None = None  # explicit clear level; overrides hysteresis
    min_samples: int = 2  # consecutive breaching samples before raising
    window_min: int = 60  # rise: lookback window
    unit: str = "m3/s"

    def __post_init__(self) -> None:
        if self.kind not in ("high", "low", "rise"):
            raise ValueError(f"unknown rule kind {self.kind!r}")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if self.clear is None:
            f = self.hysteresis
            self.clear = {"high": self.trigger * (1 - f), "low": self.trigger * (1 + f), "rise": self.trigger * 0.5}[self.kind]
        ok = {"high": self.clear <= self.trigger, "low": self.clear >= self.trigger, "rise": self.clear <= self.trigger}
        if not ok[self.kind]:
            raise ValueError(f"rule {self.name}: clear level on the wrong side of the trigger")


@dataclass
class Alarm:
    id: int
    station: str
    rule: str
    severity: str
    state: str
    t_first_breach: pd.Timestamp
    t_raised: pd.Timestamp
    t_clear: pd.Timestamp | None = None
    peak: float = float("nan")
    ack_by: str | None = None
    t_ack: pd.Timestamp | None = None

    def to_dict(self) -> dict:
        d = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in vars(self).items()}
        if isinstance(d["peak"], float) and math.isnan(d["peak"]):
            d["peak"] = None
        return d


@dataclass
class _RuleState:
    breaches: int = 0
    first_breach: pd.Timestamp | None = None
    alarm: Alarm | None = None


class AlarmEngine:
    def __init__(self) -> None:
        self.alarms: list[Alarm] = []
        self._rs: dict[tuple[str, str], _RuleState] = {}
        self._next_id = 1

    # -- evaluation ---------------------------------------------------------------------------
    def feed(self, station: str, rule: Rule, ts: pd.Timestamp, metric: float) -> Alarm | None:
        """Feed the rule's metric (value, or % per hour for rise rules). Returns a new alarm if raised."""
        st = self._rs.setdefault((station, rule.name), _RuleState())
        if metric is None or (isinstance(metric, float) and math.isnan(metric)):
            return None  # hold state on missing data
        a = st.alarm
        breach = metric <= rule.trigger if rule.kind == "low" else metric >= rule.trigger
        if a is None:
            if breach:
                if st.breaches == 0:
                    st.first_breach = ts
                st.breaches += 1
                if st.breaches >= rule.min_samples:
                    a = Alarm(self._next_id, station, rule.name, rule.severity, ACTIVE_UNACK, st.first_breach, ts, peak=metric)
                    self._next_id += 1
                    self.alarms.append(a)
                    st.alarm = a
                    st.breaches = 0
                    return a
            else:
                st.breaches = 0
            return None
        # alarm is open: track extreme, check clear
        a.peak = min(a.peak, metric) if rule.kind == "low" else max(a.peak, metric)
        cleared = metric >= rule.clear if rule.kind == "low" else metric <= rule.clear
        if cleared:
            a.t_clear = ts
            a.state = CLEARED if a.state == ACTIVE_ACK else CLEARED_UNACK
            st.alarm = None
            st.breaches = 0
        return None

    def ack(self, alarm_id: int, by: str, at: pd.Timestamp) -> Alarm:
        for a in self.alarms:
            if a.id == alarm_id:
                if a.state == ACTIVE_UNACK:
                    a.state = ACTIVE_ACK
                elif a.state == CLEARED_UNACK:
                    a.state = CLEARED
                elif a.state in (ACTIVE_ACK, CLEARED):
                    return a  # idempotent
                a.ack_by, a.t_ack = by, at
                return a
        raise KeyError(f"no alarm with id {alarm_id}")

    # -- batch --------------------------------------------------------------------------------
    def process_series(self, station: str, x: pd.Series, rules: list[Rule]) -> None:
        """Replay a regular-grid series (NaN = unusable sample) through the rules."""
        step_min = pd.Series(x.index).diff().median().total_seconds() / 60
        for rule in rules:
            if rule.kind == "rise":
                lag = max(1, round(rule.window_min / step_min))
                prev = x.shift(lag)
                metric = (x - prev) / prev.where(prev > 0) * 100.0 * (60.0 / (lag * step_min))
            else:
                metric = x
            for ts, m in metric.items():
                self.feed(station, rule, ts, float(m))

    def open_alarms(self) -> list[Alarm]:
        return [a for a in self.alarms if a.state in (ACTIVE_UNACK, ACTIVE_ACK, CLEARED_UNACK)]

    def for_station(self, station: str) -> list[Alarm]:
        return [a for a in self.alarms if a.station == station]


@dataclass
class FloodComparison:
    station: str
    max_discharge: float | None
    thresholds: dict[int, float] = field(default_factory=dict)
    level_reached: int = 1  # official level 1 = below level-2 threshold
    pct_of_level2: float | None = None

    def to_dict(self) -> dict:
        return {
            "station": self.station,
            "max_discharge_m3s": self.max_discharge,
            "thresholds_m3s": self.thresholds,
            "official_level_reached": self.level_reached,
            "pct_of_level2": self.pct_of_level2,
        }


def compare_flood_levels(station: str, q: pd.Series, thresholds: dict[int, float]) -> FloodComparison:
    """Compare peak discharge with official danger-level thresholds (level 2..5, m3/s)."""
    qmax = float(q.max()) if q.notna().any() else None
    fc = FloodComparison(station, qmax, dict(thresholds))
    if qmax is not None:
        for lvl in sorted(thresholds):
            if qmax >= thresholds[lvl]:
                fc.level_reached = lvl
        if 2 in thresholds and thresholds[2] > 0:
            fc.pct_of_level2 = round(100.0 * qmax / thresholds[2], 1)
    return fc
