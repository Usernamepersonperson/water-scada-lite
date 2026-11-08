"""Residual-flow (Restwasser) compliance check for a hypothetical small-hydro intake.

Model: the gauge discharge Q is taken as the natural flow arriving at the intake. The intake may
divert at most ``capacity`` and must leave at least ``min_flow`` in the river:

    allowed_diversion = clip(Q - min_flow, 0, capacity)

Two outputs are reported:
* ``closed``: minutes in which Q < min_flow, i.e. the intake must be fully closed;
* ``breach_if_unregulated``: minutes in which an intake that always diverts ``capacity`` would leave
  less than ``min_flow`` (Q - capacity < min_flow).
Violation episodes need at least ``min_duration_min`` of consecutive non-compliant, QA-clean samples.
The legal minimum flow of a real scheme is set per concession; the value here is configurable input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Intake:
    min_flow: float  # m3/s
    capacity: float  # m3/s
    min_duration_min: int = 15

    def __post_init__(self) -> None:
        if self.min_flow < 0 or self.capacity <= 0:
            raise ValueError("min_flow must be >= 0 and capacity > 0")


def allowed_diversion(q: pd.Series, intake: Intake) -> pd.Series:
    return (q - intake.min_flow).clip(lower=0, upper=intake.capacity)


def _episodes(mask: pd.Series, step_min: float, min_duration_min: int) -> list[dict]:
    eps, start, prev = [], None, None
    for ts, v in mask.items():
        if v and start is None:
            start = ts
        if not v and start is not None:
            eps.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        eps.append((start, prev))
    out = []
    for s, e in eps:
        dur = (e - s).total_seconds() / 60 + step_min
        if dur >= min_duration_min:
            out.append({"start": s.isoformat(), "end": e.isoformat(), "duration_min": dur})
    return out


def check_residual_flow(q: pd.Series, intake: Intake) -> dict:
    """``q``: regular-grid discharge with NaN for QA-flagged/missing samples."""
    step_min = pd.Series(q.index).diff().median().total_seconds() / 60
    valid = q.notna()
    below = (q < intake.min_flow) & valid
    unreg = ((q - intake.capacity) < intake.min_flow) & valid
    eps_closed = _episodes(below, step_min, intake.min_duration_min)
    eps_unreg = _episodes(unreg, step_min, intake.min_duration_min)
    n_valid = int(valid.sum())
    div = allowed_diversion(q, intake)
    unreg_volume = float(np.nansum(np.minimum(q.fillna(0), intake.capacity)) * step_min * 60)
    reg_volume = float(np.nansum(div) * step_min * 60)
    return {
        "min_flow_m3s": intake.min_flow,
        "capacity_m3s": intake.capacity,
        "valid_minutes": n_valid * step_min,
        "closed_minutes": float(below.sum() * step_min),
        "closed_pct": round(float(100.0 * below.sum() / n_valid), 2) if n_valid else None,
        "closed_episodes": eps_closed,
        "breach_if_unregulated_minutes": float(unreg.sum() * step_min),
        "breach_if_unregulated_pct": round(float(100.0 * unreg.sum() / n_valid), 2) if n_valid else None,
        "breach_episodes": eps_unreg,
        "diverted_volume_regulated_m3": round(reg_volume),
        "diverted_volume_unregulated_m3": round(unreg_volume),
        "compliant": len(eps_unreg) == 0,
    }
