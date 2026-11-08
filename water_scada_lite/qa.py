"""Data-quality flags for a regularly sampled sensor series: gap, stuck, spike, range, stale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FLAGS = ("missing", "stuck", "spike", "range")


@dataclass
class QAConfig:
    stuck_samples: int = 24  # identical readings in a row (24 x 5 min = 2 h)
    spike_window: int = 9  # centred window (odd) for the Hampel filter
    spike_k: float = 8.0  # robust sigmas
    spike_rel_floor: float = 0.003  # min scale as a fraction of the local median (sensor resolution)
    min_value: float = 0.0  # discharge cannot be negative
    max_value: float | None = None


def infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing of a timestamp index, rounded to the minute."""
    if len(index) < 2:
        raise ValueError("need at least two samples to infer the sampling step")
    step = pd.Series(index).diff().dropna().median()
    return pd.Timedelta(minutes=max(1, round(step.total_seconds() / 60)))


def regularise(s: pd.Series, step: pd.Timedelta, end: pd.Timestamp | None = None) -> pd.Series:
    """Reindex onto a regular grid; absent or null samples become NaN (a gap)."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    start = s.index[0].floor(step)
    stop = (end if end is not None else s.index[-1]).floor(step)
    grid = pd.date_range(start, stop, freq=step, tz="UTC")
    return s.reindex(grid)


def stuck_mask(x: pd.Series, n: int) -> pd.Series:
    """True for every sample inside a run of >= n identical, non-missing values."""
    v = x.to_numpy(float)
    out = np.zeros(len(v), bool)
    i = 0
    while i < len(v):
        if np.isnan(v[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        if j - i + 1 >= n:
            out[i : j + 1] = True
        i = j + 1
    return pd.Series(out, index=x.index)


def spike_mask(x: pd.Series, window: int, k: float, rel_floor: float) -> pd.Series:
    """Hampel filter (|x - rolling median| > k * robust sigma, resolution floor) for isolated points."""
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=3).median()
    scale = np.maximum(1.4826 * mad, rel_floor * med.abs())
    scale = scale.where(scale > 0, 1e-6)
    dev = (x - med).abs()
    # A spike is isolated: both neighbours sit much closer to the median than the sample itself.
    # (A steep but real flank has one neighbour on the far side, so it is not flagged.)
    iso = ((x.shift(1) - med).abs() < 0.5 * dev) & ((x.shift(-1) - med).abs() < 0.5 * dev)
    return ((dev > k * scale) & iso).fillna(False)


def qa_flags(
    x: pd.Series,
    cfg: QAConfig | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a regular-grid frame: value + one boolean column per flag + ``ok``.

    ``end`` extends the expected grid (e.g. to the fetch time) so a stale tail counts as a gap.
    """
    cfg = cfg or QAConfig()
    step = infer_step(x.index)
    r = regularise(x, step, end)
    missing = r.isna()
    stuck = stuck_mask(r, cfg.stuck_samples)
    spike = spike_mask(r, cfg.spike_window, cfg.spike_k, cfg.spike_rel_floor) & ~stuck
    rng = (r < cfg.min_value) | ((r > cfg.max_value) if cfg.max_value is not None else False)
    out = pd.DataFrame({"value": r, "missing": missing, "stuck": stuck, "spike": spike, "range": rng.fillna(False)})
    out["ok"] = ~(out[list(FLAGS)].any(axis=1))
    out.attrs["step_min"] = step.total_seconds() / 60
    return out


def summarise(qa: pd.DataFrame) -> dict:
    n = len(qa)
    res = {"samples_expected": n, "step_min": qa.attrs.get("step_min")}
    res["completeness_pct"] = round(100.0 * (1 - qa["missing"].mean()), 2) if n else 0.0
    for f in FLAGS:
        res[f"{f}_pct"] = round(100.0 * qa[f].mean(), 2) if n else 0.0
    res["clean_pct"] = round(100.0 * qa["ok"].mean(), 2) if n else 0.0
    res["spike_count"] = int(qa["spike"].sum())
    res["stuck_runs"] = int(((qa["stuck"]) & ~(qa["stuck"].shift(1, fill_value=False))).sum())
    # longest trailing gap = stale feed
    tail = 0
    for v in qa["missing"].to_numpy()[::-1]:
        if not v:
            break
        tail += 1
    res["stale_tail_min"] = round(tail * (qa.attrs.get("step_min") or 0), 1)
    return res


def clean(qa: pd.DataFrame) -> pd.Series:
    """Values with flagged samples set to NaN (what the alarm engine and forecaster see)."""
    return qa["value"].where(qa["ok"])
