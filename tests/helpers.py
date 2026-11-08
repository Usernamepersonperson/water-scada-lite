from __future__ import annotations

import numpy as np
import pandas as pd


def idx(n: int, start: str = "2026-11-01T00:00:00Z", step_min: int = 5) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=f"{step_min}min", tz="UTC")


def series(values, step_min: int = 5, start: str = "2026-11-01T00:00:00Z") -> pd.Series:
    v = np.asarray(values, float)
    return pd.Series(v, index=idx(len(v), start, step_min))


def plot_doc(
    q,
    h=None,
    bands=((360.0, 430.0, "#FFFF00"), (430.0, 500.0, "#FF9900")),
    title="Test - Station PQPlot 20:38 (08.11.26)",
):
    n = len(q)
    xs = [t.isoformat() for t in idx(n)]
    data = []
    if h is not None:
        data.append({"name": "Wasserstand", "yaxis": "y", "x": xs, "y": list(h)})
    data.append({"name": "Abfluss", "yaxis": "y2", "x": xs, "y": list(q)})
    shapes = []
    for lo, hi, col in bands:
        for _opacity in (0.2, 0.8):  # the portal draws each band twice (history + latest strip)
            shapes.append({"type": "rect", "xref": "paper", "yref": "y2", "y0": lo, "y1": hi, "fillcolor": col})
    shapes.append({"type": "line", "yref": "paper", "x0": xs[0], "x1": xs[0]})
    return {"plot": {"layout": {"title": title, "shapes": shapes}, "data": data}}
