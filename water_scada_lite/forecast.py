"""Short-horizon water-level forecast: persistence vs linear extrapolation vs a ridge ARX model.

All models predict the level ``h`` steps ahead from QA-clean samples only. The ARX model regresses
the future level change on the last ``lags`` level differences and on relative discharge changes.
Evaluation is chronological (train on the first part, test on the rest); no shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ArxModel:
    horizon: int
    lags: int = 12
    q_lags: tuple[int, ...] = (1, 3, 6, 12)
    ridge: float = 1e-3
    coef: np.ndarray | None = None

    def features(self, level: pd.Series, q: pd.Series | None) -> pd.DataFrame:
        cols = {f"dl{k}": level.shift(k) - level for k in range(1, self.lags + 1)}
        if q is not None:
            for k in self.q_lags:
                cols[f"dq{k}"] = (q.shift(k) - q) / q.where(q > 0)
        return pd.DataFrame(cols)

    def _xy(self, level: pd.Series, q: pd.Series | None):
        X = self.features(level, q)
        y = level.shift(-self.horizon) - level
        m = X.notna().all(axis=1) & y.notna()
        return X[m], y[m]

    def fit(self, level: pd.Series, q: pd.Series | None = None) -> ArxModel:
        X, y = self._xy(level, q)
        if len(X) < 3 * X.shape[1]:
            raise ValueError(f"not enough clean samples to fit ({len(X)})")
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        reg = self.ridge * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(A.T @ A + reg, A.T @ y.to_numpy())
        return self

    def predict(self, level: pd.Series, q: pd.Series | None = None) -> pd.Series:
        """Predicted level at t+h, indexed by t (NaN where features are incomplete)."""
        if self.coef is None:
            raise RuntimeError("model is not fitted")
        X = self.features(level, q)
        ok = X.notna().all(axis=1)
        pred = pd.Series(np.nan, index=level.index)
        pred[ok] = level[ok] + self.coef[0] + X[ok].to_numpy() @ self.coef[1:]
        return pred


def persistence(level: pd.Series, horizon: int) -> pd.Series:
    return level.copy()


def linear_extrapolation(level: pd.Series, horizon: int, span: int = 6) -> pd.Series:
    slope = (level - level.shift(span)) / span
    return level + slope * horizon


def mae_cm(pred: pd.Series, level: pd.Series, horizon: int) -> tuple[float, int]:
    """MAE in cm of prediction made at t against observed level at t+h, on clean pairs."""
    truth = level.shift(-horizon)
    m = pred.notna() & truth.notna()
    if not m.any():
        return float("nan"), 0
    return float((pred[m] - truth[m]).abs().mean() * 100.0), int(m.sum())


def evaluate(level: pd.Series, q: pd.Series | None, horizon: int, train_frac: float = 0.6) -> dict:
    """Fit on the first ``train_frac`` of the series and score all models on the remainder."""
    if not 0.2 <= train_frac <= 0.9:
        raise ValueError("train_frac must be in [0.2, 0.9]")
    cut = int(len(level) * train_frac)
    lv_tr, lv_te = level.iloc[:cut], level.iloc[cut:]
    q_tr = q.iloc[:cut] if q is not None else None
    q_te = q.iloc[cut:] if q is not None else None
    model = ArxModel(horizon).fit(lv_tr, q_tr)
    res = {"horizon_steps": horizon, "train_samples": int(lv_tr.notna().sum()), "test_start": lv_te.index[0].isoformat()}
    for name, pred in (
        ("persistence", persistence(lv_te, horizon)),
        ("linear", linear_extrapolation(lv_te, horizon)),
        ("arx", model.predict(lv_te, q_te)),
    ):
        m, n = mae_cm(pred, lv_te, horizon)
        res[f"mae_cm_{name}"] = round(m, 3)
        res["n_test"] = n
    base = res["mae_cm_persistence"]
    res["arx_skill_vs_persistence_pct"] = round(100.0 * (1 - res["mae_cm_arx"] / base), 1) if base > 0 else None
    return res
