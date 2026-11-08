import numpy as np
import pytest
from helpers import series

from water_scada_lite import forecast as fc


def test_persistence_mae_on_ramp_known_value():
    lv = series(np.arange(100) * 0.01)  # +1 cm per step
    mae, n = fc.mae_cm(fc.persistence(lv, 12), lv, 12)
    assert mae == pytest.approx(12.0) and n == 88


def test_linear_extrapolation_exact_on_ramp():
    lv = series(np.arange(100) * 0.01)
    mae, _ = fc.mae_cm(fc.linear_extrapolation(lv, 12), lv, 12)
    assert mae == pytest.approx(0.0, abs=1e-9)


def test_arx_learns_trend_and_beats_persistence():
    t = np.arange(600)
    lv = series(500 + 0.5 * np.sin(t / 30.0))
    q = series(50 + 5 * np.sin(t / 30.0))
    r = fc.evaluate(lv, q, 6)
    assert r["mae_cm_arx"] < r["mae_cm_persistence"]
    assert r["arx_skill_vs_persistence_pct"] > 50


def test_arx_not_fitted_raises():
    with pytest.raises(RuntimeError):
        fc.ArxModel(6).predict(series(np.arange(50.0)))


def test_arx_needs_enough_samples():
    with pytest.raises(ValueError):
        fc.ArxModel(6).fit(series(np.arange(20.0)))


def test_arx_works_without_discharge():
    lv = series(500 + 0.5 * np.sin(np.arange(400) / 20.0))
    m = fc.ArxModel(6).fit(lv, None)
    assert m.predict(lv).notna().sum() > 300


def test_evaluate_rejects_bad_train_frac():
    with pytest.raises(ValueError):
        fc.evaluate(series(np.arange(300.0)), None, 6, train_frac=0.95)


def test_nan_pairs_are_skipped():
    lv = series(np.arange(100) * 0.01)
    lv.iloc[40:50] = np.nan
    mae, n = fc.mae_cm(fc.persistence(lv, 12), lv, 12)
    assert n < 88 and mae == pytest.approx(12.0)


def test_mae_all_nan_returns_nan():
    mae, n = fc.mae_cm(series([np.nan] * 5), series([np.nan] * 5), 1)
    assert np.isnan(mae) and n == 0
