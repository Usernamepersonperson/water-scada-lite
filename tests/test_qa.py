import numpy as np
import pandas as pd
import pytest
from helpers import series

from water_scada_lite import qa


def test_infer_step_median():
    s = series([1, 2, 3, 4], step_min=10)
    assert qa.infer_step(s.index) == pd.Timedelta(minutes=10)


def test_infer_step_needs_two_samples():
    with pytest.raises(ValueError):
        qa.infer_step(series([1]).index)


def test_gap_detected_after_regularise():
    full = series(np.arange(10.0))
    f = qa.qa_flags(full.drop(full.index[[3, 4]]))
    assert f["missing"].sum() == 2 and len(f) == 10
    assert f["ok"].sum() == 8


def test_null_value_counts_as_missing():
    s = series([1.0, np.nan, 3.0, 4.0])
    assert qa.qa_flags(s)["missing"].tolist() == [False, True, False, False]


def test_stale_tail_counts_as_gap_with_end():
    s = series(np.arange(10.0) + 1)
    end = s.index[-1] + pd.Timedelta(minutes=25)
    f = qa.qa_flags(s, end=end)
    assert len(f) == 15 and f["missing"].iloc[-5:].all()
    assert qa.summarise(f)["stale_tail_min"] == 25.0


def test_stuck_run_flagged_only_when_long_enough():
    v = list(np.linspace(1, 2, 20)) + [5.0] * 24 + list(np.linspace(2, 3, 20)) + [9.0] * 23
    m = qa.stuck_mask(series(v), 24)
    assert m.sum() == 24
    assert m.iloc[20:44].all()


def test_stuck_ignores_nan_runs():
    assert not qa.stuck_mask(series([np.nan] * 30), 24).any()


def test_isolated_spike_flagged():
    v = 10 + 0.05 * np.sin(np.arange(60) / 4)
    v[30] = 40
    f = qa.qa_flags(series(v))
    assert f["spike"].iloc[30] and f["spike"].sum() == 1


def test_steep_real_flank_not_flagged_as_spike():
    v = np.r_[np.full(30, 3.5), np.full(30, 12.0)]
    assert qa.qa_flags(series(v))["spike"].sum() == 0


def test_negative_discharge_flagged_range():
    v = np.r_[np.full(10, 2.0), [-1.0], np.full(10, 2.0)]
    f = qa.qa_flags(series(v))
    assert f["range"].iloc[10] and f["range"].sum() == 1


def test_max_value_range():
    f = qa.qa_flags(series([1.0, 2.0, 99.0, 2.0, 1.0]), qa.QAConfig(max_value=50))
    assert f["range"].tolist() == [False, False, True, False, False]


def test_summary_numbers():
    full = series(np.linspace(1, 5, 100))
    r = qa.summarise(qa.qa_flags(full.drop(full.index[10:20])))
    assert r["samples_expected"] == 100
    assert r["completeness_pct"] == 90.0
    assert r["missing_pct"] == 10.0
    assert r["clean_pct"] == 90.0
    assert r["step_min"] == 5.0


def test_clean_masks_flagged_samples():
    v = np.r_[np.full(10, 2.0), [-1.0], np.full(10, 2.0)]
    c = qa.clean(qa.qa_flags(series(v)))
    assert np.isnan(c.iloc[10]) and c.notna().sum() == 20
