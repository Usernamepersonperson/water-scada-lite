import numpy as np
import pytest
from helpers import series

from water_scada_lite.residual import Intake, allowed_diversion, check_residual_flow


def test_allowed_diversion_clips():
    d = allowed_diversion(series([1.0, 2.0, 3.0, 10.0]), Intake(min_flow=2.0, capacity=3.0))
    assert d.tolist() == [0.0, 0.0, 1.0, 3.0]


def test_invalid_intake():
    with pytest.raises(ValueError):
        Intake(min_flow=-1, capacity=1)
    with pytest.raises(ValueError):
        Intake(min_flow=1, capacity=0)


def test_closed_minutes_and_episode():
    q = series([5, 5, 1, 1, 1, 1, 5, 5])
    r = check_residual_flow(q, Intake(min_flow=2.0, capacity=1.0, min_duration_min=15))
    assert r["closed_minutes"] == 20.0
    assert len(r["closed_episodes"]) == 1 and r["closed_episodes"][0]["duration_min"] == 20.0


def test_short_dip_below_min_duration_not_an_episode():
    q = series([5, 5, 1, 5, 5, 5])
    r = check_residual_flow(q, Intake(min_flow=2.0, capacity=1.0, min_duration_min=15))
    assert r["closed_minutes"] == 5.0 and r["closed_episodes"] == []


def test_breach_if_unregulated_threshold():
    # Q - capacity < min_flow  <=>  Q < 3.0
    q = series([2.9, 3.0, 3.1, 2.0, 5.0, 5.0])
    r = check_residual_flow(q, Intake(min_flow=2.0, capacity=1.0, min_duration_min=5))
    assert r["breach_if_unregulated_minutes"] == 10.0
    assert not r["compliant"]


def test_compliant_when_flow_is_high():
    r = check_residual_flow(series([9.0] * 12), Intake(min_flow=2.0, capacity=3.0))
    assert r["compliant"] and r["closed_minutes"] == 0 and r["breach_episodes"] == []


def test_nan_samples_excluded_from_denominator():
    q = series([1.0, np.nan, np.nan, 1.0])
    r = check_residual_flow(q, Intake(min_flow=2.0, capacity=1.0, min_duration_min=5))
    assert r["valid_minutes"] == 10.0 and r["closed_pct"] == 100.0


def test_volumes_known_values():
    # 1 hour of Q=4 m3/s; min 1, cap 2 -> regulated diverts min(3, 2) = 2 m3/s = 7200 m3
    r = check_residual_flow(series([4.0] * 12), Intake(min_flow=1.0, capacity=2.0))
    assert r["diverted_volume_regulated_m3"] == 7200
    assert r["diverted_volume_unregulated_m3"] == 7200
