import numpy as np
import pandas as pd
import pytest
from helpers import series

from water_scada_lite.alarms import (
    ACTIVE_ACK,
    ACTIVE_UNACK,
    CLEARED,
    CLEARED_UNACK,
    AlarmEngine,
    Rule,
    compare_flood_levels,
)

T = pd.Timestamp("2026-11-08T00:00:00Z")
FIVE = pd.Timedelta("5min")


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule("x", "weird", 1.0)
    with pytest.raises(ValueError):
        Rule("x", "high", 10.0, clear=12.0)
    with pytest.raises(ValueError):
        Rule("x", "low", 10.0, clear=8.0)
    with pytest.raises(ValueError):
        Rule("x", "high", 10.0, min_samples=0)


def test_default_hysteresis_levels():
    assert Rule("h", "high", 100.0, hysteresis=0.05).clear == pytest.approx(95.0)
    assert Rule("l", "low", 100.0, hysteresis=0.1).clear == pytest.approx(110.0)
    assert Rule("r", "rise", 20.0).clear == pytest.approx(10.0)


def test_debounce_single_sample_does_not_alarm():
    e = AlarmEngine()
    e.process_series("S", series([1, 1, 10, 1, 1]), [Rule("h", "high", 5.0, min_samples=2)])
    assert e.alarms == []


def test_high_alarm_raised_and_peak_tracked():
    e = AlarmEngine()
    e.process_series("S", series([1, 6, 7, 9, 8, 1, 1]), [Rule("h", "high", 5.0, min_samples=2)])
    (a,) = e.alarms
    assert a.peak == 9 and a.state == CLEARED_UNACK
    assert a.t_first_breach < a.t_raised < a.t_clear


CHATTER = [1, 6, 6, 5.2, 4.9, 5.2, 4.9, 5.2, 4.9, 1, 1]


def test_hysteresis_prevents_chatter():
    e = AlarmEngine()
    e.process_series("S", series(CHATTER), [Rule("h", "high", 5.0, hysteresis=0.05, min_samples=2)])
    assert len(e.alarms) == 1


def test_without_hysteresis_would_chatter():
    e = AlarmEngine()
    e.process_series("S", series(CHATTER), [Rule("h", "high", 5.0, clear=5.0, min_samples=1)])
    assert len(e.alarms) > 1


def test_retrigger_after_clear_is_new_instance():
    e = AlarmEngine()
    e.process_series("S", series([6, 6, 1, 1, 6, 6, 1]), [Rule("h", "high", 5.0)])
    assert [a.id for a in e.alarms] == [1, 2]


def test_nan_holds_state():
    e = AlarmEngine()
    e.process_series("S", series([6, 6, np.nan, np.nan, 6, 1]), [Rule("h", "high", 5.0)])
    assert len(e.alarms) == 1 and e.alarms[0].state == CLEARED_UNACK


def test_low_alarm():
    e = AlarmEngine()
    e.process_series("S", series([5, 2, 2, 3, 5, 5]), [Rule("lo", "low", 3.0, hysteresis=0.1, min_samples=2)])
    (a,) = e.alarms
    assert a.peak == 2 and a.t_clear is not None


def test_rise_rate_known_value():
    # 100 -> 130 over one hour (12 samples at 5 min) = 30 %/h
    v = np.r_[np.full(12, 100.0), np.linspace(100, 130, 13)[1:], np.full(10, 130.0)]
    e = AlarmEngine()
    e.process_series("S", series(v), [Rule("rise", "rise", 25.0, window_min=60, min_samples=1)])
    (a,) = e.alarms
    assert a.peak == pytest.approx(30.0, abs=0.01)


def test_rise_below_trigger_no_alarm():
    v = np.r_[np.full(12, 100.0), np.linspace(100, 110, 13)[1:]]
    e = AlarmEngine()
    e.process_series("S", series(v), [Rule("rise", "rise", 25.0, min_samples=1)])
    assert e.alarms == []


def test_ack_state_machine_active_path():
    e = AlarmEngine()
    r = Rule("h", "high", 5.0, min_samples=1)
    a = e.feed("S", r, T, 6.0)
    assert a.state == ACTIVE_UNACK
    e.ack(a.id, "op", T)
    assert a.state == ACTIVE_ACK and a.ack_by == "op"
    e.feed("S", r, T + FIVE, 1.0)
    assert a.state == CLEARED


def test_ack_after_clear():
    e = AlarmEngine()
    r = Rule("h", "high", 5.0, min_samples=1)
    a = e.feed("S", r, T, 6.0)
    e.feed("S", r, T + FIVE, 1.0)
    assert a.state == CLEARED_UNACK
    e.ack(a.id, "op", T + pd.Timedelta("1h"))
    assert a.state == CLEARED and a.t_ack is not None


def test_ack_idempotent_and_unknown():
    e = AlarmEngine()
    a = e.feed("S", Rule("h", "high", 5.0, min_samples=1), T, 6.0)
    e.ack(a.id, "first", T)
    e.ack(a.id, "second", T)
    assert a.ack_by == "first"
    with pytest.raises(KeyError):
        e.ack(999, "x", T)


def test_open_alarms_excludes_cleared_acked():
    e = AlarmEngine()
    r = Rule("h", "high", 5.0, min_samples=1)
    a = e.feed("S", r, T, 6.0)
    e.ack(a.id, "op", T)
    assert len(e.open_alarms()) == 1
    e.feed("S", r, T + FIVE, 1.0)
    assert e.open_alarms() == []


def test_alarm_to_dict_json_safe():
    e = AlarmEngine()
    a = e.feed("S", Rule("h", "high", 5.0, min_samples=1), T, 6.0)
    d = a.to_dict()
    assert d["t_raised"].startswith("2026-11-08") and d["state"] == ACTIVE_UNACK


def test_flood_comparison_levels():
    fc = compare_flood_levels("S", series([100, 250, 440, 300]), {2: 360.0, 3: 430.0, 4: 500.0})
    assert fc.level_reached == 3 and fc.max_discharge == 440
    assert fc.pct_of_level2 == pytest.approx(122.2, abs=0.05)


def test_flood_comparison_below_level2_and_no_thresholds():
    assert compare_flood_levels("S", series([1, 2]), {2: 360.0}).level_reached == 1
    assert compare_flood_levels("S", series([1, 2]), {}).pct_of_level2 is None
    assert compare_flood_levels("S", series([np.nan]), {2: 1.0}).max_discharge is None
