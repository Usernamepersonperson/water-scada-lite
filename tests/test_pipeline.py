import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from water_scada_lite import foen, mqtt_out
from water_scada_lite.app import create_app
from water_scada_lite.cli import main
from water_scada_lite.pipeline import run, simulate_acks
from water_scada_lite.settings import load_config

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def _base(cfg):
    return run(foen.load_snapshot(FIX, cfg.station_ids), cfg)


@pytest.fixture()
def result(_base):
    return copy.deepcopy(_base)


# -- settings -------------------------------------------------------------------------------
def test_default_config_ten_stations_four_intakes(cfg):
    assert len(cfg.station_ids) == 10 and len(cfg.intakes) == 4
    assert cfg.horizons_min == [60, 180]


def test_config_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[stations]]\nid = "1"\n[[stations]]\nid = "1"\n', "utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_config_without_stations_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[qa]\nstuck_samples = 5\n", "utf-8")
    with pytest.raises(ValueError):
        load_config(p)


# -- pipeline on the cached real snapshot (regression on measured numbers) --------------------
def test_pipeline_period_and_station_count(result):
    assert len(result.stations) == 10
    assert result.period[0].startswith("2026-11-01") and result.period[1].startswith("2026-11-08")


def test_measured_alarm_counts(result):
    s = result.summary()
    assert s["alarms_total"] == 36
    assert s["stations"]["2084"]["alarms_by_rule"] == {"rise_rate": 12, "elevated_flow": 6}
    assert s["stations"]["2473"]["alarms_by_rule"] == {"rise_rate": 10}
    assert s["stations"]["2135"]["alarms_total"] == 0


def test_no_official_flood_level_reached_in_snapshot(result):
    assert all(r.flood["official_level_reached"] == 1 for r in result.stations.values())
    assert result.stations["2009"].flood["pct_of_level2"] == pytest.approx(27.6, abs=0.1)


def test_stale_station_detected(result):
    q = result.summary()["stations"]["2327"]["qa_discharge"]
    assert q["completeness_pct"] < 70 and q["stale_tail_min"] > 200


def test_complete_stations_report_full_completeness(result):
    assert result.summary()["stations"]["2135"]["qa_discharge"]["completeness_pct"] == 100.0


def test_residual_only_for_configured_intakes(result):
    s = result.summary()["stations"]
    assert s["2084"]["residual"] is not None and s["2135"]["residual"] is None
    assert s["2084"]["residual"]["breach_if_unregulated_pct"] == pytest.approx(52.26, abs=0.01)


def test_forecast_arx_beats_persistence_on_hydropeaking_station(result):
    f = next(f for f in result.summary()["stations"]["2473"]["forecast"] if f["horizon_min"] == 60)
    assert f["mae_cm_arx"] < f["mae_cm_persistence"] / 2


def test_summary_is_json_serialisable(result):
    json.dumps(result.summary())


def test_simulate_acks_changes_states(result):
    simulate_acks(result)
    states = result.summary()["alarms_by_state"]
    assert states == {"cleared_unack": 18, "cleared": 18}


# -- mqtt ---------------------------------------------------------------------------------------
class FakeClient:
    def __init__(self):
        self.sent = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append((topic, json.loads(payload), qos, retain))


def test_mqtt_topics_and_retain(result):
    c = FakeClient()
    n = mqtt_out.publish(result, c)
    assert n == len(c.sent) == 10 + 4 + 36
    topics = [t for t, *_ in c.sent]
    assert "waterscada/2084/state" in topics and "waterscada/2084/residual_flow" in topics
    assert any(t.startswith("waterscada/2084/alarm/") for t in topics)
    assert all(r and q == 1 for _, _, q, r in c.sent)


def test_mqtt_state_payload(result):
    state = dict((t, p) for t, p in mqtt_out.build_messages(result))["waterscada/2135/state"]
    assert state["station"] == "2135" and state["discharge_m3s"] > 0 and state["qa_ok"] is True


def test_mqtt_dry_run_file(result, tmp_path):
    n = mqtt_out.dry_run(result, tmp_path / "m.jsonl")
    lines = (tmp_path / "m.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == n and "topic" in json.loads(lines[0])


# -- web app --------------------------------------------------------------------------------------
def test_api_stations_and_series(result):
    c = TestClient(create_app(result))
    st = c.get("/api/stations").json()
    assert len(st) == 10
    s = c.get("/api/stations/2084/series").json()
    assert len(s["t"]) == len(s["q"]) == len(s["bad"]) and s["min_flow"] == 2.5
    assert "elevated_flow" in s["thresholds"] and len(s["alarms"]) == 18


def test_api_unknown_station_404(result):
    assert TestClient(create_app(result)).get("/api/stations/0000/series").status_code == 404


def test_api_ack_flow(result):
    c = TestClient(create_app(result))
    first = c.get("/api/alarms").json()[0]
    assert first["state"] == "cleared_unack"
    r = c.post(f"/api/alarms/{first['id']}/ack", json={"by": "tester"})
    assert r.status_code == 200 and r.json()["state"] == "cleared" and r.json()["ack_by"] == "tester"
    assert c.post("/api/alarms/99999/ack", json={}).status_code == 404
    assert len(c.get("/api/alarms", params={"state": "cleared"}).json()) == 1


def test_dashboard_page_served(result):
    r = TestClient(create_app(result)).get("/")
    assert r.status_code == 200 and "water-scada-lite" in r.text


# -- cli ----------------------------------------------------------------------------------------
def test_cli_demo_writes_results(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "alarms: 36" in out
    data = json.loads((tmp_path / "results.json").read_text("utf-8"))
    assert data["alarms_total"] == 36 and (tmp_path / "mqtt_messages.jsonl").exists()
