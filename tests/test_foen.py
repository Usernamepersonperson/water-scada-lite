import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from helpers import plot_doc

from water_scada_lite import foen
from water_scada_lite.foen import ParseError, parse_plot


def test_parse_basic_series_and_name():
    sd = parse_plot(plot_doc([1.0, 2.0, 3.0], [500.1, 500.2, 500.3]), "9999")
    assert sd.name == "Test - Station"
    assert list(sd.frame.columns) == ["level_m", "discharge_m3s"]
    assert sd.frame["discharge_m3s"].tolist() == [1.0, 2.0, 3.0]
    assert str(sd.frame.index.tz) == "UTC"


def test_parse_null_becomes_nan():
    sd = parse_plot(plot_doc([1.0, None, 3.0], [5, None, 5]), "1")
    assert pd.isna(sd.frame["discharge_m3s"].iloc[1])


def test_bands_deduplicated_and_mapped_to_levels():
    sd = parse_plot(plot_doc([1.0, 2.0]), "1")
    assert [(b.level, b.lower, b.upper) for b in sd.bands] == [(2, 360.0, 430.0), (3, 430.0, 500.0)]
    assert sd.threshold(2) == 360.0
    assert sd.threshold(5) is None


def test_station_without_bands():
    sd = parse_plot(plot_doc([1.0, 2.0], bands=()), "1")
    assert sd.bands == []


def test_discharge_only_station_has_nan_level():
    sd = parse_plot(plot_doc([1.0, 2.0]), "1")
    assert sd.frame["level_m"].isna().all()


@pytest.mark.parametrize("doc", [{}, {"plot": {}}, {"plot": {"layout": {}, "data": []}}, None])
def test_malformed_documents_raise(doc):
    with pytest.raises(ParseError):
        parse_plot(doc, "1")


def test_mismatched_xy_raises():
    doc = plot_doc([1.0, 2.0, 3.0])
    doc["plot"]["data"][0]["y"] = [1.0]
    with pytest.raises(ParseError):
        parse_plot(doc, "1")


def test_bad_timestamp_raises():
    doc = plot_doc([1.0, 2.0])
    doc["plot"]["data"][0]["x"][0] = "not-a-date"
    with pytest.raises(ParseError):
        parse_plot(doc, "1")


def test_duplicate_timestamps_keep_last():
    doc = plot_doc([1.0, 2.0])
    doc["plot"]["data"][0]["x"][1] = doc["plot"]["data"][0]["x"][0]
    sd = parse_plot(doc, "1")
    assert len(sd.frame) == 1 and sd.frame["discharge_m3s"].iloc[0] == 2.0


def test_fetch_station_uses_station_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=plot_doc([1.0, 2.0]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sd = foen.fetch_station("2084", client)
    assert seen["url"].endswith("/2084_p_q_7days_de.json")
    assert sd.fetched_at is not None


def test_fetch_station_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        foen.fetch_station("1", client)


def test_snapshot_roundtrip(tmp_path):
    sd = parse_plot(plot_doc([1.0, None, 3.0], [5, 6, 7]), "42")
    sd.fetched_at = pd.Timestamp("2026-11-08T12:00:00Z")
    foen.save_snapshot([sd], tmp_path)
    back = foen.load_snapshot(tmp_path)[0]
    assert back.station_id == "42" and back.name == sd.name
    assert back.frame["discharge_m3s"].isna().tolist() == [False, True, False]
    assert back.bands == sd.bands
    assert back.fetched_at == sd.fetched_at
    assert json.loads((tmp_path / "stations.json").read_text("utf-8"))["42"]["bands"][0]["level"] == 2


def test_committed_fixture_has_ten_stations_with_official_levels():
    stations = foen.load_snapshot(Path(__file__).parent.parent / "fixtures")
    assert len(stations) == 10
    bern = next(s for s in stations if s.station_id == "2135")
    assert [b.lower for b in bern.bands] == [360.0, 430.0, 500.0, 560.0]
