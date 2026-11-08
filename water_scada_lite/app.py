"""Small FastAPI dashboard + JSON API over a pipeline :class:`Result`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import Result

WEB = Path(__file__).parent / "web"


class AckBody(BaseModel):
    by: str = "operator"


def _nums(s: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), 4) for v in s]


def create_app(result: Result) -> FastAPI:
    app = FastAPI(title="water-scada-lite", version="0.1.0")

    def station(sid: str):
        if sid not in result.stations:
            raise HTTPException(404, f"unknown station {sid}")
        return result.stations[sid]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/summary")
    def summary():
        return result.summary()

    @app.get("/api/stations")
    def stations():
        out = []
        for sid, r in result.stations.items():
            qv = r.qa_q["value"].dropna()
            open_n = sum(1 for a in result.engine.open_alarms() if a.station == sid)
            out.append(
                {
                    "id": sid,
                    "name": r.data.name,
                    "last_ts": qv.index[-1].isoformat() if len(qv) else None,
                    "discharge_m3s": float(qv.iloc[-1]) if len(qv) else None,
                    "completeness_pct": round(100 * (1 - r.qa_q["missing"].mean()), 1),
                    "open_alarms": open_n,
                    "has_intake": r.residual is not None,
                }
            )
        return out

    @app.get("/api/stations/{sid}/series")
    def series(sid: str):
        r = station(sid)
        qa = r.qa_q
        thresholds = {x.name: x.trigger for x in r.rules if x.kind == "high"}
        return {
            "id": sid,
            "name": r.data.name,
            "t": [t.isoformat() for t in qa.index],
            "q": _nums(qa["value"]),
            "bad": [bool(v) for v in ~qa["ok"]],
            "thresholds": thresholds,
            "min_flow": r.residual["min_flow_m3s"] if r.residual else None,
            "alarms": [a.to_dict() for a in result.engine.for_station(sid)],
        }

    @app.get("/api/alarms")
    def alarms(state: str | None = None):
        items = [a.to_dict() for a in result.engine.alarms if state is None or a.state == state]
        return items

    @app.post("/api/alarms/{alarm_id}/ack")
    def ack(alarm_id: int, body: AckBody):
        try:
            a = result.engine.ack(alarm_id, body.by, pd.Timestamp.now(tz="UTC"))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return a.to_dict()

    return app
