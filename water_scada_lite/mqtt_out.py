"""MQTT output: per-station state, alarms and compliance as retained JSON topics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .pipeline import Result


class Publisher(Protocol):
    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False): ...


def build_messages(result: Result, prefix: str = "waterscada") -> list[tuple[str, dict]]:
    msgs: list[tuple[str, dict]] = []
    for sid, r in result.stations.items():
        qv = r.qa_q["value"].dropna()
        lv = r.data.frame["level_m"].dropna()
        last = qv.index[-1] if len(qv) else None
        msgs.append(
            (
                f"{prefix}/{sid}/state",
                {
                    "station": sid,
                    "name": r.data.name,
                    "ts": last.isoformat() if last is not None else None,
                    "discharge_m3s": float(qv.iloc[-1]) if len(qv) else None,
                    "level_m": float(lv.iloc[-1]) if len(lv) else None,
                    "qa_ok": bool(r.qa_q["ok"].iloc[-1]),
                    "completeness_pct": round(100 * (1 - r.qa_q["missing"].mean()), 2),
                },
            )
        )
        if r.residual is not None:
            body = {k: v for k, v in r.residual.items() if not k.endswith("episodes")}
            msgs.append((f"{prefix}/{sid}/residual_flow", body))
    for a in result.engine.alarms:
        msgs.append((f"{prefix}/{a.station}/alarm/{a.id}", a.to_dict()))
    return msgs


def publish(result: Result, client: Publisher, prefix: str = "waterscada") -> int:
    msgs = build_messages(result, prefix)
    for topic, payload in msgs:
        client.publish(topic, json.dumps(payload, default=str), qos=1, retain=True)
    return len(msgs)


def dry_run(result: Result, path: Path, prefix: str = "waterscada") -> int:
    msgs = build_messages(result, prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for topic, payload in msgs:
            f.write(json.dumps({"topic": topic, "payload": payload}, default=str) + "\n")
    return len(msgs)


def connect(host: str, port: int = 1883):  # pragma: no cover - needs a broker
    import paho.mqtt.client as mqtt

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.connect(host, port, 30)
    c.loop_start()
    return c
