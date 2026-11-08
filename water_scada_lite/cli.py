"""Command line: ``python -m water_scada_lite {fetch,demo,serve,mqtt}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import foen
from .mqtt_out import connect, dry_run, publish
from .pipeline import Result, run, simulate_acks
from .settings import load_config

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def _load(args) -> tuple[Result, object]:
    cfg = load_config(args.config)
    if getattr(args, "live", False):
        stations = [foen.fetch_station(i) for i in cfg.station_ids]
    else:
        stations = foen.load_snapshot(Path(args.snapshot), cfg.station_ids)
    return run(stations, cfg), cfg


def cmd_fetch(args) -> int:
    cfg = load_config(args.config)
    stations = [foen.fetch_station(i) for i in cfg.station_ids]
    foen.save_snapshot(stations, Path(args.snapshot))
    print(f"saved {len(stations)} stations to {args.snapshot}")
    return 0


def cmd_demo(args) -> int:
    res, _ = _load(args)
    simulate_acks(res)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = res.summary()
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=str), "utf-8")
    n = dry_run(res, out / "mqtt_messages.jsonl")
    print(f"period (UTC): {summary['period_utc'][0]} .. {summary['period_utc'][1]}")
    print(f"alarms: {summary['alarms_total']} {summary['alarms_by_state']}")
    print(f"{'station':34} {'complete%':>9} {'flat%':>6} {'spikes':>6} {'alarms':>6} {'peak%L2':>8} {'MAE1h per/arx cm':>17}")
    for sid, s in summary["stations"].items():
        f1 = next((f for f in s["forecast"] if f["horizon_min"] == 60), None)
        fc = f"{f1['mae_cm_persistence']:.2f}/{f1['mae_cm_arx']:.2f}" if f1 else "-"
        q = s["qa_discharge"]
        print(f"{sid} {s['name'][:28]:30} {q['completeness_pct']:9.2f} {q['stuck_pct']:6.2f} {q['spike_count']:6d} "
              f"{s['alarms_total']:6d} {s['flood']['pct_of_level2'] or 0:8.1f} {fc:>17}")
    for sid, s in summary["stations"].items():
        r = s["residual"]
        if r:
            print(f"residual-flow {sid} min {r['min_flow_m3s']} m3/s cap {r['capacity_m3s']}: "
                  f"unregulated breach {r['breach_if_unregulated_pct']}% of time, {len(r['breach_episodes'])} episodes, "
                  f"intake closed {r['closed_pct']}%")
    print(f"MQTT dry-run: {n} messages -> {out / 'mqtt_messages.jsonl'}")
    if args.figures:
        from .figures import main_figure, post_figure

        docs = ROOT / "docs"
        docs.mkdir(exist_ok=True)
        sid = args.hero
        by = summary["stations"][sid]["alarms_by_rule"]
        txt = f"{summary['stations'][sid]['alarms_total']} alarms in 7 days ({by.get('elevated_flow', 0)} high-flow, {by.get('rise_rate', 0)} fast-rise)"
        post_figure(res, sid, docs / "post-figure.png", txt)
        main_figure(res, sid, docs / "figure.png")
        print("figures written to", docs)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .app import create_app

    res, _ = _load(args)
    simulate_acks(res)
    uvicorn.run(create_app(res), host=args.host, port=args.port)
    return 0


def cmd_mqtt(args) -> int:
    res, _ = _load(args)
    n = publish(res, connect(args.host, args.port))
    print(f"published {n} retained messages")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="water_scada_lite", description=__doc__)
    p.add_argument("--config", default=None, help="TOML config (default: bundled)")
    p.add_argument("--snapshot", default=str(FIXTURES), help="cached snapshot directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download current 7-day data and cache it as the snapshot").set_defaults(fn=cmd_fetch)
    d = sub.add_parser("demo", help="run the full pipeline on the cached snapshot and print the measured results")
    d.add_argument("--live", action="store_true", help="fetch live data instead of the snapshot")
    d.add_argument("--out", default=str(ROOT / "out"))
    d.add_argument("--figures", action="store_true", help="also regenerate docs/figure.png and docs/post-figure.png")
    d.add_argument("--hero", default="2084", help="station id for the hydrograph figures")
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve", help="start the web dashboard")
    s.add_argument("--live", action="store_true")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("mqtt", help="publish retained state/alarm topics to a broker")
    m.add_argument("--live", action="store_true")
    m.add_argument("--host", default="localhost")
    m.add_argument("--port", type=int, default=1883)
    m.set_defaults(fn=cmd_mqtt)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
