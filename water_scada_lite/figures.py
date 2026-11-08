"""Result figures: README figure (1600x900) and phone-sized post figure (960x700)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .pipeline import Result, StationResult  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def _style(ax, size: float) -> None:
    ax.set_facecolor(BG)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=size)
    for s in ax.spines.values():
        s.set_color(GRID)


def _local(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("Europe/Zurich").tz_localize(None)


def _events(res: Result, sid: str, rules: tuple[str, ...] | None = None):
    return [a for a in res.engine.for_station(sid) if rules is None or a.rule in rules]


def _hydrograph(ax, res: Result, sid: str, fs: float, lw: float = 1.8) -> None:
    r: StationResult = res.stations[sid]
    q = r.qa_q["value"]
    x = _local(q.index)
    ax.plot(x, q.values, color=ACCENT, lw=lw, label="discharge (5 min)")
    thr = next((x_ for x_ in r.rules if x_.name == "elevated_flow"), None)
    if thr is not None:
        ax.axhline(thr.trigger, color=AMBER, ls="--", lw=1.6, label=f"alarm: 3 x baseline = {thr.trigger:.1f}")
    for a in _events(res, sid):
        t0 = a.t_first_breach.tz_convert("Europe/Zurich").tz_localize(None)
        t1 = (a.t_clear or q.index[-1]).tz_convert("Europe/Zurich").tz_localize(None)
        ax.axvspan(t0, t1, color=RED, alpha=0.22, lw=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xlim(x[0], x[-1])
    _style(ax, fs)


def post_figure(res: Result, sid: str, path: Path, n_alarm_text: str) -> None:
    font = _font()
    plt.rcParams["font.family"] = font
    r = res.stations[sid]
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.28, left=0.09, right=0.97, top=0.78, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    _hydrograph(ax, res, sid, 11)
    ax.set_ylabel("discharge, m3/s", color=MUTED, fontsize=11)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11)
    ax.text(0.99, 0.70, "red bands = detected alarm events", transform=ax.transAxes, ha="right", va="top", color=RED, fontsize=11)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    q = r.qa_q["value"]
    rate = (q - q.shift(12)) / q.shift(12) * 100.0
    ax2.plot(_local(q.index), rate.values, color=GREEN, lw=1.4)
    rule = next(x for x in r.rules if x.kind == "rise")
    ax2.axhline(rule.trigger, color=AMBER, ls="--", lw=1.6)
    ax2.text(_local(q.index)[5], rule.trigger + 45, f"rise alarm: +{rule.trigger:.0f} %/h", color=AMBER, fontsize=11)
    ax2.set_ylabel("rise, %/h", color=MUTED, fontsize=11)
    _style(ax2, 11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    start, end = (pd.Timestamp(t).strftime("%d %b") for t in res.period)
    name = r.data.name.replace(" - ", ", ")
    fig.text(0.04, 0.945, f"{name}: {n_alarm_text}", color=TEXT, fontsize=13.7, fontweight="bold", ha="left")
    fig.text(0.04, 0.895, f"Open data: FOEN/BAFU, {start} to {end} 2026 (UTC), 5-min readings", color=MUTED, fontsize=11, ha="left")
    fig.text(0.04, 0.855, "Alarms with hysteresis and acknowledgement, all computed by water-scada-lite", color=MUTED, fontsize=11, ha="left")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def main_figure(res: Result, sid: str, path: Path) -> None:
    font = _font()
    plt.rcParams["font.family"] = font
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.28, left=0.06, right=0.98, top=0.86, bottom=0.09)
    ax = fig.add_subplot(gs[0, :])
    _hydrograph(ax, res, sid, 12)
    ax.set_ylabel("discharge, m3/s", color=MUTED, fontsize=12)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=12)
    ax.set_title("(a) Hydrograph with alarm threshold; red = detected alarm events", color=MUTED, fontsize=12, loc="left")

    ids = list(res.stations)
    labels = [res.stations[i].data.name.split(" - ")[0] + "\n" + res.stations[i].data.name.split(" - ")[-1].split(",")[0] for i in ids]
    ax = fig.add_subplot(gs[1, 0])
    comp = [res.stations[i].qa_q["missing"].mean() * 100 for i in ids]
    stuck = [res.stations[i].qa_q["stuck"].mean() * 100 for i in ids]
    spike = [res.stations[i].qa_q["spike"].mean() * 100 for i in ids]
    y = range(len(ids))
    ax.barh(y, comp, color=RED, label="gap", lw=0)
    ax.barh(y, stuck, left=comp, color=AMBER, label="flat >= 2 h", lw=0)
    ax.barh(y, spike, left=[a + b for a, b in zip(comp, stuck, strict=True)], color=VIOLET, label="spike", lw=0)
    ax.set_yticks(list(y), [lb.split("\n")[1][:11] for lb in labels], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("% of samples flagged", color=MUTED, fontsize=11)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper right")
    ax.set_title("(b) QA flags per station", color=MUTED, fontsize=12, loc="left")
    _style(ax, 10)

    ax = fig.add_subplot(gs[1, 1])
    w = 0.27
    h1 = {i: next((f for f in res.stations[i].forecast if f["horizon_min"] == 60), None) for i in ids}
    ok = [i for i in ids if h1[i]]
    xs = range(len(ok))
    ax.bar([x - w for x in xs], [h1[i]["mae_cm_persistence"] for i in ok], w, color=GREY, label="persistence")
    ax.bar(list(xs), [h1[i]["mae_cm_linear"] for i in ok], w, color=AMBER, label="linear")
    ax.bar([x + w for x in xs], [h1[i]["mae_cm_arx"] for i in ok], w, color=GREEN, label="ARX")
    ax.set_yscale("log")
    ax.set_ylim(0.08, 60)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(list(xs), [res.stations[i].data.name.split(" - ")[-1].split(",")[0][:9] for i in ok], rotation=60, fontsize=9)
    ax.set_ylabel("MAE, cm (log)", color=MUTED, fontsize=11)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper left", ncol=3)
    ax.set_title("(c) 1 h level forecast error", color=MUTED, fontsize=12, loc="left")
    _style(ax, 10)

    ax = fig.add_subplot(gs[1, 2])
    pct = [res.stations[i].flood["pct_of_level2"] or 0 for i in ids]
    ax.barh(list(y), pct, color=ACCENT)
    ax.axvline(100, color=RED, ls="--", lw=1.6)
    ax.text(98, len(ids) - 0.4, "official level 2", color=RED, fontsize=10, ha="right")
    ax.set_yticks(list(y), [lb.split("\n")[1][:11] for lb in labels], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("week peak, % of official level-2 discharge", color=MUTED, fontsize=11)
    ax.set_title("(d) Distance to official flood level", color=MUTED, fontsize=12, loc="left")
    _style(ax, 10)

    start, end = (pd.Timestamp(t).strftime("%d %b") for t in res.period)
    fig.text(0.06, 0.945, "water-scada-lite: monitoring and alarms on 10 FOEN/BAFU river stations", color=TEXT, fontsize=19, fontweight="bold")
    fig.text(0.06, 0.905, f"Open data: FOEN/BAFU, {start} to {end} 2026 (UTC), 5-min readings. (a) Muota - Ingenbohl.", color=MUTED, fontsize=12)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
