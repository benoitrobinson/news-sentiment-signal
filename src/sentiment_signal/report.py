"""REPORT.md and the event-study chart, built only from results/*.json."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LIMITATIONS = [
    "Price history misses some delisted companies (survivorship bias).",
    "Headlines only: no article bodies.",
    "Timestamps are the publisher's, not the time the market first saw the news.",
    "The holdout is two years and a different source, so its condition checks direction only.",
    "Validation year Y-1 text was inside the ChronoBERT version used for test year Y, so config "
    "selection is mildly optimistic; test-year scores are lookahead-free.",
    "If Alpha Vantage's timezone is undocumented, New York wall time is assumed (conservative).",
]


def _load(results: Path, name: str) -> dict:
    path = results / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.4f}"
    return str(x)


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]


def _bar(label: str, bar: dict) -> list[str]:
    verdict = "clears" if bar.get("publish") else "does not clear"
    rows = [[k, v["value"], v["pass"]] for k, v in bar.get("conditions", {}).items()]
    return [f"{label}: **{verdict}**.", ""] + _table(["Condition", "Value", "Pass"], rows)


def _plot_event_study(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for q in sorted({r["q"] for r in rows}):
        pts = sorted((r["offset"], r["cum_r"]) for r in rows if r["q"] == q)
        ax.plot([p[0] for p in pts], [p[1] * 1e4 for p in pts], marker="o", label=f"quintile {q}")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("trading days (0 = first tradable next-day return)")
    ax.set_ylabel("cumulative excess return (bps)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_report(results: Path, out_md: Path) -> None:
    summary = _load(results, "summary")
    gates = _load(results, "gates")
    deviations = _load(results, "deviations").get("deviations", [])
    primary = summary.get("primary", "n/a")
    lines = ["# News sentiment signal — report", "", f"**Primary model:** `{primary}`.", ""]
    lines += ["## Publish bar", ""] + _bar("The pre-registered publish bar", summary.get("bar", {}))
    lines += ["", "## Models, walk-forward 2013–2020 out of sample", ""]
    rows = []
    # dict.fromkeys keeps the order and drops the repeat when TF-IDF is the primary
    for name in dict.fromkeys((primary, f"{primary}_null", "tfidf", "vader", "old_bug_vader")):
        overall = _load(results, name).get("overall")
        if overall:
            rows.append(
                [
                    name,
                    overall["days"],
                    overall["mean_ic"],
                    overall["ic_t"],
                    overall["net_sharpe"],
                    overall.get("hit_rate", float("nan")),
                ]
            )
    lines += _table(["Run", "Days", "Mean IC", "IC t (NW)", "Net Sharpe", "Hit rate"], rows)
    lines += [
        "",
        "`*_null` retrains on labels permuted across all training rows and must show |t| < 2. "
        "`old_bug_vader` repeats the original repo's same-day join.",
        "",
        "## Robustness",
        "",
        f"- Mean lag IC: {_fmt(summary.get('mean_lag_ic', float('nan')))}",
        f"- Deflated Sharpe ratio: {_fmt(summary.get('deflated_sharpe', float('nan')))}",
        "",
    ]
    lines += _table(
        ["Liquidity tercile", "Days", "Mean IC", "IC t (NW)", "Net Sharpe"],
        [
            [t["tercile"], t["days"], t["mean_ic"], t["ic_t"], t["net_sharpe"]]
            for t in summary.get("terciles", [])
        ],
    )
    if summary.get("event_study"):
        _plot_event_study(summary["event_study"], results / "event_study.png")
        lines += ["", "![event study](results/event_study.png)"]
    lines += ["", "## Gates", ""]
    lines += _table(
        ["Gate", "Pass", "Recorded"],
        [[k, v.get("pass"), v.get("recorded_at")] for k, v in gates.items()],
    )
    lines += ["", "## Deviations", ""]
    lines += [f"- {d}" for d in deviations] or ["None."]
    if deviations:  # spec section 5: conditions 1-4 are also reported under the original protocol
        original = _load(results, "summary_original").get("bar")
        lines += [""] + (
            _bar("Publish bar under the original protocol", original)
            if original
            else [
                "Publish bar under the original protocol: not available (no summary was produced)."
            ]
        )
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS] + [""]
    out_md.write_text("\n".join(lines))
