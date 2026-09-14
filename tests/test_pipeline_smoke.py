"""End-to-end CLI smoke test on synthetic data: TF-IDF primary, no network, no model download."""

import json
import sys
import zipfile
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sentiment_signal import cli, config


def _business_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "TEST_YEARS", (2013,))
    monkeypatch.setattr(config, "CALENDAR_MIN_TICKERS", 1)
    monkeypatch.setattr(config, "HOLDOUT_TRAIN_YEARS", (2010, 2012))
    monkeypatch.setattr(config, "HOLDOUT_START", "2013-01-01")
    monkeypatch.setattr(config, "HOLDOUT_END", "2013-12-31")
    rng = np.random.default_rng(0)
    days = _business_days(date(2009, 1, 5), date(2013, 12, 31))
    tickers = [f"T{i:02d}" for i in range(25)]  # >= 20 names a day, as in production
    # prices: each ticker's next-day return sign follows a hidden daily coin flip
    flips = {(t, d): rng.random() < 0.5 for t in tickers for d in days}
    with zipfile.ZipFile(tmp_path / "prices.zip", "w") as z:
        for t in tickers:
            price, lines = 50.0, ["date,volume,open,high,low,close,adj close"]
            for d in days:
                lines.append(f"{d},100000,{price},{price},{price},{price},{price}")
                price *= 1.01 if flips[(t, d)] else 0.99
            z.writestr(f"full_history/{t}.csv", "\n".join(lines))
    # news: one headline per ticker-day at 10:00 New York time, word tells tomorrow's direction
    rows = ["idx,title,date,stock"]
    for i, (t, d) in enumerate(flips):
        word = "soars" if flips[(t, d)] else "slumps"
        stamp = datetime.combine(d, time(10), ZoneInfo(config.TZ)).isoformat(sep=" ")
        rows.append(f"{i},{t} {word} on news,{stamp},{t}")
    (tmp_path / "news.csv").write_text("\n".join(rows))
    # the same 2013 headlines in Alpha Vantage's format stand in for the holdout collection
    av = [
        {
            "title": f"{t} {'soars' if up else 'slumps'} on news",
            "time_published": f"{d:%Y%m%d}T100000",
            "ticker_sentiment": [{"ticker": t, "relevance_score": "0.9"}],
        }
        for (t, d), up in flips.items()
        if d.year == 2013
    ]
    (tmp_path / "data/raw/av").mkdir(parents=True)
    (tmp_path / "data/raw/av/2013.json").write_text(json.dumps(av))
    return tmp_path


def _cli(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ss", *argv])
    cli.main()


def test_full_pipeline_runs_and_writes_a_report(workspace, monkeypatch):
    news = str(workspace / "news.csv")
    _cli(monkeypatch, "gate-g1", "--file", news, "--licence", "CC0", "--licence-ok")
    _cli(monkeypatch, "prices", "--zip", str(workspace / "prices.zip"))
    _cli(monkeypatch, "panel", "--file", news)
    with pytest.raises(SystemExit, match="not recorded"):
        _cli(monkeypatch, "run", "--model", "tfidf")
    gates_path = workspace / "results/gates.json"
    gates = json.loads(gates_path.read_text())
    # G4 needs the real model and G2ab the network, so both are recorded by hand
    gates["G4"] = {"pass": False, "note": "synthetic run: TF-IDF primary"}
    gates["G2ab"] = {"pass": False, "note": "synthetic run: no network"}
    gates_path.write_text(json.dumps(gates))
    _cli(monkeypatch, "holdout-panel")
    with pytest.raises(SystemExit, match="failed"):
        _cli(monkeypatch, "run", "--model", "tfidf")
    gates = json.loads(gates_path.read_text())
    gates["G2ab"]["pass"] = True
    gates_path.write_text(json.dumps(gates))
    _cli(monkeypatch, "run", "--model", "tfidf")
    _cli(monkeypatch, "run", "--model", "tfidf", "--null")
    with pytest.raises(SystemExit, match="gates are frozen"):
        _cli(monkeypatch, "gate-g1", "--file", news, "--licence", "CC0", "--licence-ok")
    holdout_panel = workspace / "data/processed/panel_holdout.parquet"
    written = holdout_panel.stat().st_mtime_ns
    with pytest.raises(SystemExit, match="gates are frozen"):
        _cli(monkeypatch, "holdout-panel")
    assert holdout_panel.stat().st_mtime_ns == written  # refused before rewriting its file
    # one informative word makes any fitted coefficient rank perfectly in one direction, so this
    # synthetic null is far from |t| < 2, and the pipeline must stop there
    with pytest.raises(SystemExit, match="leaks"):
        _cli(monkeypatch, "summarize")
    with pytest.raises(SystemExit, match="leaks"):
        _cli(monkeypatch, "report")
    with pytest.raises(SystemExit, match="leaks"):
        _cli(monkeypatch, "holdout")
    null_path = workspace / "results/tfidf_null.json"
    null = json.loads(null_path.read_text())
    assert abs(null["overall"]["ic_t"]) >= 2
    null["overall"]["ic_t"] = 0.0  # a stand-in clean null, so the rest of the chain runs
    null_path.write_text(json.dumps(null))
    _cli(monkeypatch, "vader")
    with pytest.raises(SystemExit, match="holdout"):
        _cli(monkeypatch, "summarize")
    with pytest.raises(SystemExit, match="summarize"):
        _cli(monkeypatch, "report")
    _cli(monkeypatch, "holdout")
    holdout_path = workspace / "results/holdout.json"
    holdout_path.write_text(holdout_path.read_text().replace('"tfidf"', '"chrono"'))
    with pytest.raises(SystemExit, match="holdout"):  # condition 4 must come from the primary
        _cli(monkeypatch, "summarize")
    _cli(monkeypatch, "holdout")
    _cli(monkeypatch, "summarize")
    _cli(monkeypatch, "report")
    summary = json.loads((workspace / "results/summary.json").read_text())
    tfidf = json.loads((workspace / "results/tfidf.json").read_text())["overall"]
    holdout = json.loads((workspace / "results/holdout.json").read_text())
    assert summary["primary"] == holdout["model"] == "tfidf"
    assert tfidf["mean_ic"] > 0.5  # the synthetic word fully determines the next-day sign
    bar = summary["bar"]["conditions"]
    assert bar["1_ic_t_ge_2"]["value"] == tfidf["ic_t"]
    assert bar["2_net_sharpe_ge_0_5"]["value"] == tfidf["net_sharpe"]
    lag = summary["mean_lag_ic"]
    assert bar["3_lag_ic_le_half"] == {"value": lag, "pass": lag <= 0.5 * tfidf["mean_ic"]}
    assert bar["4_holdout_ic_gt_0"]["value"] == holdout["mean_ic"] > 0.5
    report = (workspace / "REPORT.md").read_text()
    assert report.count("| tfidf |") == 1 and "| tfidf_null |" in report
    assert "permuted across all training rows" in report
    recorded = json.loads(gates_path.read_text())
    assert recorded["G1"]["pass"] is True and recorded["G2c"]["pass"] is True
    assert datetime.fromisoformat(recorded["G1"]["recorded_at"])


def test_report_shows_the_original_protocol_bar_when_deviations_exist(tmp_path):
    from sentiment_signal.report import build_report

    bar = {"conditions": {"1_ic_t_ge_2": {"value": 2.5, "pass": True}}, "publish": True}
    (tmp_path / "summary.json").write_text(json.dumps({"primary": "tfidf", "bar": bar}))
    (tmp_path / "deviations.json").write_text(json.dumps({"deviations": ["holdout shortened"]}))
    build_report(tmp_path, tmp_path / "REPORT.md")
    assert "original protocol: not available" in (tmp_path / "REPORT.md").read_text()
    original = {"bar": {**bar, "publish": False}}
    (tmp_path / "summary_original.json").write_text(json.dumps(original))
    build_report(tmp_path, tmp_path / "REPORT.md")
    assert "original protocol: **does not clear**" in (tmp_path / "REPORT.md").read_text()


def test_holdout_config_skips_nan_years_and_breaks_ties_early(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS", tmp_path)

    def best(ics_by_config: list[list[float]]) -> dict:
        validation = [
            {"test_year": 2013 + year, "config": c, "mean_ic": ic}
            for c, ics in enumerate(ics_by_config)
            for year, ic in enumerate(ics)
        ]
        (tmp_path / "tfidf.json").write_text(json.dumps({"validation": validation}))
        return cli._best_config("tfidf")

    nan = float("nan")
    # a config's NaN year is skipped for that config, as walk_forward skips it
    assert best([[0.05, nan], [nan, nan], [0.03, 0.03]]) == config.TFIDF_GRID[0]
    assert best([[0.01, 0.01], [nan, nan], [0.03, 0.03], [0.03, 0.03]]) == config.TFIDF_GRID[2]
