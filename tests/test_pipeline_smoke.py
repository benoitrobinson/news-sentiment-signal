"""End-to-end CLI smoke test on synthetic data: TF-IDF primary, no network, no model download."""

import json
import sys
import zipfile
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
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
        offset = "-04:00" if 4 <= d.month <= 10 else "-05:00"  # New York summer / winter time
        rows.append(f"{i},{t} {word} on news,{d} 10:00:00{offset},{t}")
    (tmp_path / "news.csv").write_text("\n".join(rows))
    return tmp_path


def _cli(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ss", *argv])
    cli.main()


def test_full_pipeline_runs_and_writes_a_report(workspace, monkeypatch):
    news = str(workspace / "news.csv")
    _cli(monkeypatch, "gate-g1", "--file", news, "--licence", "CC0", "--licence-ok")
    _cli(monkeypatch, "prices", "--zip", str(workspace / "prices.zip"))
    _cli(monkeypatch, "panel", "--file", news)
    gates = json.loads((workspace / "results/gates.json").read_text())
    gates["G4"] = {"pass": False, "note": "synthetic run: TF-IDF primary"}
    (workspace / "results/gates.json").write_text(json.dumps(gates))
    _cli(monkeypatch, "run", "--model", "tfidf")
    _cli(monkeypatch, "run", "--model", "tfidf", "--null")
    _cli(monkeypatch, "vader")
    holdout = pl.read_parquet(workspace / "data/processed/panel.parquet").filter(
        pl.col("signal_date").dt.year() == 2013
    )
    holdout.write_parquet(workspace / "data/processed/panel_holdout.parquet")
    _cli(monkeypatch, "holdout")
    _cli(monkeypatch, "summarize")
    _cli(monkeypatch, "report")
    summary = json.loads((workspace / "results/summary.json").read_text())
    assert summary["primary"] == "tfidf"
    tfidf = json.loads((workspace / "results/tfidf.json").read_text())["overall"]
    null = json.loads((workspace / "results/tfidf_null.json").read_text())["overall"]
    assert tfidf["mean_ic"] > 0.5  # the synthetic word fully determines the next-day sign
    # one informative word makes any fitted coefficient rank perfectly in one direction, so the
    # null magnitude is not testable on synthetic data; the null run must still complete
    assert null["days"] == tfidf["days"]
    assert "## Publish bar" in (workspace / "REPORT.md").read_text()
    assert gates["G1"]["pass"] is True
    assert datetime.fromisoformat(gates["G1"]["recorded_at"])
