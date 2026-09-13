"""Alpha Vantage NEWS_SENTIMENT: resumable market-wide collection for the holdout, and a loader."""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import polars as pl

from sentiment_signal import config

log = logging.getLogger(__name__)
API = "https://www.alphavantage.co/query"
CAP = 1000
MIN_WINDOW = timedelta(minutes=2)


class RateLimited(RuntimeError):
    """The API replied without a feed (limit reached or error). Re-run later to resume."""


def http_fetch(params: dict[str, str]) -> dict:
    with urlopen(f"{API}?{urlencode(params)}", timeout=60) as response:
        return json.load(response)


def _stamp(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M")


def collect_av_news(
    start: datetime,
    end: datetime,
    out_dir: Path,
    api_key: str,
    *,
    fetch: Callable[[dict[str, str]], dict] = http_fetch,
    window: timedelta = timedelta(days=1),
    pause_seconds: float = 0.0,
) -> int:
    """Fetch [start, end) window by window; one checkpoint file per finished window.
    Returns the number of windows fetched in this run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pending, t = [], start
    while t < end:
        pending.append((t, min(t + window, end)))
        t += window
    fetched = 0
    while pending:
        a, b = pending.pop(0)
        checkpoint = out_dir / f"{_stamp(a)}_{_stamp(b)}.json"
        if checkpoint.exists():
            continue
        params = {
            "function": "NEWS_SENTIMENT",
            "time_from": _stamp(a),
            "time_to": _stamp(b),
            "limit": str(CAP),
            "sort": "EARLIEST",
            "apikey": api_key,
        }
        reply = fetch(params)
        if "feed" not in reply:
            log.warning("no feed for %s-%s; reply keys %s", _stamp(a), _stamp(b), sorted(reply))
            raise RateLimited(f"stopped at window {_stamp(a)}-{_stamp(b)}")
        if len(reply["feed"]) >= CAP and b - a > MIN_WINDOW:
            middle = a + (b - a) / 2
            pending[:0] = [(a, middle), (middle, b)]
            continue
        checkpoint.write_text(json.dumps(reply["feed"]))
        fetched += 1
        if pause_seconds:
            time.sleep(pause_seconds)
    return fetched


def load_av_news(out_dir: Path, min_relevance: float = config.AV_MIN_RELEVANCE) -> pl.DataFrame:
    rows = [
        {"ticker": s["ticker"], "time_published": item["time_published"], "headline": item["title"]}
        for f in sorted(out_dir.glob("*.json"))
        for item in json.loads(f.read_text())
        for s in item.get("ticker_sentiment", [])
        if float(s["relevance_score"]) >= min_relevance
    ]
    schema = {"ticker": pl.String, "time_published": pl.String, "headline": pl.String}
    return (
        pl.DataFrame(rows, schema=schema)
        .with_columns(
            ts=pl.col("time_published")
            .str.to_datetime("%Y%m%dT%H%M%S")
            .dt.replace_time_zone(config.AV_ASSUMED_TZ)
            .dt.convert_time_zone(config.TZ)
        )
        .select("ticker", "ts", "headline")
        .unique(subset=["ticker", "ts", "headline"], maintain_order=True)
    )
