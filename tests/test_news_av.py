import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from sentiment_signal.data.news_av import CAP, RateLimited, collect_av_news, load_av_news

KEY = "SECRET-KEY-123"
START = datetime(2022, 1, 3)


def _item(minute: int, ticker: str = "AAPL", relevance: str = "0.9") -> dict:
    return {
        "title": f"headline {minute}",
        "time_published": f"20220103T10{minute:02d}00",
        "ticker_sentiment": [{"ticker": ticker, "relevance_score": relevance}],
    }


def test_collects_windows_resumes_and_never_writes_or_logs_the_key(tmp_path, caplog):
    calls = []

    def fetch(params):
        calls.append(params["time_from"])
        if len(calls) == 2:
            return {"Information": "rate limit reached"}
        return {"feed": [_item(len(calls))]}

    caplog.set_level(logging.DEBUG)
    with pytest.raises(RateLimited):
        collect_av_news(START, START + timedelta(days=3), tmp_path, KEY, fetch=fetch)
    assert len(list(tmp_path.glob("*.json"))) == 1
    calls.clear()
    fetched = collect_av_news(
        START, START + timedelta(days=3), tmp_path, KEY, fetch=lambda p: {"feed": [_item(5)]}
    )
    assert fetched == 2  # the finished window is skipped on rerun
    assert KEY not in caplog.text
    assert all(KEY not in f.read_text() for f in tmp_path.glob("*.json"))


def test_capped_window_is_split_in_half_and_not_refetched_on_resume(tmp_path):
    spans = []

    def fetch(params):
        a = datetime.strptime(params["time_from"], "%Y%m%dT%H%M")
        b = datetime.strptime(params["time_to"], "%Y%m%dT%H%M")
        spans.append(b - a)
        return {"feed": [_item(1)] * (CAP if b - a > timedelta(hours=12) else 3)}

    collect_av_news(START, START + timedelta(days=1), tmp_path, KEY, fetch=fetch)
    assert spans == [timedelta(days=1), timedelta(hours=12), timedelta(hours=12)]
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert collect_av_news(START, START + timedelta(days=1), tmp_path, KEY, fetch=fetch) == 0
    assert len(spans) == 3


def test_window_still_capped_at_the_floor_is_kept_with_a_warning(tmp_path, caplog):
    fetched = collect_av_news(
        START,
        START + timedelta(minutes=2),
        tmp_path,
        KEY,
        fetch=lambda p: {"feed": [_item(1)] * CAP},
    )
    assert fetched == 1 and "still at the cap" in caplog.text


def test_loader_keeps_relevant_tickers_in_new_york_time(tmp_path):
    (tmp_path / "w.json").write_text(
        json.dumps([_item(1), _item(2, relevance="0.1"), _item(1), _item(3, ticker="MSFT")])
    )
    df = load_av_news(tmp_path, min_relevance=0.5).sort("ticker")
    assert df["ticker"].to_list() == ["AAPL", "MSFT"]
    assert df["ts"][0].hour == 10
    assert str(df.schema["ts"]) == "Datetime(time_unit='us', time_zone='America/New_York')"


def test_loader_resolves_new_york_dst_transitions_conservatively(tmp_path):
    items = [_item(0) | {"time_published": s} for s in ("20220313T023000", "20221106T013000")]
    (tmp_path / "w.json").write_text(json.dumps(items))
    df = load_av_news(tmp_path, min_relevance=0.5)
    # 02:30 never happened on 13 March 2022; 01:30 on 6 November happened twice: keep the later
    utc = df["ts"].dt.convert_time_zone("UTC").to_list()
    assert utc == [datetime(2022, 11, 6, 6, 30, tzinfo=UTC)]
