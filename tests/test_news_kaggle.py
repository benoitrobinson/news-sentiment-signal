from pathlib import Path

import pytest

from sentiment_signal.data.news_kaggle import load_kaggle_news, midnight_share

CSV = """,title,date,stock
0,Apple beats estimates,2020-06-05 10:30:54-04:00,aapl
1,Apple beats estimates,2020-06-05 10:30:54-04:00,aapl
2,Tesla recalls cars,2020-06-05 00:00:00-04:00,TSLA
3,"Deal, finally, done",2020-01-10 16:05:00-05:00,MSFT
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "news.csv"
    path.write_text(text)
    return path


def test_loads_converts_to_new_york_and_drops_exact_duplicates(tmp_path):
    df = load_kaggle_news(_write(tmp_path, CSV), "title", "date", "stock")
    assert df.height == 3
    assert df["ticker"].to_list() == ["AAPL", "TSLA", "MSFT"]
    assert str(df.schema["ts"]) == "Datetime(time_unit='us', time_zone='America/New_York')"
    assert df["ts"][2].hour == 16  # -05:00 offset in January is New York winter time
    assert df["headline"][2] == "Deal, finally, done"


def test_midnight_share_counts_exact_midnight_stamps(tmp_path):
    assert midnight_share(_write(tmp_path, CSV), "date") == pytest.approx(0.25)


def test_refuses_timestamps_without_utc_offset(tmp_path):
    naive = ",title,date,stock\n0,x,2020-06-05 10:30:54,AAPL\n"
    with pytest.raises(ValueError, match="UTC offset"):
        load_kaggle_news(_write(tmp_path, naive), "title", "date", "stock")
