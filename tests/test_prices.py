import zipfile
from datetime import date, timedelta

import polars as pl
import pytest

from sentiment_signal.data.prices import (
    build_returns,
    extract_prices,
    load_prices,
    trading_calendar,
)

DAYS = [date(2020, 1, 1) + timedelta(days=i) for i in range(25)]


def _prices(rows: list[tuple[str, date, float, float, float]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows, schema=["ticker", "date", "close", "adj_close", "volume"], orient="row"
    )


def _fixture() -> pl.LazyFrame:
    rows = []
    for i, d in enumerate(DAYS):
        rows.append(("A", d, 10.0, 100.0 * 1.01**i, 1000.0))  # dollar volume 10k, +1%/day
        rows.append(("B", d, 20.0, 50.0, 1000.0))  # dollar volume 20k, flat
        rows.append(("C", d, 4.0, 4.0, 1e6))  # huge volume but price < $5
    return _prices(rows)


def test_calendar_keeps_dates_with_enough_tickers():
    prices = _prices([("A", DAYS[0], 1, 1, 1), ("B", DAYS[0], 1, 1, 1), ("A", DAYS[1], 1, 1, 1)])
    assert trading_calendar(prices, min_tickers=2).to_list() == [DAYS[0]]


def test_eligibility_uses_prior_day_data_and_excess_is_vs_eligible_mean():
    prices = _fixture()
    out = build_returns(prices, trading_calendar(prices, min_tickers=1), universe_size=2, window=21)
    elig = out.filter(pl.col("eligible"))
    # rolling 21-day median needs rows 0..20, shifted by one day -> first eligible row is day 21;
    # the last day has no next price -> not eligible
    assert sorted(elig.filter(pl.col("ticker") == "A")["date"].to_list()) == DAYS[21:24]
    assert elig.filter(pl.col("ticker") == "C").height == 0
    a = elig.filter((pl.col("ticker") == "A") & (pl.col("date") == DAYS[21])).row(0, named=True)
    assert a["ret_next"] == pytest.approx(0.01)
    assert a["ret_next_excess"] == pytest.approx(0.005)  # mean of A (+1%) and B (0%)
    assert a["med_dollar_vol"] == pytest.approx(10_000.0)
    assert a["liquidity_tercile"] == 3  # B is more liquid: rank 2 of 2 -> ceil(3*2/2) = 3


def test_universe_size_keeps_most_liquid():
    prices = _fixture()
    out = build_returns(prices, trading_calendar(prices, min_tickers=1), universe_size=1, window=21)
    assert set(out.filter(pl.col("eligible"))["ticker"].to_list()) == {"B"}


def test_ticker_skipping_a_calendar_date_gets_null_return_and_is_not_eligible_after_gap():
    rows = [r for r in _fixture().collect().iter_rows() if not (r[0] == "A" and r[1] == DAYS[22])]
    prices = _prices(rows)
    out = build_returns(prices, trading_calendar(prices, min_tickers=1), universe_size=2, window=21)
    a = out.filter(pl.col("ticker") == "A").sort("date")
    assert a.filter(pl.col("date") == DAYS[21])["ret_next"].item() is None
    assert not a.filter(pl.col("date") == DAYS[23])["eligible"].item()


# the real FNSPID zip uses both column orders: 5,705 files one way, 1,988 the other
HEADERS = ("date,volume,open,high,low,close,adj close", "date,open,high,low,close,adj close,volume")


def test_extract_and_load_both_fnspid_column_orders(tmp_path):
    with zipfile.ZipFile(tmp_path / "prices.zip", "w") as z:
        z.writestr(
            "full_history/AAA.csv",
            f"{HEADERS[0]}\n2020-01-02,1000,10,11,9,10.5,10.4\n2020-01-03,,10,11,9,10.6,10.5\n",
        )
        z.writestr("full_history/BBB.csv", f"{HEADERS[1]}\n2020-01-02,20,21,19,20.5,20.1,500\n")
        z.writestr("__MACOSX/full_history/._AAA.csv", "resource fork junk")
    assert extract_prices(tmp_path / "prices.zip", tmp_path / "csv") == 2
    prices = load_prices(tmp_path / "csv").collect().sort("ticker", "date")
    assert prices.columns == ["ticker", "date", "close", "adj_close", "volume"]
    # AAA's blank-volume row is dropped; BBB's volume is read from its last column
    assert prices.rows() == [
        ("AAA", date(2020, 1, 2), 10.5, 10.4, 1000.0),
        ("BBB", date(2020, 1, 2), 20.5, 20.1, 500.0),
    ]


def test_bad_ticks_leave_no_next_day_return():
    adj, vol = [100.0] * 25, [1000.0] * 25
    vol[5] = 0.0  # no trade: neither the return into day 5 nor out of it is real
    adj[10] = 0.0  # a price that is not positive
    adj[12] = adj[13] = -50.0  # negative on consecutive days: the ratio alone would look normal
    adj[15] = 0.2  # a placeholder tick between normal days
    adj[19] = 450.0  # beyond 4x up, then beyond 4x down
    adj[22] = 400.0  # exactly 4x up and back: the band is closed, so both returns are kept
    prices = _prices([("A", d, 10.0, a, v) for d, a, v in zip(DAYS, adj, vol, strict=True)])
    cal = trading_calendar(prices, min_tickers=1)
    out = build_returns(prices, cal, universe_size=1, window=3, splice_dates=(DAYS[7],))
    missing = {i for i, r in enumerate(out.sort("date")["ret_next"].to_list()) if r is None}
    # day 7 is a price-vintage splice: its next-day return is dropped even though prices look fine
    assert missing == {4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 24}
