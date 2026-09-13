from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from sentiment_signal import config
from sentiment_signal.align import build_panel, build_panel_same_day, signal_date

NY = ZoneInfo(config.TZ)
# Fri 8 Mar 2019, (weekend), Mon 11 Mar 2019 is the first day of New York summer time, Tue 12 Mar
CAL = pl.Series("date", [date(2019, 3, 7), date(2019, 3, 8), date(2019, 3, 11), date(2019, 3, 12)])


def _ts(*stamps: datetime) -> pl.Series:
    return pl.Series("ts", list(stamps), dtype=pl.Datetime("us", config.TZ))


def test_cutoff_boundaries_weekend_and_daylight_saving():
    ts = _ts(
        datetime(2019, 3, 8, 15, 49, tzinfo=NY),  # before cut-off -> same day
        datetime(2019, 3, 8, 15, 50, tzinfo=NY),  # exactly at cut-off -> next trading day
        datetime(2019, 3, 8, 16, 30, tzinfo=NY),  # Friday after close -> Monday
        datetime(2019, 3, 11, 15, 49, tzinfo=NY),  # first summer-time day, before cut-off
        datetime(2019, 3, 11, 15, 51, tzinfo=NY),
    )
    assert signal_date(ts, CAL).to_list() == [
        date(2019, 3, 8),
        date(2019, 3, 11),
        date(2019, 3, 11),
        date(2019, 3, 11),
        date(2019, 3, 12),
    ]


def test_early_close_day_moves_the_cutoff_to_12_50():
    # Fri 29 Nov 2019 closed at 13:00; a 14:00 headline must wait for Monday 2 Dec
    cal = pl.Series("date", [date(2019, 11, 27), date(2019, 11, 29), date(2019, 12, 2)])
    assert date(2019, 11, 29) in config.EARLY_CLOSE_DATES
    ts = _ts(
        datetime(2019, 11, 29, 12, 49, tzinfo=NY),
        datetime(2019, 11, 29, 12, 50, tzinfo=NY),
        datetime(2019, 11, 29, 14, 0, tzinfo=NY),
    )
    assert signal_date(ts, cal).to_list() == [
        date(2019, 11, 29),
        date(2019, 12, 2),
        date(2019, 12, 2),
    ]


def test_date_missing_from_calendar_is_skipped():
    cal = pl.Series("date", [date(2019, 3, 7), date(2019, 3, 11)])  # 8 March absent
    ts = _ts(datetime(2019, 3, 7, 17, 0, tzinfo=NY))
    assert signal_date(ts, cal).to_list() == [date(2019, 3, 11)]


@settings(max_examples=300, deadline=None)
@given(st.integers(min_value=0, max_value=6 * 24 * 60 - 1))
def test_property_cutoff_is_strictly_after_timestamp(minutes):
    # sources give instants with UTC offsets, so generate instants (this spans the 10 March DST gap)
    instant = datetime(2019, 3, 7, 5, 0, tzinfo=UTC) + timedelta(minutes=minutes)
    ts = pl.Series("ts", [instant], dtype=pl.Datetime("us", "UTC")).dt.convert_time_zone(config.TZ)
    d = signal_date(ts, CAL).item()
    if d is not None:
        assert datetime.combine(d, config.CUTOFF, tzinfo=NY) > instant


def _returns() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A"] * 4,
            "date": CAL.to_list(),
            "ret_next": [0.01, 0.02, 0.03, 0.04],
            "ret_next_excess": [0.001, 0.002, -0.003, 0.004],
            "eligible": [True, True, True, True],
            "med_dollar_vol": [1.0] * 4,
            "liquidity_tercile": pl.Series([1, 1, 1, 1], dtype=pl.Int8),
        }
    )


def _news() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "ts": _ts(
                datetime(2019, 3, 8, 11, 0, tzinfo=NY), datetime(2019, 3, 8, 17, 0, tzinfo=NY)
            ),
            "headline": ["morning news", "evening news"],
        }
    )


def test_panel_joins_next_day_return_of_signal_date():
    panel = build_panel(_news(), _returns(), CAL)
    assert panel["signal_date"].to_list() == [date(2019, 3, 8), date(2019, 3, 11)]
    assert panel["ret_next_excess"].to_list() == [0.002, -0.003]
    assert panel["y"].to_list() == [1, 0]
    assert panel["row_id"].to_list() == [0, 1]


def test_same_day_panel_reproduces_the_original_lookahead_join():
    # both headlines dated 8 March get the return that ENDS on 8 March (7 -> 8 March)
    panel = build_panel_same_day(_news(), _returns(), CAL)
    assert panel["signal_date"].to_list() == [date(2019, 3, 8), date(2019, 3, 8)]
    assert panel["ret_next_excess"].to_list() == [0.001, 0.001]
