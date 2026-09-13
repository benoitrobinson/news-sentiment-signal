from datetime import date

import polars as pl

from sentiment_signal.splits import YearSplit, select_years, walk_forward_splits


def test_rolling_four_year_splits():
    assert walk_forward_splits([2013, 2020], train_years=4) == [
        YearSplit(2013, 2009, 2011, 2012, 2009, 2012),
        YearSplit(2020, 2016, 2018, 2019, 2016, 2019),
    ]


def test_embargo_drops_last_trading_date_of_block():
    cal = pl.Series("date", [date(2011, 12, 29), date(2011, 12, 30), date(2012, 1, 3)])
    panel = pl.DataFrame({"signal_date": cal.to_list(), "row_id": [0, 1, 2]})
    assert select_years(panel, 2011, 2011, cal, embargo=True)["row_id"].to_list() == [0]
    assert select_years(panel, 2011, 2011, cal, embargo=False)["row_id"].to_list() == [0, 1]
    assert select_years(panel, 2011, 2012, cal, embargo=False)["row_id"].to_list() == [0, 1, 2]


def test_embargo_on_a_multi_year_block_drops_only_its_final_trading_date():
    cal = pl.Series(
        "date", [date(2010, 12, 30), date(2010, 12, 31), date(2011, 12, 29), date(2011, 12, 30)]
    )
    panel = pl.DataFrame({"signal_date": cal.to_list(), "row_id": [0, 1, 2, 3]})
    assert select_years(panel, 2010, 2011, cal, embargo=True)["row_id"].to_list() == [0, 1, 2]


def test_embargo_uses_the_last_trading_date_in_the_block_when_its_final_year_is_empty():
    cal = pl.Series("date", [date(2010, 12, 30), date(2010, 12, 31), date(2012, 1, 3)])
    panel = pl.DataFrame({"signal_date": cal.to_list(), "row_id": [0, 1, 2]})
    assert select_years(panel, 2010, 2011, cal, embargo=True)["row_id"].to_list() == [0]
