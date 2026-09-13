"""Rolling walk-forward year splits with a one-trading-day embargo at the end of training blocks."""

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from sentiment_signal import config


@dataclass(frozen=True)
class YearSplit:
    test_year: int
    select_first: int
    select_last: int
    validate_year: int
    refit_first: int
    refit_last: int


def walk_forward_splits(
    test_years: Iterable[int], train_years: int = config.TRAIN_YEARS
) -> list[YearSplit]:
    return [YearSplit(y, y - train_years, y - 2, y - 1, y - train_years, y - 1) for y in test_years]


def select_years(
    panel: pl.DataFrame, first: int, last: int, calendar: pl.Series, *, embargo: bool
) -> pl.DataFrame:
    rows = panel.filter(pl.col("signal_date").dt.year().is_between(first, last))
    if not embargo:
        return rows
    # the label of the block's last trading date is a return that ends in the next block; with no
    # trading date in the block there are no rows, so there is nothing to embargo
    last_day = calendar.filter(calendar.dt.year().is_between(first, last)).max()
    return rows if last_day is None else rows.filter(pl.col("signal_date") != last_day)
