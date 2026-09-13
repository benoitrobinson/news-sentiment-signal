"""Headline timestamps -> the first trading-day signal they could have been used for."""

from datetime import date, time

import polars as pl

from sentiment_signal import config

PANEL_COLUMNS = [
    "row_id",
    "ticker",
    "ts",
    "headline",
    "signal_date",
    "ret_next_excess",
    "liquidity_tercile",
    "y",
]


def signal_date(
    ts: pl.Series,
    calendar: pl.Series,
    cutoff: time = config.CUTOFF,
    early_closes: tuple[date, ...] = config.EARLY_CLOSE_DATES,
    early_cutoff: time = config.EARLY_CUTOFF,
) -> pl.Series:
    """First calendar date d with ts strictly before d's cut-off, New York time. The cut-off is
    15:50, or 12:50 on NYSE early-close days: a headline after a 13:00 close waits a session."""
    cutoffs = (
        pl.DataFrame({"signal_date": calendar})
        .with_columns(
            cutoff_ts=pl.when(pl.col("signal_date").is_in(list(early_closes)))
            .then(pl.col("signal_date").dt.combine(early_cutoff))
            .otherwise(pl.col("signal_date").dt.combine(cutoff))
            .dt.replace_time_zone(config.TZ)
        )
        .sort("cutoff_ts")
    )
    left = pl.DataFrame({"ts": ts}).with_row_index("i").sort("ts")
    joined = left.join_asof(
        cutoffs,
        left_on="ts",
        right_on="cutoff_ts",
        strategy="forward",
        allow_exact_matches=False,
    )
    return joined.sort("i")["signal_date"]


def _finish(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.with_columns(y=(pl.col("ret_next_excess") > 0).cast(pl.Int8))
        .sort("signal_date", "ticker", "ts")
        .with_row_index("row_id")
        .select(PANEL_COLUMNS)
    )


def build_panel(news: pl.DataFrame, returns: pl.DataFrame, calendar: pl.Series) -> pl.DataFrame:
    eligible = returns.filter(pl.col("eligible")).select(
        "ticker", pl.col("date").alias("signal_date"), "ret_next_excess", "liquidity_tercile"
    )
    return _finish(
        news.with_columns(signal_date=signal_date(news["ts"], calendar))
        .drop_nulls("signal_date")
        .join(eligible, on=["ticker", "signal_date"], how="inner")
    )


def build_panel_same_day(
    news: pl.DataFrame, returns: pl.DataFrame, calendar: pl.Series
) -> pl.DataFrame:
    """The original repo's alignment: headline's own date joined to the return ENDING that date."""
    cal = pl.DataFrame({"date": calendar}).with_columns(end_date=pl.col("date").shift(-1))
    ending = (
        returns.filter(pl.col("eligible"))
        .join(cal, on="date", how="inner")
        .select(
            "ticker",
            pl.col("end_date").alias("signal_date"),
            "ret_next_excess",
            "liquidity_tercile",
        )
        .drop_nulls("signal_date")
    )
    return _finish(
        news.with_columns(signal_date=pl.col("ts").dt.date()).join(
            ending, on=["ticker", "signal_date"], how="inner"
        )
    )
