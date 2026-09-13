"""Kaggle 'Daily Financial News for 6000+ Stocks' -> ticker, ts (New York time), headline."""

from pathlib import Path

import polars as pl

from sentiment_signal import config

TS_FORMAT = "%Y-%m-%d %H:%M:%S%z"


def midnight_share(path: Path, ts_col: str) -> float:
    stamps = pl.scan_csv(path, infer_schema=False).select(ts_col).collect()[ts_col]
    return float(stamps.str.contains(" 00:00:00").mean())


def load_kaggle_news(path: Path, headline_col: str, ts_col: str, ticker_col: str) -> pl.DataFrame:
    df = (
        pl.scan_csv(path, infer_schema=False)
        .select(
            pl.col(ticker_col).str.strip_chars().str.to_uppercase().alias("ticker"),
            pl.col(ts_col)
            .str.to_datetime(TS_FORMAT, strict=False)
            .dt.convert_time_zone(config.TZ)
            .alias("ts"),
            pl.col(headline_col).str.strip_chars().alias("headline"),
        )
        .collect()
    )
    parsed = df["ts"].is_not_null().mean()
    if parsed is None or parsed < 0.99:
        raise ValueError(f"{parsed} of timestamps carry a parseable UTC offset; gate G1 fails")
    return df.drop_nulls().unique(subset=["ticker", "ts", "headline"], maintain_order=True)
