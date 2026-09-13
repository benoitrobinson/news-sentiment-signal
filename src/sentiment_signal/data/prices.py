"""FNSPID daily prices -> trading calendar and next-day excess returns on a liquid universe."""

import zipfile
from pathlib import Path

import polars as pl

from sentiment_signal import config


def extract_prices(zip_path: Path, out_dir: Path) -> int:
    """Extract full_history/*.csv, skipping macOS metadata. Returns the number of CSVs written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.filename.startswith("__MACOSX/") or not info.filename.endswith(".csv"):
                continue
            with z.open(info) as src, (out_dir / Path(info.filename).name).open("wb") as dst:
                dst.write(src.read())
            written += 1
    return written


def load_prices(csv_dir: Path) -> pl.LazyFrame:
    """FNSPID files come in two column orders (volume second or last), so a single glob scan fails
    with 'schema names differ'. Each file is scanned on its own and its columns selected by name."""
    frames = [
        pl.scan_csv(path, infer_schema=False).select(
            pl.lit(path.stem).alias("ticker"),
            pl.col("date").str.to_date("%Y-%m-%d", strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("adj close").cast(pl.Float64, strict=False).alias("adj_close"),
            pl.col("volume").cast(pl.Float64, strict=False),
        )
        for path in sorted(csv_dir.glob("*.csv"))
    ]
    return pl.concat(frames).drop_nulls()


def trading_calendar(
    prices: pl.LazyFrame, min_tickers: int = config.CALENDAR_MIN_TICKERS
) -> pl.Series:
    return (
        prices.group_by("date")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_tickers)
        .sort("date")
        .collect()["date"]
    )


def build_returns(
    prices: pl.LazyFrame,
    calendar: pl.Series,
    *,
    min_price: float = config.MIN_PRICE,
    universe_size: int = config.UNIVERSE_SIZE,
    window: int = config.DOLLAR_VOLUME_WINDOW,
) -> pl.DataFrame:
    """One row per ticker-date on the calendar. Eligibility on date d uses data through d-1 only."""
    cal = pl.DataFrame({"date": calendar}).with_columns(
        cal_prev=pl.col("date").shift(1), cal_next=pl.col("date").shift(-1)
    )
    base = (
        prices.join(cal.lazy(), on="date", how="inner")
        .sort("ticker", "date")
        .with_columns(dollar_vol=pl.col("close") * pl.col("volume"))
        .with_columns(
            prev_date=pl.col("date").shift(1).over("ticker"),
            prev_close=pl.col("close").shift(1).over("ticker"),
            med_dollar_vol=pl.col("dollar_vol").rolling_median(window).shift(1).over("ticker"),
            next_date=pl.col("date").shift(-1).over("ticker"),
            next_adj=pl.col("adj_close").shift(-1).over("ticker"),
        )
        .with_columns(
            ret_next=pl.when(pl.col("next_date") == pl.col("cal_next")).then(
                pl.col("next_adj") / pl.col("adj_close") - 1.0
            ),
            base_ok=(
                (pl.col("prev_date") == pl.col("cal_prev"))
                & (pl.col("prev_close") >= min_price)
                & pl.col("med_dollar_vol").is_not_null()
            ).fill_null(False),
        )
        .with_columns(
            dv_rank=pl.when(pl.col("base_ok"))
            .then(pl.col("med_dollar_vol"))
            .rank(method="ordinal", descending=True)
            .over("date")
        )
        .with_columns(
            eligible=(
                pl.col("base_ok")
                & (pl.col("dv_rank") <= universe_size)
                & pl.col("ret_next").is_not_null()
            ).fill_null(False)
        )
        .collect()
    )
    market = (
        base.filter(pl.col("eligible"))
        .group_by("date")
        .agg(mkt=pl.col("ret_next").mean(), n_elig=pl.len())
    )
    terciles = (
        base.filter(pl.col("eligible"))
        .join(market, on="date")
        .with_columns(
            r=pl.col("med_dollar_vol").rank(method="ordinal", descending=True).over("date")
        )
        .select(
            "ticker",
            "date",
            liquidity_tercile=(3 * pl.col("r") / pl.col("n_elig")).ceil().cast(pl.Int8),
        )
    )
    return (
        base.join(market, on="date", how="left")
        .with_columns(
            ret_next_excess=pl.when(pl.col("eligible")).then(pl.col("ret_next") - pl.col("mkt"))
        )
        .join(terciles, on=["ticker", "date"], how="left")
        .select(
            "ticker",
            "date",
            "ret_next",
            "ret_next_excess",
            "eligible",
            "med_dollar_vol",
            "liquidity_tercile",
        )
        .sort("ticker", "date")
    )
