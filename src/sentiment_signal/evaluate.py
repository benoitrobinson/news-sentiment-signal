"""Pre-registered evaluation (spec section 5). Inputs are stock-day score frames with columns
ticker, signal_date, score, ret_next_excess, liquidity_tercile."""

import math
from collections.abc import Sequence

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from sentiment_signal import config

EULER_GAMMA = 0.5772156649015329


def daily_ic(
    scores: pl.DataFrame,
    ret_col: str = "ret_next_excess",
    min_names: int = config.MIN_NAMES_PER_DAY,
) -> pl.DataFrame:
    return (
        scores.drop_nulls(["score", ret_col])
        .group_by("signal_date")
        .agg(ic=pl.corr("score", ret_col, method="spearman"), n=pl.len())
        .filter((pl.col("n") >= min_names) & pl.col("ic").is_not_nan())
        .sort("signal_date")
    )


def newey_west_t(x: Sequence[float] | np.ndarray, lags: int = config.NW_LAGS) -> float:
    x = np.asarray(x, dtype=float)
    fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0] / fit.bse[0])


def _quintiles(scores: pl.DataFrame, min_names: int) -> pl.DataFrame:
    """Ordinal rank r of n each day, quintile ceil(5r/n). Tied scores are ordered by a hash of
    ticker and date, not by ticker, so ties cannot hold the same alphabetical basket every day; a
    day whose scores are all equal has no ranking and is skipped, as daily_ic skips it."""
    return (
        scores.drop_nulls(["score", "ret_next_excess"])
        .with_columns(
            n=pl.len().over("signal_date"),
            tie=pl.concat_str("ticker", pl.col("signal_date").cast(pl.String)).hash(seed=0),
        )
        .filter((pl.col("n") >= min_names) & (pl.col("score").n_unique().over("signal_date") > 1))
        .sort("signal_date", "score", "tie")
        .with_columns(
            q=(5 * pl.int_range(1, pl.len() + 1).over("signal_date") / pl.col("n"))
            .ceil()
            .cast(pl.Int8)
        )
        .drop("tie")
    )


def long_short(
    scores: pl.DataFrame,
    min_names: int = config.MIN_NAMES_PER_DAY,
    cost: float = config.COST_PER_UNIT_TURNOVER,
) -> pl.DataFrame:
    legs = (
        _quintiles(scores, min_names)
        .filter(pl.col("q").is_in([1, 5]))
        .with_columns(leg_n=pl.len().over("signal_date", "q"))
        .with_columns(
            w=pl.when(pl.col("q") == 5)
            .then(1.0 / pl.col("leg_n"))
            .otherwise(-1.0 / pl.col("leg_n"))
        )
    )
    gross = legs.group_by("signal_date").agg(gross=(pl.col("w") * pl.col("ret_next_excess")).sum())
    dates = (
        gross.select("signal_date")
        .sort("signal_date")
        .with_columns(prev=pl.col("signal_date").shift(1))
    )
    weights = legs.select("signal_date", "ticker", "w")
    previous = (
        weights.rename({"signal_date": "prev", "w": "w_prev"})
        .join(dates.drop_nulls("prev"), on="prev")
        .select("signal_date", "ticker", "w_prev")
    )
    turnover = (
        weights.join(previous, on=["signal_date", "ticker"], how="full", coalesce=True)
        .with_columns(pl.col("w").fill_null(0.0), pl.col("w_prev").fill_null(0.0))
        .group_by("signal_date")
        .agg(turnover=(pl.col("w") - pl.col("w_prev")).abs().sum())
    )
    return (
        gross.join(turnover, on="signal_date")
        .with_columns(cost=cost * pl.col("turnover"))
        .with_columns(net=pl.col("gross") - pl.col("cost"))
        .sort("signal_date")
    )


def annualised_sharpe(returns: Sequence[float] | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(config.TRADING_DAYS)) if sd > 0 else float("nan")


def lag_ic(
    scores: pl.DataFrame,
    returns: pl.DataFrame,
    calendar: pl.Series,
    min_names: int = config.MIN_NAMES_PER_DAY,
) -> pl.DataFrame:
    cal = pl.DataFrame({"signal_date": calendar}).with_columns(
        next_date=pl.col("signal_date").shift(-1)
    )
    following = returns.filter(pl.col("eligible")).select(
        "ticker", pl.col("date").alias("next_date"), pl.col("ret_next_excess").alias("ret_lag")
    )
    lagged = (
        scores.select("ticker", "signal_date", "score")
        .join(cal, on="signal_date")
        .join(following, on=["ticker", "next_date"])
    )
    return daily_ic(lagged, ret_col="ret_lag", min_names=min_names)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Bailey & Lopez de Prado (2014): expected maximum Sharpe of n unskilled trials."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: Sequence[float] | np.ndarray, n_trials: int, sr_var: float) -> float:
    """Probability that the per-period Sharpe beats the expected max of n unskilled trials."""
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    sr0 = expected_max_sharpe(n_trials, sr_var)
    denom = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom))


def summarise(scores: pl.DataFrame, min_names: int = config.MIN_NAMES_PER_DAY) -> dict:
    ic = daily_ic(scores, min_names=min_names)
    ls = long_short(scores, min_names=min_names)
    ic_values = ic["ic"].to_numpy()
    return {
        "days": int(ic.height),
        "mean_ic": float(ic_values.mean()) if ic.height else float("nan"),
        "ic_t": newey_west_t(ic_values) if ic.height > config.NW_LAGS + 1 else float("nan"),
        "net_sharpe": annualised_sharpe(ls["net"].to_numpy()),
        "gross_sharpe": annualised_sharpe(ls["gross"].to_numpy()),
        "mean_turnover": float(ls["turnover"].mean()) if ls.height else float("nan"),
    }


def by_tercile(scores: pl.DataFrame, min_names: int = config.MIN_NAMES_PER_DAY) -> list[dict]:
    return [
        {"tercile": int(t), **summarise(scores.filter(pl.col("liquidity_tercile") == t), min_names)}
        for t in sorted(scores["liquidity_tercile"].drop_nulls().unique().to_list())
    ]


def event_study(
    scores: pl.DataFrame,
    returns: pl.DataFrame,
    calendar: pl.Series,
    offsets: Sequence[int] = (-1, 0, 1, 2, 3, 4, 5),
    min_names: int = config.MIN_NAMES_PER_DAY,
) -> pl.DataFrame:
    """Mean cumulative excess return by score quintile. Offset 0 = the traded next-day return."""
    cal = pl.DataFrame({"date": calendar}).with_row_index("idx")
    base = (
        _quintiles(scores, min_names)
        .select("ticker", "signal_date", "q")
        .join(cal.rename({"date": "signal_date"}), on="signal_date")
    )
    rets = (
        returns.filter(pl.col("eligible"))
        .select("ticker", "date", pl.col("ret_next_excess").alias("r"))
        .join(cal, on="date")
        .select("ticker", "idx", "r")
    )
    per_offset = pl.concat(
        base.with_columns(idx=pl.col("idx") + k, offset=pl.lit(k))
        .join(rets, on=["ticker", "idx"])
        .select("q", "offset", "r")
        for k in offsets
    )
    return (
        per_offset.group_by("q", "offset")
        .agg(mean_r=pl.col("r").mean())
        .sort("q", "offset")
        .with_columns(cum_r=pl.col("mean_r").cum_sum().over("q"))
    )


def publish_bar(
    ic_t: float, net_sharpe: float, mean_ic: float, mean_lag_ic: float, holdout_mean_ic: float
) -> dict:
    conditions = {
        "1_ic_t_ge_2": (ic_t, ic_t >= 2.0),
        "2_net_sharpe_ge_0_5": (net_sharpe, net_sharpe >= 0.5),
        # a lag check is only meaningful for a positive signal; a non-positive mean IC fails it
        "3_lag_ic_le_half": (mean_lag_ic, mean_ic > 0 and mean_lag_ic <= 0.5 * mean_ic),
        "4_holdout_ic_gt_0": (holdout_mean_ic, holdout_mean_ic > 0),
    }
    return {
        "conditions": {k: {"value": float(v), "pass": bool(p)} for k, (v, p) in conditions.items()},
        "publish": all(bool(p) for _, p in conditions.values()),
    }
