"""Walk-forward fitting of a logistic head over any featurizer (spec section 4)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from sentiment_signal.evaluate import daily_ic, long_short
from sentiment_signal.splits import YearSplit, select_years

# (train_rows, [eval_rows, ...], params, test_year) -> (X_train, [X_eval, ...])
Featurize = Callable[[pl.DataFrame, list[pl.DataFrame], dict, int], tuple[Any, list[Any]]]


@dataclass
class WalkForward:
    scores: (
        pl.DataFrame
    )  # ticker, signal_date, score, ret_next_excess, liquidity_tercile, test_year
    validation: pl.DataFrame  # test_year, config, mean_ic
    validation_returns: pl.DataFrame  # test_year, config, signal_date, net


def stock_day_scores(rows: pl.DataFrame, proba: np.ndarray) -> pl.DataFrame:
    return (
        rows.with_columns(p=pl.Series(proba, dtype=pl.Float64))
        .group_by("ticker", "signal_date")
        .agg(
            score=pl.col("p").mean(),
            ret_next_excess=pl.col("ret_next_excess").first(),
            liquidity_tercile=pl.col("liquidity_tercile").first(),
        )
        .sort("signal_date", "ticker")
    )


def shuffle_labels(rows: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Permute labels across ALL rows. A within-date shuffle keeps each date's up-count, and a
    genuine signal survives through that composition, tripping the leak check on real signal."""
    permuted = np.random.default_rng(seed).permutation(rows["y"].to_numpy())
    return rows.with_columns(y=pl.Series(permuted, dtype=rows.schema["y"]))


def standardize_inplace(
    train: np.ndarray, others: Sequence[np.ndarray], chunk: int = 20_000
) -> None:
    """StandardScaler's transform, in place and in chunks. StandardScaler itself needs about 2.25x
    the matrix in float64 working copies: too much for 5 years of embeddings on an 8 GB machine."""
    mean = train.mean(axis=0, dtype=np.float64)
    sq = np.zeros(train.shape[1])
    for start in range(0, len(train), chunk):
        sq += np.square(train[start : start + chunk].astype(np.float64) - mean).sum(axis=0)
    scale = np.sqrt(sq / len(train))
    scale[scale == 0] = 1.0
    for array in (train, *others):
        array -= mean.astype(array.dtype)
        array /= scale.astype(array.dtype)


def fit_predict(
    featurize: Featurize,
    train: pl.DataFrame,
    evals: list[pl.DataFrame],
    params: dict,
    test_year: int,
    *,
    dense: bool,
) -> list[np.ndarray]:
    x_train, x_evals = featurize(train, evals, params, test_year)
    if dense:  # featurizers return fresh arrays, so scaling in place is safe
        standardize_inplace(x_train, x_evals)
    model = LogisticRegression(C=params["C"], max_iter=1000)
    model.fit(x_train, train["y"].to_numpy())
    return [model.predict_proba(x)[:, 1] for x in x_evals]


def walk_forward(
    panel: pl.DataFrame,
    calendar: pl.Series,
    splits: Sequence[YearSplit],
    featurize: Featurize,
    grid: Sequence[dict],
    *,
    dense: bool,
    shuffle_seed: int | None = None,
) -> WalkForward:
    scores, validation, validation_returns = [], [], []
    for sp in splits:
        train = select_years(panel, sp.select_first, sp.select_last, calendar, embargo=True)
        valid = select_years(panel, sp.validate_year, sp.validate_year, calendar, embargo=False)
        refit = select_years(panel, sp.refit_first, sp.refit_last, calendar, embargo=True)
        test = select_years(panel, sp.test_year, sp.test_year, calendar, embargo=False)
        if shuffle_seed is not None:
            train = shuffle_labels(train, shuffle_seed)
            refit = shuffle_labels(refit, shuffle_seed)
        best, best_ic = grid[0], -np.inf
        for i, params in enumerate(grid):
            (p_valid,) = fit_predict(featurize, train, [valid], params, sp.test_year, dense=dense)
            sd = stock_day_scores(valid, p_valid)
            ic = daily_ic(sd)["ic"]
            mean_ic = float(ic.mean()) if ic.len() else float("nan")
            validation.append({"test_year": sp.test_year, "config": i, "mean_ic": mean_ic})
            validation_returns.append(
                long_short(sd)
                .select("signal_date", "net")
                .with_columns(test_year=pl.lit(sp.test_year), config=pl.lit(i))
            )
            if mean_ic > best_ic:
                best, best_ic = params, mean_ic
        (p_test,) = fit_predict(featurize, refit, [test], best, sp.test_year, dense=dense)
        scores.append(stock_day_scores(test, p_test).with_columns(test_year=pl.lit(sp.test_year)))
    return WalkForward(
        scores=pl.concat(scores),
        validation=pl.DataFrame(validation),
        validation_returns=pl.concat(validation_returns),
    )
