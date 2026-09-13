from datetime import date, timedelta

import numpy as np
import polars as pl

from sentiment_signal.baseline import tfidf_featurize, vader_scores
from sentiment_signal.model import shuffle_labels, standardize_inplace, walk_forward
from sentiment_signal.splits import YearSplit


def _panel() -> tuple[pl.DataFrame, pl.Series]:
    """2009-2013, 3 dates a year, 30 tickers a date; 'surge' headlines have positive returns."""
    rows, dates = [], []
    for year in range(2009, 2014):
        for k in range(3):
            d = date(year, 6, 1) + timedelta(days=k)
            dates.append(d)
            for i in range(30):
                good = i % 2 == 0
                word = "surge" if good else "plunge"
                rows.append(
                    {
                        "ticker": f"T{i}",
                        "signal_date": d,
                        "headline": f"shares {word} today",
                        "ret_next_excess": 0.01 if good else -0.01,
                        "liquidity_tercile": 1,
                    }
                )
    panel = (
        pl.DataFrame(rows)
        .with_columns(
            y=(pl.col("ret_next_excess") > 0).cast(pl.Int8),
            liquidity_tercile=pl.col("liquidity_tercile").cast(pl.Int8),
        )
        .with_row_index("row_id")
    )
    return panel, pl.Series("date", dates)


SPLIT = YearSplit(2013, 2009, 2011, 2012, 2009, 2012)


def test_tfidf_vocabulary_is_fitted_on_training_rows_only():
    panel, _ = _panel()
    train = panel.filter(pl.col("signal_date").dt.year() < 2013)
    # 'zebra' is in all 90 test-year headlines, far above min_df=5: a vocabulary fitted on
    # evaluation rows would keep it
    test = panel.filter(pl.col("signal_date").dt.year() == 2013).with_columns(
        headline=pl.lit("shares zebra today")
    )
    zebra_only = test.head(1).with_columns(headline=pl.lit("zebra"))
    _, (x_test, x_zebra) = tfidf_featurize(train, [test, zebra_only], {"ngram": (1, 1)}, 2013)
    assert x_test.nnz > 0  # 'shares' and 'today' are training words
    assert x_zebra.nnz == 0


def test_walk_forward_learns_signal_and_never_fits_on_test_year():
    panel, cal = _panel()
    fitted_years: list[set[int]] = []

    def spy(train, evals, params, test_year):
        fitted_years.append(set(train["signal_date"].dt.year().unique().to_list()))
        return tfidf_featurize(train, evals, params, test_year)

    wf = walk_forward(panel, cal, [SPLIT], spy, [{"ngram": (1, 1), "C": 1.0}], dense=False)
    assert all(2013 not in years for years in fitted_years)
    assert set(wf.scores["test_year"].to_list()) == {2013}
    good = wf.scores.filter(pl.col("ret_next_excess") > 0)["score"].mean()
    bad = wf.scores.filter(pl.col("ret_next_excess") < 0)["score"].mean()
    assert good > bad
    assert wf.validation.height == 1 and wf.validation["test_year"].item() == 2013


def test_walk_forward_refits_the_best_validation_config_on_dense_features():
    panel, cal = _panel()
    rng = np.random.default_rng(0)
    calls: list[tuple[str, list[int]]] = []

    def featurize(train, evals, params, test_year):
        calls.append((params["name"], sorted(train["signal_date"].dt.year().unique().to_list())))

        def features(rows):
            if params["name"] == "noise":
                return rng.normal(size=(rows.height, 1)).astype(np.float32)
            return (
                rows.select(pl.col("headline").str.contains("surge")).to_numpy().astype(np.float32)
            )

        return features(train), [features(e) for e in evals]

    # the best config is neither first nor last, so keeping grid[0] or the loop variable fails
    grid = [{"name": "noise", "C": 1.0}, {"name": "words", "C": 1.0}, {"name": "noise", "C": 1.0}]
    wf = walk_forward(panel, cal, [SPLIT], featurize, grid, dense=True)
    # selection fits exclude validation year 2012; the refit adds it and uses the best config
    assert calls == [
        ("noise", [2009, 2010, 2011]),
        ("words", [2009, 2010, 2011]),
        ("noise", [2009, 2010, 2011]),
        ("words", [2009, 2010, 2011, 2012]),
    ]
    ic = wf.validation.sort("config")["mean_ic"]
    assert ic[1] > max(ic[0], ic[2])


def test_walk_forward_null_fits_on_permuted_labels():
    panel, cal = _panel()
    changed: list[int] = []

    def spy(train, evals, params, test_year):
        original = train.select("row_id").join(panel.select("row_id", "y"), on="row_id")
        changed.append(int((train["y"] != original["y"]).sum()))
        return tfidf_featurize(train, evals, params, test_year)

    grid = [{"ngram": (1, 1), "C": 1.0}]
    walk_forward(panel, cal, [SPLIT], spy, grid, dense=False, shuffle_seed=0)
    assert len(changed) == 2 and all(n > 0 for n in changed)


def test_shuffle_keeps_label_counts_and_breaks_the_feature_label_link():
    panel, _ = _panel()
    shuffled = shuffle_labels(panel, seed=0)
    assert shuffled["y"].sum() == panel["y"].sum()
    # every date has 15 up labels; only a permutation across dates changes those counts
    assert shuffled.group_by("signal_date").agg(pl.col("y").sum())["y"].n_unique() > 1
    word_up = panel["headline"].str.contains("surge").cast(pl.Int8).to_numpy()
    assert np.corrcoef(word_up, panel["y"].to_numpy())[0, 1] > 0.95
    assert abs(np.corrcoef(word_up, shuffled["y"].to_numpy())[0, 1]) < 0.15


def test_standardize_inplace_matches_standard_scaler_without_copies():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    train = rng.normal(3.0, 2.0, (1_001, 8)).astype(np.float32)
    train[:, 5] = 7.0  # constant column: scale 1, like StandardScaler
    test = rng.normal(3.0, 2.0, (50, 8)).astype(np.float32)
    scaler = StandardScaler().fit(train)
    expected_train, expected_test = scaler.transform(train), scaler.transform(test)
    buffer_address = train.__array_interface__["data"][0]
    standardize_inplace(train, [test], chunk=100)
    assert train.__array_interface__["data"][0] == buffer_address and train.dtype == np.float32
    np.testing.assert_allclose(train, expected_train, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(test, expected_test, rtol=1e-4, atol=1e-4)


def test_vader_scores_positive_above_negative():
    panel = pl.DataFrame(
        {
            "ticker": ["A", "B"],
            "signal_date": [date(2009, 6, 1)] * 2,
            "headline": ["great results, shares soar", "terrible losses, shares crash"],
            "ret_next_excess": [0.01, -0.01],
            "liquidity_tercile": pl.Series([1, 1], dtype=pl.Int8),
        }
    )
    scores = vader_scores(panel).sort("ticker")
    assert scores["score"][0] > 0 > scores["score"][1]
