from datetime import date

import numpy as np
import polars as pl
import pytest
import torch

from sentiment_signal.embed import embed_rows, make_chrono_featurize, mean_pool, vintage_for


class FakeEmbedder:
    """Deterministic 4-dim 'embedding' = [len, vowels, 1, year-independent hash]; counts calls."""

    calls = 0

    def __init__(self, model_id: str):
        self.model_id = model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        FakeEmbedder.calls += 1
        return np.array(
            [[len(t), sum(c in "aeiou" for c in t), 1.0, (hash(t) % 97) / 97] for t in texts],
            dtype=np.float16,
        )


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signal_date": [date(2011, 5, 2), date(2011, 5, 2), date(2012, 5, 2), date(2012, 5, 3)],
            "headline": ["same text", "same text", "other", "third one"],
        }
    ).with_row_index("row_id")


def test_vintage_is_previous_year_end():
    assert vintage_for(2013) == "manelalab/chrono-bert-v1-20121231"


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]


def test_embed_rows_is_row_aligned_deduplicated_and_cached(tmp_path):
    FakeEmbedder.calls = 0
    panel = _panel()
    first = embed_rows(panel, "m/x", tmp_path, FakeEmbedder, dim=4)
    assert first.shape == (4, 4)
    np.testing.assert_array_equal(first[0], first[1])  # duplicate headline, same vector
    assert FakeEmbedder.calls == 2  # one call per year
    second = embed_rows(panel, "m/x", tmp_path, FakeEmbedder, dim=4)
    assert FakeEmbedder.calls == 2  # served from cache
    np.testing.assert_array_equal(first, second)


def test_embed_rows_refuses_a_cache_missing_headlines(tmp_path):
    panel = _panel()
    embed_rows(panel.filter(pl.col("row_id") != 3), "m/x", tmp_path, FakeEmbedder, dim=4)
    with pytest.raises(ValueError, match="cache"):
        embed_rows(panel, "m/x", tmp_path, FakeEmbedder, dim=4)


def test_chrono_featurize_uses_one_block_per_test_year(tmp_path):
    panel = _panel()
    featurize = make_chrono_featurize(panel, tmp_path, FakeEmbedder, train_years=1, dim=4)
    train = panel.filter(pl.col("signal_date").dt.year() == 2011)
    test = panel.filter(pl.col("signal_date").dt.year() == 2012)
    x_train, (x_test,) = featurize(train, [test], {"C": 1.0}, 2012)
    assert x_train.shape == (2, 4) and x_test.shape == (2, 4)
    assert x_train.dtype == np.float32
    assert (tmp_path / "manelalab__chrono-bert-v1-20111231").exists()
    # fit_predict scales dense features in place, so a second call must not see the first's edits
    before = x_train.copy()
    x_train *= 0
    x_again, _ = featurize(train, [test], {"C": 1.0}, 2012)
    np.testing.assert_array_equal(x_again, before)
