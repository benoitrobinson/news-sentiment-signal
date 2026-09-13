import math
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from sentiment_signal.evaluate import (
    annualised_sharpe,
    by_tercile,
    daily_ic,
    deflated_sharpe,
    expected_max_sharpe,
    lag_ic,
    long_short,
    newey_west_t,
    publish_bar,
)

D0, D1 = date(2020, 1, 2), date(2020, 1, 3)


def _scores(
    day: date, scores: Sequence[float], rets: Sequence[float], terc: int = 1
) -> pl.DataFrame:
    n = len(scores)
    return pl.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "signal_date": [day] * n,
            "score": [float(s) for s in scores],
            "ret_next_excess": [float(r) for r in rets],
            "liquidity_tercile": pl.Series([terc] * n, dtype=pl.Int8),
        }
    )


def test_daily_ic_perfect_reversed_and_min_names():
    up = list(range(20))
    perfect = _scores(D0, up, [float(x) for x in up])
    reversed_ = _scores(D1, up, [float(-x) for x in up])
    small = _scores(date(2020, 1, 6), [1.0, 2.0], [1.0, 2.0])
    ic = daily_ic(pl.concat([perfect, reversed_, small]), min_names=20)
    assert ic["signal_date"].to_list() == [D0, D1]
    assert ic["ic"].to_list() == pytest.approx([1.0, -1.0])


def test_long_short_weights_turnover_cost_and_net():
    # 20 names -> quintiles of 4; long the top 4 scores, short the bottom 4
    rets = [0.01 * i for i in range(20)]
    day0 = _scores(D0, list(range(20)), rets)
    day1 = _scores(D1, list(reversed(range(20))), rets)  # full reversal of both legs
    ls = long_short(pl.concat([day0, day1]), min_names=20, cost=0.001)
    top, bottom = np.mean(rets[16:]), np.mean(rets[:4])
    assert ls["gross"].to_list() == pytest.approx([top - bottom, bottom - top])
    assert ls["turnover"].to_list() == pytest.approx([2.0, 4.0])
    assert ls["net"].to_list() == pytest.approx([top - bottom - 0.002, bottom - top - 0.004])


def test_newey_west_t_close_to_iid_t_for_iid_data():
    x = np.random.default_rng(0).normal(0.1, 1.0, 5000)
    iid_t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    assert newey_west_t(x, lags=5) == pytest.approx(iid_t, rel=0.15)


def test_annualised_sharpe():
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert annualised_sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))


def test_lag_ic_uses_following_calendar_date_return():
    cal = pl.Series("date", [D0, D1])
    scores = _scores(D0, list(range(20)), [0.0] * 20)
    returns = pl.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(20)],
            "date": [D1] * 20,
            "ret_next_excess": [float(i) for i in range(20)],
            "eligible": [True] * 20,
        }
    )
    assert lag_ic(scores, returns, cal, min_names=20)["ic"].to_list() == pytest.approx([1.0])


def test_deflated_sharpe_is_half_at_expected_max_and_increases():
    r = np.random.default_rng(1).normal(0.001, 0.01, 1000)
    sr = r.mean() / r.std(ddof=1)
    z = expected_max_sharpe(6, 1.0)  # scale factor for unit variance
    at_max = deflated_sharpe(r, n_trials=6, sr_var=(sr / z) ** 2)
    assert at_max == pytest.approx(0.5, abs=1e-9)
    assert deflated_sharpe(r + 0.0005, n_trials=6, sr_var=(sr / z) ** 2) > at_max


def test_by_tercile_reports_each_tercile():
    frames = []
    for k in range(3):
        day = D0 + timedelta(days=k)
        frames.append(_scores(day, list(range(20)), [float(i) for i in range(20)], terc=1))
        frames.append(_scores(day, list(range(20)), [float(-i) for i in range(20)], terc=2))
    rows = {r["tercile"]: r for r in by_tercile(pl.concat(frames), min_names=20)}
    assert rows[1]["mean_ic"] == pytest.approx(1.0)
    assert rows[2]["mean_ic"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("args", "failing"),
    [
        ((2.5, 0.8, 0.02, 0.005, 0.01), set()),
        ((1.9, 0.8, 0.02, 0.005, 0.01), {"1_ic_t_ge_2"}),
        ((2.5, 0.4, 0.02, 0.005, 0.01), {"2_net_sharpe_ge_0_5"}),
        ((2.5, 0.8, 0.02, 0.015, 0.01), {"3_lag_ic_le_half"}),
        ((2.5, 0.8, 0.02, 0.005, -0.001), {"4_holdout_ic_gt_0"}),
        ((2.5, 0.8, 0.02, 0.005, float("nan")), {"4_holdout_ic_gt_0"}),
    ],
)
def test_publish_bar(args, failing):
    bar = publish_bar(*args)
    assert {k for k, v in bar["conditions"].items() if not v["pass"]} == failing
    assert bar["publish"] is (not failing)
