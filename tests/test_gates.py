import json
from datetime import date

import polars as pl
import pytest

from sentiment_signal import config
from sentiment_signal.cli import (
    _primary,
    _years_needed,
    g4_projection,
    offsets_follow_dst,
    probe_days,
)


def _counts(value: int) -> dict[int, int]:
    return {y: value for y in range(2009, 2021)}


def test_g4_passes_when_time_disk_and_ram_fit():
    out = g4_projection(_counts(10_000), _counts(20_000), rate=400.0, free_disk_gb=20.0, ram_gb=8.0)
    blocks = _years_needed()
    total = 10_000 * sum(len(ys) for ys in blocks.values())
    assert out["headlines_to_embed"] == total
    assert out["cache_plus_models_gb"] == total * config.EMBED_DIM * 2 / 1e9 + 0.6 * len(blocks)
    assert out["largest_block_rows"] == 20_000 * 5
    assert out["projected_fit_ram_gb"] == 100_000 * config.EMBED_DIM * 6 / 1e9 + 1.0
    assert out["pass"] is True


def test_g4_fails_on_each_limit_separately():
    ok = {"unique_by_year": _counts(10_000), "rows_by_year": _counts(20_000)}
    ok_env = {"rate": 400.0, "free_disk_gb": 20.0, "ram_gb": 8.0}
    assert not g4_projection(**ok, **{**ok_env, "rate": 1.0})["pass"]  # far over 12 hours
    assert not g4_projection(**ok, **{**ok_env, "free_disk_gb": 5.0})["pass"]  # no headroom
    big = {**ok, "rows_by_year": _counts(300_000)}  # 1.5M rows x 768 x 6 B = 6.9 GB + 1 GB
    assert not g4_projection(**big, **ok_env)["pass"]


def test_g1_offsets_must_follow_new_york_daylight_saving_or_be_utc():
    def follows(*stamps: str) -> bool:
        return offsets_follow_dst(pl.Series(stamps))[0]

    assert follows("2020-01-15 10:00:00-05:00", "2020-07-15 10:00:00-04:00")
    assert follows("2020-01-15 15:00:00+00:00", "2020-07-15 14:00:00+00:00")
    assert not follows("2020-01-15 10:00:00-04:00", "2020-07-15 10:00:00-04:00")  # fixed offset
    assert not follows("2020-01-15 10:00:00-04:00", "2020-07-15 10:00:00-05:00")  # flipped
    assert not follows("2020-01-15 10:00:00", "2020-07-15 10:00:00")  # no offsets at all
    assert not follows("2020-01-15 10:00:00-05:00", "2020-07-15 10:00:00")  # half without one


def test_g2_probe_days_are_distinct_weekdays_spanning_the_whole_holdout():
    days = probe_days(date(2022, 1, 1), date(2023, 12, 28), 6)
    assert len(set(days)) == 6 and all(d.weekday() < 5 for d in days)
    assert days[0] == date(2022, 1, 3) and days[-1] == date(2023, 12, 28)
    assert {d.year for d in probe_days(date(2022, 1, 1), date(2023, 12, 28), 2)} == {2022, 2023}
    assert len(set(probe_days(date(2022, 1, 3), date(2022, 1, 7), 5))) == 5


def test_modelling_waits_until_every_gate_is_recorded_and_g1_g2_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    gates = {"G1": {"pass": True}, "G2ab": {"pass": True}, "G2c": {"pass": True}}
    gates["G4"] = {"pass": False}

    def primary_with(recorded: dict) -> str:
        (tmp_path / "gates.json").write_text(json.dumps(recorded))
        return _primary()

    for gate in gates:
        with pytest.raises(SystemExit, match="not recorded"):
            primary_with({g: v for g, v in gates.items() if g != gate})
    for gate in ("G1", "G2ab", "G2c"):
        with pytest.raises(SystemExit, match="failed"):
            primary_with({**gates, gate: {"pass": False}})
    assert primary_with(gates) == "tfidf"
    assert primary_with({**gates, "G4": {"pass": True}}) == "chrono"
