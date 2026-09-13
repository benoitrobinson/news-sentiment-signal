from sentiment_signal import config
from sentiment_signal.cli import _years_needed, g4_projection, offsets_follow_dst


def _counts(value: int) -> dict[int, int]:
    return {y: value for y in range(2009, 2021)}


def test_g4_passes_when_time_disk_and_ram_fit():
    out = g4_projection(_counts(10_000), _counts(20_000), rate=400.0, free_disk_gb=20.0, ram_gb=8.0)
    blocks = _years_needed()
    assert out["headlines_to_embed"] == 10_000 * sum(len(ys) for ys in blocks.values())
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


def test_g1_offsets_must_follow_daylight_saving_or_be_utc():
    assert offsets_follow_dst({"-04:00": 700, "-05:00": 500})
    assert offsets_follow_dst({"+00:00": 1200})
    assert not offsets_follow_dst({"-04:00": 1200})  # fixed summer offset all year
