from datetime import time

from sentiment_signal import config


def test_protocol_constants_match_the_pre_registered_spec():
    assert config.TZ == "America/New_York"
    assert config.CUTOFF == time(15, 50)
    assert (config.MIN_PRICE, config.UNIVERSE_SIZE, config.DOLLAR_VOLUME_WINDOW) == (5.0, 1000, 21)
    assert config.TEST_YEARS == tuple(range(2013, 2021))
    assert config.TRAIN_YEARS == 4
    assert [g["C"] for g in config.LOGIT_C_GRID] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    assert [(g["ngram"], g["C"]) for g in config.TFIDF_GRID] == [
        ((1, 1), 0.1),
        ((1, 1), 1.0),
        ((1, 1), 10.0),
        ((1, 2), 0.1),
        ((1, 2), 1.0),
        ((1, 2), 10.0),
    ]
    assert config.CHRONOBERT.format(year=2012) == "manelalab/chrono-bert-v1-20121231"
    assert (config.MIN_NAMES_PER_DAY, config.COST_PER_UNIT_TURNOVER, config.NW_LAGS) == (
        20,
        0.0010,
        5,
    )
    assert (config.HOLDOUT_START, config.HOLDOUT_END) == ("2022-01-01", "2023-12-28")
