"""Every fixed number of the pre-registered protocol (spec sections 2, 4, 5, 6)."""

from datetime import date, time
from pathlib import Path

TZ = "America/New_York"
CUTOFF = time(15, 50)
# NYSE 13:00 early closes, 2008-2023, generated from exchange_calendars 4.13.2 (XNYS schedule).
# On these dates the market-on-close deadline moves with the close, so the cut-off is 12:50.
EARLY_CUTOFF = time(12, 50)
EARLY_CLOSE_DATES = tuple(
    date.fromisoformat(d)
    for d in (
        "2008-07-03",
        "2008-11-28",
        "2008-12-24",
        "2009-11-27",
        "2009-12-24",
        "2010-11-26",
        "2011-11-25",
        "2012-07-03",
        "2012-11-23",
        "2012-12-24",
        "2013-07-03",
        "2013-11-29",
        "2013-12-24",
        "2014-07-03",
        "2014-11-28",
        "2014-12-24",
        "2015-11-27",
        "2015-12-24",
        "2016-11-25",
        "2017-07-03",
        "2017-11-24",
        "2018-07-03",
        "2018-11-23",
        "2018-12-24",
        "2019-07-03",
        "2019-11-29",
        "2019-12-24",
        "2020-11-27",
        "2020-12-24",
        "2021-11-26",
        "2022-11-25",
        "2023-07-03",
        "2023-11-24",
    )
)

# Universe and calendar (spec section 2)
CALENDAR_MIN_TICKERS = 500
MIN_PRICE = 5.0
UNIVERSE_SIZE = 1000
DOLLAR_VOLUME_WINDOW = 21
# a one-day adjusted-close ratio beyond 4x either way is a bad tick, not a return: FNSPID mixes
# series and carries placeholder prices (final review, real-data probe)
MAX_DAILY_PRICE_RATIO = 4.0
# FNSPID joins price vintages after these dates, so returns across them mix split and dividend
# adjustments (2020-07-02: AAPL 364 -> 93; 2020-04-01: IBM flat close, adj close -17%). Found by
# two scans of 1982-2023: dates where 2+ eligible names move by a split ratio (2014-12-31,
# 2020-07-02), and dates where adj_close/close falls for far more tickers than on any other day
# (2020-07-02: 2,558; 2020-04-01: 550; every other date: 55 or fewer)
PRICE_SPLICE_DATES = (date(2014, 12, 31), date(2020, 4, 1), date(2020, 7, 2))

# Splits and model (spec section 4)
TEST_YEARS = tuple(range(2013, 2021))
TRAIN_YEARS = 4
LOGIT_C_GRID = tuple({"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0))
TFIDF_GRID = tuple({"ngram": ng, "C": c} for ng in ((1, 1), (1, 2)) for c in (0.1, 1.0, 10.0))
CHRONOBERT = "manelalab/chrono-bert-v1-{year}1231"
CHRONOBERT_GB = 0.6  # model.safetensors is 598,635,032 bytes
EMBED_DIM = 768
MAX_TOKENS = 64
EMBED_BATCH = 256
HOLDOUT_TRAIN_YEARS = (2017, 2020)
HOLDOUT_START, HOLDOUT_END = "2022-01-01", "2023-12-28"

# Evaluation (spec section 5)
MIN_NAMES_PER_DAY = 20
COST_PER_UNIT_TURNOVER = 0.0010
NW_LAGS = 5
TRADING_DAYS = 252
NULL_MAX_ABS_T = 2.0  # a null run at or above this |IC t| means the pipeline leaks

# Gates (spec section 6)
G1_MAX_MIDNIGHT_SHARE = 0.05
G1_MIN_DST_AGREEMENT = 0.99  # share of stamps written in New York wall time
G2_MAX_COLLECTION_DAYS = 90  # [A6] free key, 25 requests/day: Benoit chose to wait (2026-09-14)
G2_MIN_HOLDOUT_DAYS = 100
G4_MAX_EMBED_HOURS = 12.0
G4_DISK_HEADROOM_GB = 5.0
G4_BENCHMARK_HEADLINES = 10_000
G4_MAX_RAM_FRACTION = 0.6  # of physical RAM, for the largest block's logistic fit
G4_FIT_OVERHEAD_GB = 1.0  # interpreter, polars frames, chunked standardisation buffers

# Kaggle file and columns: defaults from the dataset listing, confirmed or corrected at gate G1
KAGGLE_DATASET = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
KAGGLE_FILE = "analyst_ratings_processed.csv"
KAGGLE_HEADLINE_COL = "title"
KAGGLE_TS_COL = "date"
KAGGLE_TICKER_COL = "stock"

# Alpha Vantage (spec section 6, G2)
AV_MIN_RELEVANCE = 0.5
AV_ASSUMED_TZ = TZ  # conservative if the documented timezone cannot be established

DATA = Path("data")
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EMBED_CACHE = DATA / "embeddings"
RESULTS = Path("results")
