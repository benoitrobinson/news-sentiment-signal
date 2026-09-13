"""`ss <command>`: gates, data build, pre-registered runs, summary and report. Run in the order
shown by `ss --help`; each step reads the previous step's files."""

import argparse
import json
import os
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from sentiment_signal import config
from sentiment_signal.align import build_panel, build_panel_same_day
from sentiment_signal.baseline import tfidf_featurize, vader_scores
from sentiment_signal.data.news_av import collect_av_news, load_av_news
from sentiment_signal.data.news_kaggle import load_kaggle_news, midnight_share
from sentiment_signal.data.prices import (
    build_returns,
    extract_prices,
    load_prices,
    trading_calendar,
)
from sentiment_signal.embed import ChronoEmbedder, embed_rows, make_chrono_featurize, vintage_for
from sentiment_signal.evaluate import (
    by_tercile,
    daily_ic,
    deflated_sharpe,
    event_study,
    lag_ic,
    long_short,
    publish_bar,
    summarise,
)
from sentiment_signal.model import fit_predict, stock_day_scores, walk_forward
from sentiment_signal.splits import select_years, walk_forward_splits


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _record_gate(name: str, result: dict) -> None:
    gates = _read_json(config.RESULTS / "gates.json")
    gates[name] = {**result, "recorded_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write_json(config.RESULTS / "gates.json", gates)
    print(json.dumps({name: gates[name]}, indent=2, default=str))


def _calendar() -> pl.Series:
    return pl.read_parquet(config.PROCESSED / "calendar.parquet")["date"]


def _primary() -> str:
    g4 = _read_json(config.RESULTS / "gates.json").get("G4")
    if g4 is None:
        raise SystemExit("gate G4 has not been recorded; run `ss gate-g4` first (spec section 6)")
    return "chrono" if g4["pass"] else "tfidf"


def _kaggle_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    import kagglehub

    return Path(kagglehub.dataset_download(config.KAGGLE_DATASET)) / config.KAGGLE_FILE


def offsets_follow_dst(offsets: dict[str, int]) -> bool:
    """A single non-UTC offset all year means local times were stamped with a fixed offset, which
    shifts winter (or summer) headlines by an hour across the 15:50 cut-off."""
    return len(offsets) > 1 or set(offsets) <= {"+00:00", "+0000", "Z"}


def cmd_gate_g1(args: argparse.Namespace) -> None:
    path = _kaggle_path(args.file)
    print(pl.read_csv(path, n_rows=3, infer_schema=False))
    share = midnight_share(path, config.KAGGLE_TS_COL)
    offsets = dict(
        pl.scan_csv(path, infer_schema=False)
        .select(pl.col(config.KAGGLE_TS_COL).str.extract(r"([+-]\d{2}:?\d{2}|Z)$", 1).alias("o"))
        .group_by("o")
        .len()
        .drop_nulls()
        .collect()
        .iter_rows()
    )
    try:
        news = load_kaggle_news(
            path, config.KAGGLE_HEADLINE_COL, config.KAGGLE_TS_COL, config.KAGGLE_TICKER_COL
        )
        parse_ok, rows = True, news.height
    except ValueError as err:
        parse_ok, rows = False, str(err)
    dst_ok = offsets_follow_dst(offsets)
    passed = args.licence_ok and parse_ok and share < config.G1_MAX_MIDNIGHT_SHARE and dst_ok
    _record_gate(
        "G1",
        {
            "file": str(path),
            "licence": args.licence,
            "licence_ok": args.licence_ok,
            "midnight_share": share,
            "utc_offset_parse_ok": parse_ok,
            "utc_offsets": offsets,
            "offsets_follow_dst": dst_ok,
            "rows": rows,
            "pass": passed,
        },
    )


def cmd_prices(args: argparse.Namespace) -> None:
    csv_dir = config.RAW / "prices"
    print("extracted", extract_prices(Path(args.zip), csv_dir), "price files")
    prices = load_prices(csv_dir)
    calendar = trading_calendar(prices, config.CALENDAR_MIN_TICKERS)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": calendar}).write_parquet(config.PROCESSED / "calendar.parquet")
    returns = build_returns(
        prices,
        calendar,
        min_price=config.MIN_PRICE,
        universe_size=config.UNIVERSE_SIZE,
        window=config.DOLLAR_VOLUME_WINDOW,
    )
    returns.write_parquet(config.PROCESSED / "returns.parquet")
    print(returns.filter(pl.col("eligible")).group_by(pl.col("date").dt.year()).len().sort("date"))


def cmd_panel(args: argparse.Namespace) -> None:
    news = load_kaggle_news(
        _kaggle_path(args.file),
        config.KAGGLE_HEADLINE_COL,
        config.KAGGLE_TS_COL,
        config.KAGGLE_TICKER_COL,
    )
    returns = pl.read_parquet(config.PROCESSED / "returns.parquet")
    calendar = _calendar()
    panel = build_panel(news, returns, calendar)
    panel.write_parquet(config.PROCESSED / "panel.parquet")
    build_panel_same_day(news, returns, calendar).write_parquet(
        config.PROCESSED / "panel_same_day.parquet"
    )
    print(panel.group_by(pl.col("signal_date").dt.year()).len().sort("signal_date"))


def _years_needed() -> dict[int, list[int]]:
    blocks = {y: list(range(y - config.TRAIN_YEARS, y + 1)) for y in config.TEST_YEARS}
    first, last = config.HOLDOUT_TRAIN_YEARS
    blocks[2022] = list(range(first, last + 1))  # holdout head; holdout rows are added separately
    return blocks


def g4_projection(
    unique_by_year: dict[int, int],
    rows_by_year: dict[int, int],
    rate: float,
    free_disk_gb: float,
    ram_gb: float,
) -> dict:
    """Pre-registered G4 arithmetic: embedding hours, disk for every cache and version, and peak RAM
    of the largest block's fit (resident float16 block + float32 train/eval copies)."""
    blocks = _years_needed()
    total = sum(sum(unique_by_year.get(y, 0) for y in ys) for ys in blocks.values())
    hours = total / rate / 3600
    # every (version, year) cache is kept for the null run, and every version stays in the HF cache
    disk_gb = total * config.EMBED_DIM * 2 / 1e9 + config.CHRONOBERT_GB * len(blocks)
    largest_rows = max(sum(rows_by_year.get(y, 0) for y in ys) for ys in blocks.values())
    fit_ram_gb = largest_rows * config.EMBED_DIM * (2 + 4) / 1e9 + config.G4_FIT_OVERHEAD_GB
    return {
        "headlines_per_second": rate,
        "headlines_to_embed": total,
        "projected_hours": hours,
        "cache_plus_models_gb": disk_gb,
        "free_disk_gb": free_disk_gb,
        "largest_block_rows": largest_rows,
        "projected_fit_ram_gb": fit_ram_gb,
        "ram_gb": ram_gb,
        "pass": hours <= config.G4_MAX_EMBED_HOURS
        and disk_gb <= free_disk_gb - config.G4_DISK_HEADROOM_GB
        and fit_ram_gb <= config.G4_MAX_RAM_FRACTION * ram_gb,
    }


def cmd_gate_g4(args: argparse.Namespace) -> None:
    panel = pl.read_parquet(config.PROCESSED / "panel.parquet")
    year_headlines = panel.filter(pl.col("signal_date").dt.year() == max(config.TEST_YEARS))[
        "headline"
    ].unique()
    sample = year_headlines.sample(
        min(config.G4_BENCHMARK_HEADLINES, year_headlines.len()), seed=0
    ).to_list()
    embedder = ChronoEmbedder(vintage_for(max(config.TEST_YEARS)))
    embedder.embed(sample[:512])  # warm-up: kernels and memory pools
    t0 = time.perf_counter()
    embedder.embed(sample)
    rate = len(sample) / (time.perf_counter() - t0)
    by_year = panel.group_by(pl.col("signal_date").dt.year().alias("year")).agg(
        unique=pl.col("headline").n_unique(), rows=pl.len()
    )
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    _record_gate(
        "G4",
        g4_projection(
            dict(zip(by_year["year"], by_year["unique"], strict=True)),
            dict(zip(by_year["year"], by_year["rows"], strict=True)),
            rate,
            shutil.disk_usage(".").free / 1e9,
            ram_gb,
        ),
    )


def cmd_gate_g2(args: argparse.Namespace) -> None:
    key = os.environ["ALPHA_VANTAGE_KEY"]
    start = datetime.fromisoformat(config.HOLDOUT_START)
    probe = config.RAW / "av_probe"
    windows = collect_av_news(start, start + timedelta(days=args.days), probe, key)
    items = load_av_news(probe).height
    files = len(list(probe.glob("*.json")))
    splits = len(list(probe.glob("*.split")))  # each split window cost a request but wrote no file
    holdout_days = (
        datetime.fromisoformat(config.HOLDOUT_END) - datetime.fromisoformat(config.HOLDOUT_START)
    ).days
    projected_requests = (files + splits) / args.days * holdout_days
    collection_days = projected_requests / args.daily_limit
    _record_gate(
        "G2ab",
        {
            "probe_days": args.days,
            "windows_fetched": windows,
            "relevant_rows": items,
            "projected_requests": projected_requests,
            "daily_limit": args.daily_limit,
            "projected_collection_days": collection_days,
            "timezone_assumed": config.AV_ASSUMED_TZ,
            "pass": items > 0 and collection_days <= config.G2_MAX_COLLECTION_DAYS,
        },
    )


def cmd_collect_av(args: argparse.Namespace) -> None:
    fetched = collect_av_news(
        datetime.fromisoformat(config.HOLDOUT_START),
        datetime.fromisoformat(config.HOLDOUT_END),
        config.RAW / "av",
        os.environ["ALPHA_VANTAGE_KEY"],
        pause_seconds=args.pause,
    )
    print("windows fetched this run:", fetched)


def cmd_holdout_panel(args: argparse.Namespace) -> None:
    news = load_av_news(config.RAW / "av")
    returns = pl.read_parquet(config.PROCESSED / "returns.parquet")
    panel = build_panel(news, returns, _calendar()).filter(
        pl.col("signal_date").is_between(
            datetime.fromisoformat(config.HOLDOUT_START).date(),
            datetime.fromisoformat(config.HOLDOUT_END).date(),
        )
    )
    panel.write_parquet(config.PROCESSED / "panel_holdout.parquet")
    names_per_day = panel.group_by("signal_date").agg(pl.col("ticker").n_unique().alias("n"))
    qualifying = names_per_day.filter(pl.col("n") >= config.MIN_NAMES_PER_DAY).height
    _record_gate(
        "G2c",
        {
            "holdout_rows": panel.height,
            "qualifying_days": qualifying,
            "pass": qualifying >= config.G2_MIN_HOLDOUT_DAYS,
        },
    )


def _run(model: str, panel: pl.DataFrame, shuffle_seed: int | None) -> None:
    calendar = _calendar()
    splits = walk_forward_splits(config.TEST_YEARS)
    if model == "chrono":
        featurize = make_chrono_featurize(panel, config.EMBED_CACHE)
        wf = walk_forward(
            panel,
            calendar,
            splits,
            featurize,
            config.LOGIT_C_GRID,
            dense=True,
            shuffle_seed=shuffle_seed,
        )
    else:
        wf = walk_forward(
            panel,
            calendar,
            splits,
            tfidf_featurize,
            config.TFIDF_GRID,
            dense=False,
            shuffle_seed=shuffle_seed,
        )
    name = model if shuffle_seed is None else f"{model}_null"
    wf.scores.write_parquet(config.PROCESSED / f"{name}_scores.parquet")
    wf.validation_returns.write_parquet(config.PROCESSED / f"{name}_validation_returns.parquet")
    _write_json(
        config.RESULTS / f"{name}.json",
        {
            "overall": summarise(wf.scores),
            "validation": wf.validation.to_dicts(),
            "by_year": [
                {"test_year": y, **summarise(wf.scores.filter(pl.col("test_year") == y))}
                for y in config.TEST_YEARS
            ],
        },
    )
    print(json.dumps(_read_json(config.RESULTS / f"{name}.json")["overall"], indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    if args.model != _primary() and args.model != "tfidf":
        raise SystemExit(f"G4 made {_primary()} the primary; {args.model} is not allowed")
    panel = pl.read_parquet(config.PROCESSED / "panel.parquet")
    _run(args.model, panel, shuffle_seed=0 if args.null else None)


def cmd_vader(args: argparse.Namespace) -> None:
    for name, file in (("vader", "panel.parquet"), ("old_bug_vader", "panel_same_day.parquet")):
        panel = pl.read_parquet(config.PROCESSED / file).filter(
            pl.col("signal_date").dt.year().is_in(config.TEST_YEARS)
        )
        scores = vader_scores(panel)
        hit = scores.select(
            ((pl.col("score") > 0) == (pl.col("ret_next_excess") > 0)).mean()
        ).item()
        _write_json(
            config.RESULTS / f"{name}.json", {"overall": {**summarise(scores), "hit_rate": hit}}
        )
        print(name, json.dumps(_read_json(config.RESULTS / f"{name}.json")["overall"], indent=2))


def _best_config(model: str) -> dict:
    validation = pl.DataFrame(_read_json(config.RESULTS / f"{model}.json")["validation"])
    best = validation.group_by("config").agg(pl.col("mean_ic").mean()).sort("mean_ic").tail(1)
    grid = config.LOGIT_C_GRID if model == "chrono" else config.TFIDF_GRID
    return grid[best["config"].item()]


def cmd_holdout(args: argparse.Namespace) -> None:
    model, calendar = _primary(), _calendar()
    params = _best_config(model)
    first, last = config.HOLDOUT_TRAIN_YEARS
    train = select_years(
        pl.read_parquet(config.PROCESSED / "panel.parquet"), first, last, calendar, embargo=True
    )
    holdout = pl.read_parquet(config.PROCESSED / "panel_holdout.parquet")
    if model == "chrono":
        version = vintage_for(2022)

        def featurize(tr, evals, p, year):
            embed = lambda rows: embed_rows(rows, version, config.EMBED_CACHE, ChronoEmbedder)  # noqa: E731
            return embed(tr).astype(np.float32), [embed(e).astype(np.float32) for e in evals]

        (proba,) = fit_predict(featurize, train, [holdout], params, 2022, dense=True)
    else:
        (proba,) = fit_predict(tfidf_featurize, train, [holdout], params, 2022, dense=False)
    scores = stock_day_scores(holdout, proba)
    scores.write_parquet(config.PROCESSED / "holdout_scores.parquet")
    ic = daily_ic(scores)
    _write_json(
        config.RESULTS / "holdout.json",
        {
            "model": model,
            "params": params,
            "days": ic.height,
            "mean_ic": float(ic["ic"].mean()) if ic.height else float("nan"),
        },
    )
    print(_read_json(config.RESULTS / "holdout.json"))


def cmd_summarize(args: argparse.Namespace) -> None:
    model, calendar = _primary(), _calendar()
    scores = pl.read_parquet(config.PROCESSED / f"{model}_scores.parquet")
    returns = pl.read_parquet(config.PROCESSED / "returns.parquet")
    primary = _read_json(config.RESULTS / f"{model}.json")["overall"]
    lag = lag_ic(scores, returns, calendar)
    mean_lag = float(lag["ic"].mean()) if lag.height else float("nan")
    holdout_ic = _read_json(config.RESULTS / "holdout.json").get("mean_ic", float("nan"))
    net = long_short(scores)["net"].to_numpy()
    validation = pl.read_parquet(config.PROCESSED / f"{model}_validation_returns.parquet")
    per_config = validation.group_by("config").agg(
        sr=pl.col("net").mean() / pl.col("net").std(ddof=1)
    )
    _write_json(
        config.RESULTS / "summary.json",
        {
            "primary": model,
            "bar": publish_bar(
                primary["ic_t"], primary["net_sharpe"], primary["mean_ic"], mean_lag, holdout_ic
            ),
            "mean_lag_ic": mean_lag,
            "deflated_sharpe": deflated_sharpe(
                net, n_trials=per_config.height, sr_var=float(per_config["sr"].var(ddof=1))
            ),
            "terciles": by_tercile(scores),
            "event_study": event_study(scores, returns, calendar).to_dicts(),
        },
    )
    print(json.dumps(_read_json(config.RESULTS / "summary.json")["bar"], indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from sentiment_signal.report import build_report

    build_report(config.RESULTS, Path("REPORT.md"))
    print("wrote REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ss", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g1 = sub.add_parser("gate-g1", help="1. Kaggle licence, offsets, midnight share")
    g1.add_argument("--file")
    g1.add_argument("--licence", required=True, help="licence text as shown on the Kaggle page")
    g1.add_argument("--licence-ok", action="store_true", help="it permits non-commercial research")
    g1.set_defaults(func=cmd_gate_g1)
    p = sub.add_parser("prices", help="2. extract FNSPID prices, build calendar and returns")
    p.add_argument("--zip", required=True)
    p.set_defaults(func=cmd_prices)
    pn = sub.add_parser("panel", help="3. build the next-day and same-day panels")
    pn.add_argument("--file")
    pn.set_defaults(func=cmd_panel)
    sub.add_parser("gate-g4", help="4. ChronoBERT speed and disk gate").set_defaults(
        func=cmd_gate_g4
    )
    g2 = sub.add_parser("gate-g2", help="5. Alpha Vantage probe (needs ALPHA_VANTAGE_KEY)")
    g2.add_argument("--days", type=int, default=3)
    g2.add_argument("--daily-limit", type=int, required=True, help="requests/day for your key")
    g2.set_defaults(func=cmd_gate_g2)
    c = sub.add_parser("collect-av", help="6. collect the holdout news (resumable)")
    c.add_argument("--pause", type=float, default=0.0)
    c.set_defaults(func=cmd_collect_av)
    sub.add_parser("holdout-panel", help="7. holdout panel + gate G2c").set_defaults(
        func=cmd_holdout_panel
    )
    r = sub.add_parser("run", help="8. walk-forward run of a model")
    r.add_argument("--model", choices=["chrono", "tfidf"], required=True)
    r.add_argument("--null", action="store_true", help="labels shuffled within each date")
    r.set_defaults(func=cmd_run)
    sub.add_parser("vader", help="9. VADER baseline and old-bug demo").set_defaults(func=cmd_vader)
    sub.add_parser("holdout", help="10. frozen primary on the holdout").set_defaults(
        func=cmd_holdout
    )
    sub.add_parser("summarize", help="11. bar, lag IC, DSR, terciles").set_defaults(
        func=cmd_summarize
    )
    sub.add_parser("report", help="12. REPORT.md").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)
