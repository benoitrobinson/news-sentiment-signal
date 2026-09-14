"""`ss <command>`: gates, data build, pre-registered runs, summary and report. Run in the order
shown by `ss --help`; each step reads the previous step's files."""

import argparse
import json
import os
import shutil
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from sentiment_signal import config
from sentiment_signal.align import build_panel, build_panel_same_day
from sentiment_signal.baseline import tfidf_featurize, vader_scores
from sentiment_signal.data.news_av import collect_av_news, load_av_news
from sentiment_signal.data.news_kaggle import TS_FORMAT, load_kaggle_news, midnight_share
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


def _require_gates_open() -> None:
    """Called before a gate does any work, so a refused gate spends no requests and overwrites no
    file that a later step reads."""
    seen = sorted(
        p.name for p in config.RESULTS.glob("*.json") if p.stem not in {"gates", "deviations"}
    )
    if seen:
        raise SystemExit(
            f"results exist ({', '.join(seen)}): gates are frozen once any result is seen "
            "(spec section 6); record a deviation instead"
        )


def _record_gate(name: str, result: dict) -> None:
    _require_gates_open()
    gates = _read_json(config.RESULTS / "gates.json")
    gates[name] = {**result, "recorded_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write_json(config.RESULTS / "gates.json", gates)
    print(json.dumps({name: gates[name]}, indent=2, default=str))


def _calendar() -> pl.Series:
    return pl.read_parquet(config.PROCESSED / "calendar.parquet")["date"]


def _primary() -> str:
    """The primary model as G4 decided it. Nothing is modelled until every gate is recorded and G1
    and G2 pass (spec section 6: gates come before any modelling)."""
    gates = _read_json(config.RESULTS / "gates.json")
    missing = [g for g in ("G1", "G2ab", "G2c", "G4") if g not in gates]
    if missing:
        raise SystemExit(f"gates {missing} not recorded; run them first (spec section 6)")
    failed = [g for g in ("G1", "G2ab", "G2c") if not gates[g]["pass"]]
    if failed:
        raise SystemExit(
            f"gates {failed} failed; resolve them before any modelling (spec section 6)"
        )
    return "chrono" if gates["G4"]["pass"] else "tfidf"


def _grid(model: str) -> tuple[dict, ...]:
    return config.LOGIT_C_GRID if model == "chrono" else config.TFIDF_GRID


def _require_clean_null(model: str) -> None:
    null = _read_json(config.RESULTS / f"{model}_null.json").get("overall")
    if null is None:
        raise SystemExit(f"no null run: run `ss run --model {model} --null` first (spec section 5)")
    if not abs(null["ic_t"]) < config.NULL_MAX_ABS_T:  # a NaN t fails too
        raise SystemExit(
            f"null IC t = {null['ic_t']}: the pipeline leaks; nothing is summarised or reported "
            "until the leak is found (spec section 5)"
        )


def _kaggle_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    import kagglehub

    return Path(kagglehub.dataset_download(config.KAGGLE_DATASET)) / config.KAGGLE_FILE


def offsets_follow_dst(stamps: pl.Series) -> tuple[bool, float]:
    """(pass, share of stamps whose written clock time is New York wall time at the stamped
    instant). A fixed offset all year, or offsets flipped against the season, moves headlines an
    hour across the 15:50 cut-off; stamps that are all UTC are exact and pass."""
    offsets = set(stamps.str.extract(r"([+-]\d{2}:?\d{2}|Z)$", 1).drop_nulls().unique())
    frame = pl.DataFrame({"s": stamps}).select(
        wall=pl.col("s")
        .str.to_datetime(TS_FORMAT, strict=False)
        .dt.convert_time_zone(config.TZ)
        .dt.replace_time_zone(None),
        written=pl.col("s").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    )
    # over every stamp: an unparseable or offset-less stamp counts against the share
    agreement = frame.select((pl.col("wall") == pl.col("written")).fill_null(False).mean()).item()
    agreement = float(agreement) if agreement is not None else 0.0
    all_utc = bool(offsets) and offsets <= {"+00:00", "+0000", "Z"}
    return all_utc or agreement >= config.G1_MIN_DST_AGREEMENT, agreement


def cmd_gate_g1(args: argparse.Namespace) -> None:
    _require_gates_open()
    path = _kaggle_path(args.file)
    print(pl.read_csv(path, n_rows=3, infer_schema=False))
    share = midnight_share(path, config.KAGGLE_TS_COL)
    stamps = pl.scan_csv(path, infer_schema=False).select(config.KAGGLE_TS_COL).collect()
    stamps = stamps[config.KAGGLE_TS_COL]
    offsets = dict(
        stamps.str.extract(r"([+-]\d{2}:?\d{2}|Z)$", 1).drop_nulls().value_counts().iter_rows()
    )
    try:
        news = load_kaggle_news(
            path, config.KAGGLE_HEADLINE_COL, config.KAGGLE_TS_COL, config.KAGGLE_TICKER_COL
        )
        parse_ok, rows = True, news.height
    except ValueError as err:
        parse_ok, rows = False, str(err)
    dst_ok, dst_agreement = offsets_follow_dst(stamps)
    passed = args.licence_ok and parse_ok and share < config.G1_MAX_MIDNIGHT_SHARE and dst_ok
    _record_gate(
        "G1",
        {
            "file": path.name,  # the full kagglehub path would publish the home directory
            "licence": args.licence,
            "licence_ok": args.licence_ok,
            "midnight_share": share,
            "utc_offset_parse_ok": parse_ok,
            "utc_offsets": offsets,
            "new_york_wall_time_share": dst_agreement,
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
    _require_gates_open()
    panel = pl.read_parquet(config.PROCESSED / "panel.parquet")
    year_headlines = (
        panel.filter(pl.col("signal_date").dt.year() == max(config.TEST_YEARS))["headline"]
        .unique()
        .sort()
    )  # unique() order varies between runs; sorting makes the seeded sample fixed
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


def _av_span() -> tuple[datetime, datetime]:
    """Collection span: a day either side of the holdout, so the first signal date gets the prior
    afternoon's headlines and the last gets its own morning's."""
    start, end = (datetime.fromisoformat(s) for s in (config.HOLDOUT_START, config.HOLDOUT_END))
    return start - timedelta(days=1), end + timedelta(days=1)


def probe_days(start: date, end: date, n: int) -> list[date]:
    """n distinct weekdays spread evenly over [start, end], both ends included. News volume peaks
    on weekdays, so projecting weekday request counts onto every calendar day is conservative."""
    span = (start + timedelta(days=k) for k in range((end - start).days + 1))
    weekdays = [d for d in span if d.weekday() < 5]
    if not 2 <= n <= len(weekdays):
        raise SystemExit(f"--days must be between 2 and {len(weekdays)}")
    return [weekdays[round(i * (len(weekdays) - 1) / (n - 1))] for i in range(n)]


def cmd_gate_g2(args: argparse.Namespace) -> None:
    _require_gates_open()
    key = os.environ["ALPHA_VANTAGE_KEY"]
    first, last = (date.fromisoformat(s) for s in (config.HOLDOUT_START, config.HOLDOUT_END))
    start, end = _av_span()
    probe = config.RAW / "av_probe"
    days = probe_days(first, last, args.days)
    windows = 0
    for d in days:
        day = datetime.combine(d, datetime.min.time())
        windows += collect_av_news(day, day + timedelta(days=1), probe, key)
    years = set(load_av_news(probe)["ts"].dt.year().to_list())
    stamps = {f"{d:%Y%m%d}" for d in days}
    # every window of a probe day starts on that day; a split window cost a request, wrote no file
    requests = sum(
        1 for f in probe.iterdir() if f.suffix in {".json", ".split"} and f.name[:8] in stamps
    )
    projected_requests = requests / len(days) * (end - start).days
    collection_days = projected_requests / args.daily_limit
    holdout_years = set(range(first.year, last.year + 1))
    _record_gate(
        "G2ab",
        {
            "probe_days": [str(d) for d in days],
            "windows_fetched": windows,
            "years_with_headlines": sorted(years),
            "projected_requests": projected_requests,
            "daily_limit": args.daily_limit,
            "projected_collection_days": collection_days,
            "timezone_assumed": config.AV_ASSUMED_TZ,
            "pass": holdout_years <= years and collection_days <= config.G2_MAX_COLLECTION_DAYS,
        },
    )


def cmd_collect_av(args: argparse.Namespace) -> None:
    fetched = collect_av_news(
        *_av_span(),
        config.RAW / "av",
        os.environ["ALPHA_VANTAGE_KEY"],
        pause_seconds=args.pause,
    )
    print("windows fetched this run:", fetched)


def cmd_holdout_panel(args: argparse.Namespace) -> None:
    _require_gates_open()
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
    _primary()
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
    """Highest validation IC averaged over the walk-forward validation years; like walk_forward, a
    year where a config has no finite IC is skipped for that config, and ties go to the earlier
    config."""
    run = _read_json(config.RESULTS / f"{model}.json")
    if "validation" not in run:
        raise SystemExit(f"run `ss run --model {model}` first")
    ranked = (
        pl.DataFrame(run["validation"])
        .filter(pl.col("mean_ic").is_finite())
        .group_by("config")
        .agg(pl.col("mean_ic").mean())
        .sort(["mean_ic", "config"], descending=[True, False])
    )
    if ranked.is_empty():
        raise SystemExit(f"no {model} config has a finite mean validation IC")
    return _grid(model)[ranked["config"][0]]


def cmd_holdout(args: argparse.Namespace) -> None:
    model, calendar = _primary(), _calendar()
    _require_clean_null(model)  # the holdout is scored once, and never before the leak rule
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
    _require_clean_null(model)
    holdout = _read_json(config.RESULTS / "holdout.json")
    if holdout.get("model") != model:
        raise SystemExit(f"results/holdout.json is missing or not from {model}: run `ss holdout`")
    scores = pl.read_parquet(config.PROCESSED / f"{model}_scores.parquet")
    returns = pl.read_parquet(config.PROCESSED / "returns.parquet")
    primary = _read_json(config.RESULTS / f"{model}.json")["overall"]
    lag = lag_ic(scores, returns, calendar)
    mean_lag = float(lag["ic"].mean()) if lag.height else float("nan")
    holdout_ic = holdout["mean_ic"]
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
                net, n_trials=len(_grid(model)), sr_var=float(per_config["sr"].var(ddof=1))
            ),
            "terciles": by_tercile(scores),
            "event_study": event_study(scores, returns, calendar).to_dicts(),
        },
    )
    print(json.dumps(_read_json(config.RESULTS / "summary.json")["bar"], indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from sentiment_signal.report import build_report

    _require_clean_null(_primary())
    if not (config.RESULTS / "summary.json").exists():
        raise SystemExit("run `ss summarize` first")
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
    g2.add_argument("--days", type=int, default=6, help="weekdays sampled across the holdout")
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
    r.add_argument("--null", action="store_true", help="labels permuted across all training rows")
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
