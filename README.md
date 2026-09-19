# news-sentiment-signal

Does the text of a news headline predict a stock's next-day return relative to the market, once every
source of lookahead is removed? This repo answers that with a protocol fixed before any result was seen.

## Status

**The protocol is frozen and the results are not in yet.** The code is complete, the gates that come
before any modelling are recorded in [`results/gates.json`](results/gates.json), and the holdout news
is still being collected: the Alpha Vantage free key allows 25 requests a day against a two-year
holdout, so collection runs for about a month before the primary model is fit.

That order is the point rather than an accident. Whether the signal clears the bar was decided by
four conditions written down before any of them could be measured, and this README will carry
whichever answer comes back:

| Condition | Threshold |
|---|---|
| Daily rank IC t-stat (Newey-West) | >= 2 |
| Quintile long-short net Sharpe, after 10 bps | >= 0.5 |
| Mean IC one day later | <= half the mean IC |
| Holdout mean IC, different news source | > 0 |

A label-permutation null halts the pipeline before anything is summarised or reported: if scores
learned from shuffled labels reach \|t\| >= 2, the run leaks and nothing is published until the leak is
found. `REPORT.md` and the numbers behind it appear here when `ss report` has run.

## Design in one paragraph

Headlines are matched to the first trading day whose 15:50 New York cut-off they precede, and scored
against that day's close-to-close excess return. The universe is the 1,000 most liquid stocks above $5,
chosen with past data only. The primary model embeds headlines with **ChronoBERT** (He, Lv, Manela & Wu,
2025): for each test year, the version pretrained only on text published before that year starts, so the
language model cannot know what happened next. A logistic regression trained on the previous four years
(one-day embargo) produces the scores. Results are judged on a daily rank IC with Newey–West t-stats, a
quintile long-short after 10 bps costs, a lag test, a label-permutation null, a deflated Sharpe ratio, and a
holdout from a different news source. TF-IDF and VADER run through the identical evaluation as baselines.

## Reproduce

```sh
uv sync
export ALPHA_VANTAGE_KEY=...          # holdout collection only
uv run ss --help                      # commands are numbered in run order
uv run pytest                          # no network, no model download
```

Data (not redistributed): Kaggle *Daily Financial News for 6000+ Stocks*, Alpha Vantage `NEWS_SENTIMENT`,
FNSPID daily prices (Dong et al., 2024). Models: `manelalab/chrono-bert-v1-*` (MIT).

## Page

`frontend/` renders `results/*.json` as one page: the publish bar, the event study, the model
table, the gates. No backend and no network, so it also doubles as a status board while the
pipeline runs, naming the command behind each stage that has not produced its file yet.

```sh
cd frontend && npm install
npm run dev     # reads ../results
npm run demo    # synthetic fixtures, for working on the page before the pipeline has run
npm run build   # static dist/, base './' so it serves from a subpath
```
