# news-sentiment-signal

Does the text of a news headline predict a stock's next-day return relative to the market, once every
source of lookahead is removed? This repo answers that with a protocol fixed before any result was seen.
The findings are in [`REPORT.md`](REPORT.md).

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
