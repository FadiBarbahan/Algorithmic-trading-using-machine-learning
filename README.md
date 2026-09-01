# Stock Price Movement Classification — v1 Pipeline

Three-class (up/flat/down) directional classifier using triple-barrier
labeling, walk-forward CV, and a barrier-mirrored long-only backtest.

## Files

- `config.py` — every hyperparameter, in one place, including storage paths
- `setup_data.py` — **one-time** script: downloads OHLCV via yfinance, caches
  to parquet. Run this whenever you want fresh data or a different universe.
- `data.py` — fetch (network) + load-from-cache (no network) functions
- `features.py` — the 12 features from feature_set_notes.md
- `labeling.py` — volatility-scaled triple-barrier labeling; exposes
  `scan_forward_barrier`, reused by the backtest's execution logic
- `cv.py` — walk-forward CV with purge/embargo gap (fixed or expanding window)
- `models.py` — logistic regression baseline (swap to xgboost via config)
- `backtest.py` — barrier-mirrored long-only backtest (see below), financial
  metrics via `quantstats`
- `main.py` — **repeatable** pipeline: loads cached data, runs CV/model/
  backtest, saves results + plots. Never touches the network.

## Running it

```bash
pip install -r requirements.txt
python setup_data.py   # once, or whenever you want fresh data
python main.py          # repeatable, no network access
```

Edit `config.py` to change tickers, horizon, barrier widths, lookback window,
CV folds/embargo, or model type.

## The backtest's execution rule

Enter long only when the model predicts "up," and only if not already
holding a position in that ticker. Once in a trade, hold until whichever
comes first: the upper barrier (take-profit), lower barrier (stop-loss), or
the vertical/time barrier — using the *same* `k_upper`/`k_lower`/`horizon`
as label generation. This mirrors the labels exactly rather than
approximating them, and it makes trades non-overlapping by construction
(no new entry while one is open), so equity curves compound correctly.

Two bugs from the previous version are fixed by this design:
1. **Return-compounding blowup**: the old backtest treated every day's
   5-day-forward return as if it were a fresh, non-overlapping period,
   which is not true — overlapping windows were being double/triple
   counted. Barrier-mirrored execution has each trade cover a genuinely
   distinct time span, so there's no overlap to worry about.
2. **Cross-ticker interleaving**: position changes and costs are now
   computed *within* each ticker (one state machine per ticker, combined
   into an equal-weighted portfolio afterward), not across the pooled,
   date-sorted table where unrelated tickers' positions were being diffed
   against each other.

A third fix — trimming the pooled panel to start only after the last IPO
date among the universe's tickers — addresses early walk-forward folds
landing entirely inside a period where most tickers don't have data yet
(the "warmup desert"), which was causing those folds to come back empty.

## What's been tested here

Yahoo Finance isn't reachable from this sandbox's network allowlist, so the
pipeline was smoke-tested against **synthetic OHLCV data with staggered
"IPO" dates** (mimicking your universe's real mix of decades-old and
recently-listed tickers) instead of real tickers. This confirmed:
- the common-start trim correctly detects and applies the latest first-
  available date across tickers
- walk-forward folds run without any being silently skipped
- the barrier-mirrored backtest produces realistic return magnitudes
  (single-digit-to-double-digit % swings), not near-total wipeouts
- trade counts are sane (roughly one trade per ticker per few weeks, not
  thousands of spurious "trades" per fold)
- classification metrics (precision/recall/F1, confusion matrix) and
  financial metrics (Sharpe, Sortino, max drawdown, Calmar, win rate,
  profit factor) all populate correctly
- results save to `results/` as parquet + CSV, plots save as PNG, and a
  full quantstats HTML tearsheet is generated

**Run `python setup_data.py` then `python main.py` on your machine with
real internet access before trusting any results on actual tickers.**

## Output files (in `results/`)

- `fold_results.parquet` / `.csv` — per-fold classification + backtest metrics
- `combined_portfolio_returns.parquet` — stitched out-of-sample daily returns
- `all_trades.parquet` — every individual trade taken, across all folds
- `fold_timeline.png` — visual sanity check of train/embargo/test blocks
- `confusion_matrix_fold{N}.png` — per-fold confusion matrix
- `tearsheet.html` — full quantstats tearsheet on the stitched returns

## Next steps

1. Run on real data, sanity-check `fold_timeline.png` and the printed fold
   date ranges
2. Try a few `lookback_window`, `k_upper`/`k_lower`, and `horizon`
   combinations manually — resist broad grid-searching yet, that's the
   backtest-overfitting trap Lopez de Prado warns about
3. Once trusted, switch `config["model"]["type"]` to `"xgboost"`
4. Consider `mlfinlab` as a reference implementation to sanity-check
   `labeling.py`/`cv.py` against, and `vectorbt` once you start sweeping
   hyperparameters instead of running one config at a time
5. Deflated Sharpe ratio / permutation testing remains a good stretch goal
   before trusting any single fold's Sharpe as meaningful
