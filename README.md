# Stock Price Movement Classification — v1 Pipeline

Three-class (up/flat/down) directional classifier using triple-barrier
labeling, walk-forward CV, a barrier-mirrored long-only backtest, a
label-shuffle permutation-test significance check, and buy-and-hold
benchmark comparisons.

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
- `backtest.py` — barrier-mirrored long-only backtest, financial metrics via
  `quantstats`; also exposes `run_fold_backtest`, reused by the permutation test
- `permutation_test.py` — label-shuffle significance test (see below)
- `main.py` — **repeatable** pipeline: loads cached data, runs CV/model/
  backtest/permutation-test/benchmarks, saves results + plots. Never touches
  the network.

## Running it

```bash
pip install -r requirements.txt
python setup_data.py   # once, or whenever you want fresh data
python main.py          # repeatable, no network access
```

Edit `config.py` to change tickers, horizon, barrier widths, lookback window,
CV folds/embargo, model type, or permutation-test settings.

## The backtest's execution rule

Enter long only when the model predicts "up," and only if not already
holding a position in that ticker. Once in a trade, hold until whichever
comes first: the upper barrier (take-profit), lower barrier (stop-loss), or
the vertical/time barrier — using the *same* `k_upper`/`k_lower`/`horizon`
as label generation. This mirrors the labels exactly rather than
approximating them, and makes trades non-overlapping by construction (no new
entry while one is open), so equity curves compound correctly. Position
changes and equity curves are computed per-ticker (never across the pooled,
date-sorted table), then combined into an equal-weighted portfolio.

## Permutation test: how to read the p-values

For each fold, `permutation_test.py` shuffles the training labels 200 times
(config: `permutation_test.n_permutations`), retrains the same model on each
shuffled version, and evaluates it on the real test set — both on precision
and on the full barrier-mirrored backtest. This builds a null distribution:
what a model with zero real skill would achieve on this exact data purely by
chance. The p-value is the fraction of that null distribution that matched
or beat your real, unshuffled result.

- **Low p-value (e.g. < 0.05)**: unlikely to arise from a skill-less model
  fit to noise — meaningful evidence.
- **High p-value**: a noise-trained model clears your result routinely by
  chance — treat that fold's "beats baseline" with real skepticism, even
  though it technically passed.

This tests each fold in isolation and does **not** correct for checking 5
folds at once — the deflated Sharpe ratio remains the right tool for that
broader multiple-testing correction, and is still a good stretch goal before
drawing strong conclusions across all folds combined.

## Benchmark comparisons

Two buy-and-hold series are built from the same cached data and aligned to
the strategy's exact out-of-sample dates, then compared via quantstats'
native `benchmark` parameter (no hand-rolled equity math):
- **SPY buy-and-hold** — the standard market benchmark
- **Equal-weight buy-and-hold of your own 8-ticker universe** — apples-to-
  apples with the strategy's own universe and weighting convention

Both produce a full quantstats tearsheet (`tearsheet_vs_spy.html`,
`tearsheet_vs_universe_bh.html`), plus a compact `benchmark_comparison.csv`
with total return / CAGR / Sharpe / max drawdown side by side.

## Output files (in `results/`)

- `fold_results.parquet` / `.csv` — per-fold classification + backtest
  metrics, plus permutation-test p-values (`perm_p_value_precision`,
  `perm_p_value_sharpe`, `perm_p_value_total_return`)
- `combined_portfolio_returns.parquet` — stitched out-of-sample daily returns
- `all_trades.parquet` — every individual trade taken, across all folds
- `permutation_nulls.parquet` — full null distributions per fold, for
  inspection beyond the plotted histograms
- `benchmark_comparison.csv` — strategy vs. both buy-and-hold benchmarks
- `fold_timeline.png` — visual sanity check of train/embargo/test blocks
- `confusion_matrix_fold{N}.png` — per-fold confusion matrix
- `permutation_test_fold{N}.png` — null distribution histograms (precision +
  Sharpe) with the observed value marked, one per fold
- `tearsheet_vs_spy.html` / `tearsheet_vs_universe_bh.html` — full quantstats
  tearsheets against each benchmark

## Next steps

1. Read the p-values before trusting the "beats baseline" headline — a fold
   that technically passed but has a high p-value is weak evidence
2. Compare strategy vs. both buy-and-hold benchmarks in `benchmark_comparison.csv`
   — "profitable after costs" is not the same as "beat the market"
3. Resist broad grid-searching over `k_upper`/`k_lower`/`lookback_window`/
   `horizon` — that's exactly the backtest-overfitting trap Lopez de Prado
   warns about; try a few combinations manually instead
4. Once the logistic regression baseline is trusted, switch
   `config["model"]["type"]` to `"xgboost"` (note: permutation-test runtime
   scales with model fit cost, so `n_permutations` may need to come down
   once you switch)
5. Deflated Sharpe ratio remains a good stretch goal for correcting across
   all 5 folds at once, beyond what the per-fold permutation test covers
