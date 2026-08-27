# Stock Price Movement Classification — v1 Pipeline

Three-class (up/flat/down) directional classifier using triple-barrier
labeling, walk-forward CV, and a simple cost-aware backtest.

## Files

- `config.py` — every hyperparameter we discussed, in one place
- `data.py` — fetches OHLCV via yfinance (needs internet access; not available
  in this sandbox, tested with synthetic data instead — see below)
- `features.py` — the 12 features from feature_set_notes.md
- `labeling.py` — volatility-scaled triple-barrier labeling on raw returns
- `cv.py` — walk-forward CV with purge/embargo gap (fixed or expanding window)
- `models.py` — logistic regression baseline (swap to xgboost via config)
- `backtest.py` — vectorized backtest with transaction costs
- `main.py` — orchestrates the full pipeline end-to-end

## Running it

```bash
pip install yfinance scikit-learn pandas numpy
python main.py
```

Edit `config.py` to change tickers, horizon, barrier widths, lookback window,
CV folds/embargo, or model type.

## What's been tested here

Yahoo Finance isn't reachable from this sandbox's network allowlist, so the
full pipeline (`main.run_pipeline`) was smoke-tested against **synthetic
random-walk OHLCV data** instead of real tickers. This confirmed:
- data flows correctly from OHLCV → features → triple-barrier labels
- walk-forward CV produces non-overlapping, correctly embargoed folds
- the logistic regression baseline trains and predicts without errors
- the backtest correctly applies transaction costs and computes Sharpe

The pipeline correctly reported "does not beat baseline" on synthetic
random-walk data, which is the expected (correct) result — there's no real
signal in random walk returns, so a model failing to beat baseline there is
a sign the pipeline isn't lying to you, not a bug.

**Run `python main.py` on your machine with real internet access before
trusting any results on actual tickers.**

## Next steps (per our design discussion)

1. Confirm it runs on real data and sanity-check the fold date ranges printed
2. Try a few `lookback_window`, `k_upper`/`k_lower`, and `horizon` combinations
   manually (resist the urge to grid-search broadly yet — that's exactly the
   backtest-overfitting trap Lopez de Prado warns about)
3. Once the logistic regression baseline plumbing is trusted, switch
   `config["model"]["type"]` to `"xgboost"` (`pip install xgboost` first)
4. Consider adding the deflated Sharpe ratio / permutation testing (stretch
   goal from your project plan) before treating any single fold's Sharpe as
   meaningful
