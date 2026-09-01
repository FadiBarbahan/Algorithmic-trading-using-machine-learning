"""
Central hyperparameter configuration.

Every design decision we discussed that should be tunable lives here, in one
place, so experiments change this dict rather than scattered magic numbers
in the code.
"""

DEFAULT_CONFIG = {
    # ---- Storage ----
    "storage": {
        "cache_dir": "data_cache",   # raw OHLCV parquet files, written by setup_data.py
        "results_dir": "results",    # fold results, equity curves, plots, tearsheets
    },

    # ---- Universe / data ----
    "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ"],
    "benchmark": "SPY",
    "start_date": "2012-05-18",   # 2012-05-18: with META, 2004-08-19: without META, 1997-05-15: without GOOGL
    "end_date": None,     # None = today

    # ---- Prediction horizon ----
    "horizon": 5,  # N trading days ahead (vertical barrier in triple-barrier labeling)

    # ---- Triple-barrier labeling (volatility-scaled, raw returns) ----
    "label": {
        "vol_window": 20,     # window used to estimate rolling volatility for barrier scaling
        "k_upper": 1.5,       # upper barrier = k_upper * rolling_vol
        "k_lower": 1.5,       # lower barrier = -k_lower * rolling_vol
        # NOTE: vertical barrier hit (neither touched within `horizon` days) -> label 0 (flat)
        # upper touched first -> label 1 (up), lower touched first -> label -1 (down)
    },

    # ---- Feature engineering ----
    # `lookback_window` is the primary HP to sweep; other windows are derived
    # multiples of it so the whole feature set scales together and stays
    # internally consistent (rather than tuning 10 independent windows).
    "features": {
        "lookback_window": 20,      # base window (drives realized vol, RSI, ATR, MA-short, volume MA)
        "ma_long_multiplier": 2.5,  # MA-long = lookback_window * this (e.g. 20 -> 50)
        "lag_depth": 5,             # k for lagged log-returns (r_{t-1}..r_{t-k})
        "rsi_window": 14,           # RSI conventionally 14; kept independent of lookback_window
        "atr_window": 14,           # ATR conventionally 14
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "autocorr_window": 40,      # lag-1 autocorrelation needs a longer, stable window (20-60d)
        "obv_window": 20,           # matches volume MA window per feature notes
    },

    # ---- Walk-forward cross-validation ----
    "cv": {
        "window_type": "fixed",   # "fixed" or "expanding" (fixed = sliding train window)
        "n_folds": 5,             # set depending on dataset size
        "embargo": 10,            # trading days purged between train and test to prevent leakage
        "min_train_size": 500,    # minimum training rows before the first fold is evaluated
    },

    # ---- Model ----
    "model": {
        "type": "logistic_regression",  # "logistic_regression" | "xgboost"
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",  # triple-barrier classes are rarely balanced
        },
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },

    # ---- Backtest ----
    "backtest": {
        "transaction_cost_bps": 5,   # 5 bps per side, as agreed success criterion
        "position_on_up_signal": 1,   # long when model predicts up
        "position_on_down_signal": -1,  # short when model predicts down (set 0 to disable shorting)
        "position_on_flat_signal": 0,
    },
}
