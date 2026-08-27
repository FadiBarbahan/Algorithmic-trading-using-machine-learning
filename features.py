"""
Feature engineering: the 12 features from feature_set_notes.md.

All windows are computed strictly on data up to and including time t (no
centered windows, no forward-looking rolling functions) to avoid lookahead
leakage. Every feature at row t only uses information available at the close
of day t.

Feature list (matches feature_set_notes.md numbering):
 1. Lagged log-returns (r_{t-1}..r_{t-k})
 2. Moving average ratio (P_t / MA_short)
 3. MA crossover signal (MA_short - MA_long, normalized by price)
 4. RSI (14-day)
 5. MACD histogram
 6. Realized volatility (rolling std of log-returns)
 7. ATR (14-day, normalized by price)
 8. Volume relative to its own moving average
 9. OBV change (normalized)
10. Return relative to benchmark (stock - SPY, same window)
11. Day-of-week (one-hot; cyclical deferred to LSTM stage per notes)
12. Rolling autocorrelation of returns (lag-1)
"""

import numpy as np
import pandas as pd


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd_histogram(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder's ATR, normalized by close price."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr / close


def _obv(df: pd.DataFrame) -> pd.Series:
    close, volume = df["close"], df["volume"]
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_features(df: pd.DataFrame, benchmark_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all 12 features for a single ticker.

    Args:
        df: OHLCV DataFrame for the ticker (from data.fetch_ohlcv)
        benchmark_df: OHLCV DataFrame for the benchmark (e.g. SPY)
        cfg: the `features` sub-dict from config.DEFAULT_CONFIG

    Returns:
        DataFrame of features aligned to df's index, plus the raw log_return
        column (needed downstream by labeling.py) and rolling_vol (needed for
        volatility-scaled barriers).
    """
    lookback = cfg["lookback_window"]
    ma_long = int(round(lookback * cfg["ma_long_multiplier"]))

    close = df["close"]
    log_ret = _log_returns(close)

    feat = pd.DataFrame(index=df.index)

    # 1. Lagged log-returns
    for lag in range(1, cfg["lag_depth"] + 1):
        feat[f"log_return_lag{lag}"] = log_ret.shift(lag - 1)  # lag1 = log_ret itself, at time t

    # 2. Moving average ratio
    ma_short = close.rolling(lookback).mean()
    feat["ma_ratio"] = close / ma_short

    # 3. MA crossover signal (normalized by price so it's comparable across tickers)
    ma_long_series = close.rolling(ma_long).mean()
    feat["ma_crossover"] = (ma_short - ma_long_series) / close

    # 4. RSI
    feat["rsi"] = _rsi(close, cfg["rsi_window"])

    # 5. MACD histogram (normalized by price for cross-ticker comparability)
    feat["macd_hist"] = _macd_histogram(
        close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]
    ) / close

    # 6. Realized volatility
    rolling_vol = log_ret.rolling(lookback).std()
    feat["realized_vol"] = rolling_vol

    # 7. ATR (normalized by price)
    feat["atr"] = _atr(df, cfg["atr_window"])

    # 8. Volume relative to its own moving average
    vol_ma = df["volume"].rolling(lookback).mean()
    feat["volume_ratio"] = df["volume"] / vol_ma

    # 9. OBV change, normalized by its own rolling std to keep scale sane
    obv = _obv(df)
    obv_change = obv.diff(cfg["obv_window"])
    feat["obv_change"] = obv_change / obv_change.rolling(cfg["obv_window"]).std()

    # 10. Return relative to benchmark (raw returns, matching the label definition)
    bench_ret = _log_returns(benchmark_df["close"]).reindex(df.index)
    feat["excess_return_vs_benchmark"] = log_ret - bench_ret

    # 11. Day-of-week, one-hot (Mon=0..Fri=4)
    dow = pd.get_dummies(df.index.dayofweek, prefix="dow")
    dow.index = df.index
    feat = feat.join(dow)

    # 12. Rolling lag-1 autocorrelation of returns
    feat["autocorr_lag1"] = log_ret.rolling(cfg["autocorr_window"]).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # Carried through for labeling.py (not "features" the model sees necessarily,
    # but kept alongside for convenience; drop before training if desired)
    feat["_log_return"] = log_ret
    feat["_rolling_vol_for_labeling"] = rolling_vol

    return feat
