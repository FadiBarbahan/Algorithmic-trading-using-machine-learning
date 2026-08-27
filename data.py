"""
OHLCV data fetching via yfinance.

Notes:
- Uses auto_adjust=True so splits/dividends are baked into OHLC. This matters
  for OBV (unadjusted splits corrupt the cumulative sum) and for returns in
  general.
- Survivorship bias is knowingly accepted for this v1 project: we fetch
  whatever tickers are passed in, as they exist today. No point-in-time
  constituent list.
"""

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, start=None, end=None) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV for a single ticker.

    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume
    """
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "date"
    return df


def fetch_universe(tickers, benchmark, start=None, end=None):
    """Fetch OHLCV for every ticker in the universe plus the benchmark.

    Returns:
        data: dict[ticker -> DataFrame]
        benchmark_df: DataFrame for the benchmark (e.g. SPY)
    """
    data = {}
    for t in tickers:
        try:
            data[t] = fetch_ohlcv(t, start=start, end=end)
        except ValueError as e:
            print(f"[warn] skipping {t}: {e}")

    benchmark_df = fetch_ohlcv(benchmark, start=start, end=end)
    return data, benchmark_df
