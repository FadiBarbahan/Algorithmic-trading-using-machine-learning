"""
OHLCV data fetching (yfinance) + parquet caching (pyarrow).

Two entry points:
- `download_and_cache_universe`: called once by setup_data.py. Hits the
  network, writes one parquet file per ticker to `cache_dir`.
- `load_cached_universe`: called by main.py on every run. Reads parquet from
  disk, no network access, fast and dtype-stable.

Notes:
- Uses auto_adjust=True so splits/dividends are baked into OHLC. This matters
  for OBV (unadjusted splits corrupt the cumulative sum) and for returns in
  general.
- Survivorship bias is knowingly accepted for this v1 project: we fetch
  whatever tickers are passed in, as they exist today. No point-in-time
  constituent list.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def _cache_path(cache_dir: str, ticker: str) -> Path:
    return Path(cache_dir) / f"{ticker}.parquet"


def fetch_ohlcv(ticker: str, start=None, end=None) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV for a single ticker directly from yfinance."""
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "date"
    return df


def download_and_cache_universe(tickers, benchmark, cache_dir, start=None, end=None):
    """One-time download: fetch every ticker + benchmark, write to parquet.

    Meant to be called from setup_data.py, not from the main pipeline.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    all_symbols = list(tickers) + [benchmark]

    for symbol in all_symbols:
        try:
            df = fetch_ohlcv(symbol, start=start, end=end)
        except ValueError as e:
            print(f"[warn] skipping {symbol}: {e}")
            continue
        path = _cache_path(cache_dir, symbol)
        df.to_parquet(path)
        print(f"Cached {symbol}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()} -> {path}")


def load_cached_universe(tickers, benchmark, cache_dir):
    """Load OHLCV for the universe + benchmark from local parquet cache.

    Raises a clear error (rather than silently hitting the network) if the
    cache is missing, so the setup/main split stays a hard boundary.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory '{cache_dir}' not found. Run `python setup_data.py` first."
        )

    data = {}
    for ticker in tickers:
        path = _cache_path(cache_dir, ticker)
        if not path.exists():
            print(f"[warn] no cached data for {ticker} at {path}, skipping")
            continue
        data[ticker] = pd.read_parquet(path)

    bench_path = _cache_path(cache_dir, benchmark)
    if not bench_path.exists():
        raise FileNotFoundError(
            f"No cached data for benchmark '{benchmark}' at {bench_path}. Run `python setup_data.py` first."
        )
    benchmark_df = pd.read_parquet(bench_path)

    if not data:
        raise FileNotFoundError(
            f"No cached ticker data found in '{cache_dir}'. Run `python setup_data.py` first."
        )

    return data, benchmark_df
