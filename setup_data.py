"""
One-time setup: download OHLCV for the universe + benchmark and cache to
parquet. Run this once (and again whenever you want fresher data or a
different universe) before running main.py.

Run: python setup_data.py
"""

from config import DEFAULT_CONFIG
from data import download_and_cache_universe


def main():
    cfg = DEFAULT_CONFIG
    print(f"Downloading {len(cfg['tickers'])} tickers + benchmark '{cfg['benchmark']}'...")
    download_and_cache_universe(
        tickers=cfg["tickers"],
        benchmark=cfg["benchmark"],
        cache_dir=cfg["storage"]["cache_dir"],
        start=cfg["start_date"],
        end=cfg["end_date"],
    )
    print("Done. Run `python main.py` next.")


if __name__ == "__main__":
    main()
