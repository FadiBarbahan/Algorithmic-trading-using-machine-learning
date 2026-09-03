"""
Barrier-mirrored, long-only backtest.

Execution rule (mirrors the label-generation logic exactly, using the same
`scan_forward_barrier` primitive from labeling.py):
- Enter long only when the model predicts "up" (1) AND we're not already
  holding a position in that ticker.
- Once in a trade, ignore all subsequent signals (down/flat/up) until the
  trade exits via whichever comes first: upper barrier (take-profit), lower
  barrier (stop-loss), or the vertical/time barrier -- using the SAME
  k_upper/k_lower/horizon as label generation.
- This makes trades non-overlapping *by construction* (we never enter a new
  trade while one is open), so a plain cumulative product across trades is
  valid -- no double-counting of overlapping windows.

Because this is a state machine (must know "am I in a trade right now"),
it's run per-ticker sequentially, then combined into an equal-weighted
portfolio return series across the fixed universe.

Financial metrics are computed with quantstats rather than hand-rolled.
"""

import numpy as np
import pandas as pd
import quantstats as qs

from labeling import scan_forward_barrier


def simulate_ticker_strategy(
    dates: pd.DatetimeIndex,
    log_returns: np.ndarray,
    rolling_vol: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    k_upper: float,
    k_lower: float,
    cost_bps: float,
):
    """Run the barrier-mirrored long-only strategy for a single ticker.

    Args:
        dates: DatetimeIndex, one per row, sorted ascending
        log_returns: array of per-period log returns, same length as dates
        rolling_vol: array of rolling vol known at each t (same basis as labeling)
        predictions: array of predicted labels {-1, 0, 1}, same length as dates
        horizon, k_upper, k_lower: must match label generation exactly
        cost_bps: round-trip-per-side transaction cost in bps

    Returns:
        daily_returns: pd.Series indexed by `dates`, net-of-cost returns,
            zero on days with no open/closing trade, the realized trade
            return lumped on the exit day otherwise
        trades_df: pd.DataFrame, one row per completed trade
    """
    n = len(dates)
    daily_returns = np.zeros(n)
    cost = cost_bps / 10_000
    trades = []

    t = 0
    while t < n:
        if predictions[t] == 1:
            result = scan_forward_barrier(
                log_returns, t, rolling_vol[t], horizon, k_upper, k_lower
            )
            if result is None:
                t += 1
                continue

            exit_offset, cum_log_return, hit_type = result
            exit_idx = t + exit_offset

            gross_return = np.exp(cum_log_return) - 1
            net_return = gross_return - 2 * cost  # cost charged on entry + exit

            daily_returns[exit_idx] += net_return
            trades.append({
                "entry_date": dates[t],
                "exit_date": dates[exit_idx],
                "holding_days": exit_offset,
                "exit_reason": hit_type,
                "gross_return": gross_return,
                "net_return": net_return,
            })

            t = exit_idx + 1  # no new entries until this trade is closed
        else:
            t += 1

    daily_returns_series = pd.Series(daily_returns, index=dates, name="return")
    trades_df = pd.DataFrame(trades)
    return daily_returns_series, trades_df


def combine_portfolio(per_ticker_returns: dict, universe_size: int) -> pd.Series:
    """Equal-weight the fixed universe: each ticker gets 1/universe_size of
    capital regardless of whether it's currently in a trade. Days with no
    trade for a given ticker contribute 0 for that ticker's slice.
    """
    if not per_ticker_returns:
        return pd.Series(dtype=float)

    combined = pd.concat(per_ticker_returns.values(), axis=1).fillna(0)
    portfolio_returns = combined.sum(axis=1) / universe_size
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns.sort_index()


def compute_financial_metrics(returns: pd.Series) -> dict:
    """Financial performance metrics via quantstats. Returns NaN for any
    metric that errors out (e.g. too few nonzero returns to compute Sortino).
    """
    metrics = {}
    metric_fns = {
        "total_return": lambda r: qs.stats.comp(r),
        "cagr": qs.stats.cagr,
        "sharpe": qs.stats.sharpe,
        "sortino": qs.stats.sortino,
        "max_drawdown": qs.stats.max_drawdown,
        "calmar": qs.stats.calmar,
        "win_rate": qs.stats.win_rate,
        "profit_factor": qs.stats.profit_factor,
        "volatility": qs.stats.volatility,
    }
    for name, fn in metric_fns.items():
        try:
            metrics[name] = float(fn(returns))
        except Exception:
            metrics[name] = np.nan
    return metrics


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    """Descriptive stats on the actual trades taken -- plain pandas, no
    library needed for this since it's just aggregating our own trade log.
    """
    if trades_df.empty:
        return {
            "n_trades": 0, "win_rate": np.nan, "avg_net_return": np.nan,
            "avg_holding_days": np.nan, "pct_exit_upper": np.nan,
            "pct_exit_lower": np.nan, "pct_exit_vertical": np.nan,
        }

    exit_counts = trades_df["exit_reason"].value_counts(normalize=True)
    return {
        "n_trades": len(trades_df),
        "win_rate": (trades_df["net_return"] > 0).mean(),
        "avg_net_return": trades_df["net_return"].mean(),
        "avg_holding_days": trades_df["holding_days"].mean(),
        "pct_exit_upper": exit_counts.get("upper", 0.0),
        "pct_exit_lower": exit_counts.get("lower", 0.0),
        "pct_exit_vertical": exit_counts.get("vertical", 0.0),
    }


def run_fold_backtest(test_df: pd.DataFrame, preds: pd.Series, cfg: dict):
    """Run the barrier-mirrored backtest for every ticker in this fold's
    test set, then combine into an equal-weighted portfolio return series.

    Lives here (not main.py) so both main.py and permutation_test.py can
    call it without a circular import -- the permutation test needs to run
    this exact same backtest logic on noise-trained predictions.
    """
    label_cfg = cfg["label"]
    horizon = cfg["horizon"]
    cost_bps = cfg["backtest"]["transaction_cost_bps"]
    universe_size = len(cfg["tickers"])

    per_ticker_returns = {}
    all_trades = []

    for ticker, group in test_df.groupby("ticker"):
        group = group.sort_values("date")
        dates = pd.DatetimeIndex(group["date"])
        log_returns = group["_log_return"].values
        rolling_vol = group["_rolling_vol_for_labeling"].values
        ticker_preds = preds.loc[group.index].values

        daily_returns, trades_df = simulate_ticker_strategy(
            dates=dates,
            log_returns=log_returns,
            rolling_vol=rolling_vol,
            predictions=ticker_preds,
            horizon=horizon,
            k_upper=label_cfg["k_upper"],
            k_lower=label_cfg["k_lower"],
            cost_bps=cost_bps,
        )
        per_ticker_returns[ticker] = daily_returns
        if not trades_df.empty:
            trades_df["ticker"] = ticker
            all_trades.append(trades_df)

    portfolio_returns = combine_portfolio(per_ticker_returns, universe_size)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    return portfolio_returns, trades_df
