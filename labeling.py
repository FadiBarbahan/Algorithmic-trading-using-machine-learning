"""
Triple-barrier labeling (Lopez de Prado), volatility-scaled, on raw returns.

`scan_forward_barrier` is the core primitive: given a starting index, scan
forward up to `horizon` periods and report which barrier is touched first.
It's used here to build training labels, AND reused as-is by
backtest.py's execution simulator -- the whole point of "mirroring the
labels" is that both use the exact same scanning logic, not two separate
reimplementations that could quietly drift apart.
"""

import numpy as np
import pandas as pd


def scan_forward_barrier(returns: np.ndarray, start_idx: int, vol: float,
                          horizon: int, k_upper: float, k_lower: float):
    """Scan forward from `start_idx` for up to `horizon` periods.

    Args:
        returns: full array of per-period log returns for one ticker
        start_idx: index to scan forward from (barriers apply to returns
            at start_idx+1 .. start_idx+horizon)
        vol: rolling volatility known at start_idx, used to size the barriers
        horizon: vertical barrier, in number of periods
        k_upper, k_lower: barrier width multipliers on vol

    Returns:
        (exit_offset, cum_log_return, hit_type) where exit_offset is 1..horizon,
        hit_type is one of {"upper", "lower", "vertical"}, or None if there
        isn't enough data (end of series, or vol not yet available).
    """
    n = len(returns)
    if start_idx + horizon >= n:
        return None
    if np.isnan(vol) or vol == 0:
        return None

    upper = k_upper * vol
    lower = -k_lower * vol

    cum_ret = 0.0
    for step in range(1, horizon + 1):
        r = returns[start_idx + step]
        if np.isnan(r):
            return None
        cum_ret += r
        if cum_ret >= upper:
            return step, cum_ret, "upper"
        if cum_ret <= lower:
            return step, cum_ret, "lower"

    return horizon, cum_ret, "vertical"


def triple_barrier_labels(
    log_return: pd.Series,
    rolling_vol: pd.Series,
    horizon: int,
    k_upper: float,
    k_lower: float,
) -> pd.Series:
    """Compute triple-barrier labels for a single ticker.

    upper touched first  -> label  1  (up)
    lower touched first  -> label -1  (down)
    vertical barrier hit -> label  0  (flat)

    Returns a Series of labels {-1, 0, 1}, indexed like log_return. The last
    `horizon` rows will be NaN (no full forward window available).
    """
    values = log_return.values
    vol = rolling_vol.values
    n = len(values)
    labels = np.full(n, np.nan)

    hit_type_to_label = {"upper": 1, "lower": -1, "vertical": 0}

    for t in range(n):
        result = scan_forward_barrier(values, t, vol[t], horizon, k_upper, k_lower)
        if result is None:
            continue
        _, _, hit_type = result
        labels[t] = hit_type_to_label[hit_type]

    return pd.Series(labels, index=log_return.index, name="label")
