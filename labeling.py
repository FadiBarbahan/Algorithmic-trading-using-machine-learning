"""
Triple-barrier labeling (Lopez de Prado), volatility-scaled, on raw returns.

For each time t, look forward up to `horizon` days. Using cumulative log
returns from t:
    upper barrier = +k_upper * rolling_vol[t]
    lower barrier = -k_lower * rolling_vol[t]

Whichever is touched first determines the label:
    upper touched first  -> label  1  (up)
    lower touched first  -> label -1  (down)
    neither touched by horizon (vertical barrier) -> label 0 (flat)

rolling_vol[t] is estimated using only information available at time t
(computed in features.py as a trailing rolling std), so the barrier width
itself introduces no lookahead.
"""

import numpy as np
import pandas as pd


def triple_barrier_labels(
    log_return: pd.Series,
    rolling_vol: pd.Series,
    horizon: int,
    k_upper: float,
    k_lower: float,
) -> pd.Series:
    """Compute triple-barrier labels for a single ticker.

    Args:
        log_return: per-period log returns (log(close_t / close_{t-1}))
        rolling_vol: trailing rolling std of log_return, known at time t
        horizon: vertical barrier, in number of periods (N days)
        k_upper, k_lower: barrier width multipliers on rolling_vol

    Returns:
        Series of labels {-1, 0, 1}, indexed like log_return. The last
        `horizon` rows will be NaN since they have no full forward window
        to evaluate.
    """
    n = len(log_return)
    values = log_return.values
    vol = rolling_vol.values
    labels = np.full(n, np.nan)

    for t in range(n - horizon):
        v = vol[t]
        if np.isnan(v) or v == 0:
            continue  # not enough history yet to size barriers

        upper = k_upper * v
        lower = -k_lower * v

        cum_ret = 0.0
        label = 0  # default: vertical barrier reached without a touch
        for step in range(1, horizon + 1):
            r = values[t + step]
            if np.isnan(r):
                label = np.nan
                break
            cum_ret += r
            if cum_ret >= upper:
                label = 1
                break
            if cum_ret <= lower:
                label = -1
                break

        labels[t] = label

    return pd.Series(labels, index=log_return.index, name="label")
