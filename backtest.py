"""
Simple vectorized backtest: convert predicted labels into positions, apply
transaction costs, compute the resulting equity curve and summary stats.

This is intentionally simple (no slippage model beyond a flat bps cost, no
position sizing beyond +-1/0). It exists to test the success criterion:
"profitable after ~5bps transaction costs", not to be a production backtester.
"""

import numpy as np
import pandas as pd


def positions_from_predictions(predictions: pd.Series, cfg: dict) -> pd.Series:
    mapping = {
        1: cfg["position_on_up_signal"],
        0: cfg["position_on_flat_signal"],
        -1: cfg["position_on_down_signal"],
    }
    return predictions.map(mapping)


def run_backtest(
    predictions: pd.Series,
    forward_returns: pd.Series,
    cfg: dict,
) -> dict:
    """Run a simple long/flat/short backtest.

    Args:
        predictions: predicted label per period {-1, 0, 1}, indexed by date
        forward_returns: actual realized return over the same horizon used
            for the label (i.e. the return the position would have earned)
        cfg: the `backtest` sub-dict from config.DEFAULT_CONFIG

    Returns:
        dict with equity curve, per-period net returns, and summary stats
    """
    positions = positions_from_predictions(predictions, cfg)
    positions = positions.reindex(forward_returns.index).fillna(0)

    gross_returns = positions * forward_returns

    # Transaction cost charged whenever position changes (enter/exit/flip)
    position_changes = positions.diff().abs().fillna(positions.abs())
    cost_per_change = cfg["transaction_cost_bps"] / 10_000
    costs = position_changes * cost_per_change

    net_returns = gross_returns - costs
    equity_curve = (1 + net_returns).cumprod()

    total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) else np.nan
    ann_factor = 252
    sharpe = (
        net_returns.mean() / net_returns.std() * np.sqrt(ann_factor)
        if net_returns.std() > 0
        else np.nan
    )
    win_rate = (net_returns > 0).mean() if len(net_returns) else np.nan

    return {
        "equity_curve": equity_curve,
        "net_returns": net_returns,
        "total_return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_periods": len(net_returns),
        "n_trades": int((position_changes > 0).sum()),
    }
