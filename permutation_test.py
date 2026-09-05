"""
Label-shuffle permutation test.

For each fold, repeatedly shuffle the TRAINING labels (breaking any real
relationship between features and target), retrain the same model type on
this noise, and evaluate it on the real (unshuffled) test set -- both on
classification precision and on the full barrier-mirrored backtest.

This builds a null distribution: "what performance would a model with zero
real predictive skill achieve on this exact data, purely by chance?" The
resulting p-value is the fraction of the null distribution that matches or
beats the real, unshuffled result. It directly stress-tests thin margins --
if a noise-trained model clears your observed precision edge routinely by
chance, that's much weaker evidence than the raw "beats baseline" number
suggests.

Cost note: N permutations means N model refits + N full backtests per fold.
Logistic regression fits are cheap, so this stays fast for v1; switching to
xgboost will make this substantially slower (worth revisiting n_permutations
at that point).

Each fold's full null return series (not just the summary metrics) is kept
and returned, so main.py can additionally build a POOLED null distribution --
stitching trial i's noise-model results across all folds into one full
out-of-sample curve, for each of the n_permutations trials -- giving a much
higher-power significance test than judging each fold in isolation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

from models import build_model, fit_predict
from backtest import run_fold_backtest, compute_financial_metrics


def run_permutation_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    observed_precision: float,
    observed_sharpe: float,
    observed_total_return: float,
    cfg: dict,
    fold_i: int = 0,
):
    """Run the label-shuffle permutation test for a single fold.

    Returns a dict with the null distributions (for plotting/inspection) and
    p-values for precision, backtest Sharpe, and backtest total return.
    """
    pt_cfg = cfg["permutation_test"]
    n_permutations = pt_cfg["n_permutations"]
    rng = np.random.default_rng(pt_cfg["random_state"] + fold_i)  # vary seed per fold

    y_train_values = y_train.values

    null_precision = np.empty(n_permutations)
    null_sharpe = np.empty(n_permutations)
    null_total_return = np.empty(n_permutations)
    null_returns = []  # full return series per permutation, for pooled/stitched significance testing across folds

    print(f"    Running {n_permutations} label-shuffle permutations for fold {fold_i}...")
    for i in range(n_permutations):
        shuffled_y = pd.Series(rng.permutation(y_train_values), index=y_train.index)

        model, scaler = build_model(cfg["model"])
        preds = fit_predict(model, scaler, X_train, shuffled_y, X_test)
        preds = pd.Series(preds, index=X_test.index)

        null_precision[i] = precision_score(
            y_test, preds, labels=[1], average="macro", zero_division=0
        )

        portfolio_returns, _ = run_fold_backtest(test_df, preds, cfg)
        null_returns.append(portfolio_returns)
        fin_metrics = compute_financial_metrics(portfolio_returns)
        sharpe = fin_metrics["sharpe"]
        total_return = fin_metrics["total_return"]
        # A permutation with no trades (NaN metric) has no evidence of
        # "beating" anything -- treat as -inf so it never counts toward the
        # null clearing a positive observed result.
        null_sharpe[i] = sharpe if not np.isnan(sharpe) else -np.inf
        null_total_return[i] = total_return if not np.isnan(total_return) else -np.inf

        if (i + 1) % 50 == 0:
            print(f"      permutation {i + 1}/{n_permutations}...")

    p_precision = float(np.mean(null_precision >= observed_precision))
    p_sharpe = float(np.mean(null_sharpe >= observed_sharpe))
    p_total_return = float(np.mean(null_total_return >= observed_total_return))

    print(f"    p-value (precision): {p_precision:.3f} | "
          f"p-value (sharpe): {p_sharpe:.3f} | p-value (total_return): {p_total_return:.3f}")

    return {
        "null_precision": null_precision,
        "null_sharpe": null_sharpe,
        "null_total_return": null_total_return,
        "null_returns": null_returns,
        "p_value_precision": p_precision,
        "p_value_sharpe": p_sharpe,
        "p_value_total_return": p_total_return,
    }
