"""
End-to-end pipeline: fetch data -> features -> triple-barrier labels ->
pooled walk-forward CV -> baseline model -> evaluation vs. success criterion
-> simple backtest.

Run: python main.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, classification_report

from config import DEFAULT_CONFIG
from data import fetch_universe
from features import compute_features
from labeling import triple_barrier_labels
from cv import walk_forward_splits
from models import build_model, fit_predict
from backtest import run_backtest


FEATURE_COLUMNS_EXCLUDE = {"_log_return", "_rolling_vol_for_labeling", "label", "ticker", "date"}


def build_pooled_dataset(cfg: dict) -> pd.DataFrame:
    """Fetch data, compute features and labels for every ticker, and pool
    them into a single long-format DataFrame with a `date` and `ticker`
    column so the model can be trained across the whole universe at once.
    """
    data, benchmark_df = fetch_universe(
        cfg["tickers"], cfg["benchmark"], cfg["start_date"], cfg["end_date"]
    )

    horizon = cfg["horizon"]
    label_cfg = cfg["label"]
    feature_cfg = cfg["features"]

    frames = []
    for ticker, df in data.items():
        feat = compute_features(df, benchmark_df, feature_cfg)

        labels = triple_barrier_labels(
            log_return=feat["_log_return"],
            rolling_vol=feat["_rolling_vol_for_labeling"],
            horizon=horizon,
            k_upper=label_cfg["k_upper"],
            k_lower=label_cfg["k_lower"],
        )

        # forward_returns: realized cumulative return over the horizon, used
        # later by the backtest (NOT used as a model input -- label already
        # encodes the relevant information, this is just for P&L evaluation)
        forward_return = feat["_log_return"].rolling(horizon).sum().shift(-horizon)

        feat = feat.copy()
        feat["label"] = labels
        feat["forward_return"] = forward_return
        feat["ticker"] = ticker
        feat["date"] = feat.index

        frames.append(feat)

    pooled = pd.concat(frames, axis=0, ignore_index=True)
    pooled = pooled.dropna(subset=["label"])  # drop rows with no valid label (end of series, insufficient vol history)
    pooled = pooled.sort_values("date").reset_index(drop=True)
    return pooled


def evaluate_baseline(y_true: pd.Series) -> float:
    """Majority-class precision baseline for the up-class (label == 1):
    what precision would you get by always predicting the majority class?
    If majority class isn't 1, up-class precision from a majority-class
    predictor is 0 by definition -- the more informative baseline is the
    up-class base rate (precision of a *random* predictor that guesses
    up-class at its natural frequency).
    """
    return (y_true == 1).mean()


def run_pipeline(cfg: dict = None):
    cfg = cfg or DEFAULT_CONFIG
    print("Fetching data and building pooled feature/label set...")
    pooled = build_pooled_dataset(cfg)
    print(f"Pooled dataset: {len(pooled)} rows across {pooled['ticker'].nunique()} tickers")

    feature_cols = [c for c in pooled.columns if c not in FEATURE_COLUMNS_EXCLUDE and c != "forward_return"]

    unique_dates = np.sort(pooled["date"].unique())
    cv_cfg = cfg["cv"]

    fold_results = []
    all_backtest_returns = []

    for fold_i, (train_date_idx, test_date_idx) in enumerate(
        walk_forward_splits(
            n_samples=len(unique_dates),
            n_folds=cv_cfg["n_folds"],
            embargo=cv_cfg["embargo"],
            min_train_size=cv_cfg["min_train_size"],
            window_type=cv_cfg["window_type"],
        )
    ):
        train_dates = set(unique_dates[train_date_idx])
        test_dates = set(unique_dates[test_date_idx])

        train_mask = pooled["date"].isin(train_dates)
        test_mask = pooled["date"].isin(test_dates)

        train_df = pooled[train_mask].dropna(subset=feature_cols)
        test_df = pooled[test_mask].dropna(subset=feature_cols)

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Fold {fold_i}: skipped (empty after dropna)")
            continue

        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]

        model, scaler = build_model(cfg["model"])
        preds = fit_predict(model, scaler, X_train, y_train, X_test)
        preds = pd.Series(preds, index=test_df.index)

        up_precision = precision_score(y_test, preds, labels=[1], average="macro", zero_division=0)
        baseline_precision = evaluate_baseline(y_test)

        beat_baseline = up_precision > baseline_precision

        print(f"\n--- Fold {fold_i} ---")
        print(f"Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)")
        print(f"Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")
        print(f"Up-class precision: {up_precision:.3f} | Baseline (up-rate): {baseline_precision:.3f} | Beats baseline: {beat_baseline}")

        bt_result = run_backtest(preds, test_df.loc[test_df.index, "forward_return"], cfg["backtest"])
        print(f"Backtest: total_return={bt_result['total_return']:.4f}, sharpe={bt_result['sharpe']:.2f}, n_trades={bt_result['n_trades']}")

        fold_results.append({
            "fold": fold_i,
            "up_precision": up_precision,
            "baseline_precision": baseline_precision,
            "beats_baseline": beat_baseline,
            "backtest_total_return": bt_result["total_return"],
            "backtest_sharpe": bt_result["sharpe"],
        })
        all_backtest_returns.append(bt_result["net_returns"])

    results_df = pd.DataFrame(fold_results)
    print("\n=== Summary across folds ===")
    print(results_df)

    all_folds_beat_baseline = results_df["beats_baseline"].all() if len(results_df) else False
    combined_returns = pd.concat(all_backtest_returns) if all_backtest_returns else pd.Series(dtype=float)
    overall_profitable = (1 + combined_returns).prod() > 1 if len(combined_returns) else False

    print(f"\nSuccess criterion (beats baseline on ALL folds): {all_folds_beat_baseline}")
    print(f"Success criterion (overall backtest profitable after costs): {overall_profitable}")

    return results_df, combined_returns


if __name__ == "__main__":
    run_pipeline()
