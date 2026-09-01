"""
Main pipeline (repeatable): load cached data -> features -> triple-barrier
labels -> walk-forward CV -> baseline model -> classification metrics ->
barrier-mirrored long-only backtest -> save results + plots.

Requires `python setup_data.py` to have been run first (this script does not
touch the network).

Run: python main.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
from sklearn.metrics import classification_report, confusion_matrix

from config import DEFAULT_CONFIG
from data import load_cached_universe
from features import compute_features
from labeling import triple_barrier_labels
from cv import walk_forward_splits
from models import build_model, fit_predict
from backtest import (
    simulate_ticker_strategy,
    combine_portfolio,
    compute_financial_metrics,
    compute_trade_stats,
)

FEATURE_COLUMNS_EXCLUDE = {
    "_log_return", "_rolling_vol_for_labeling", "label", "ticker", "date", "forward_return",
}


def build_pooled_dataset(cfg: dict) -> pd.DataFrame:
    """Load cached data, compute features/labels per ticker, and pool them.

    Fix applied here: the pooled panel is trimmed to start only after every
    ticker in the universe has data (i.e. after the last IPO date among
    them). Pooling dates where only 1-2 of 8 tickers exist yet is not
    meaningful for a cross-sectional model, and it was also causing early
    walk-forward folds to land entirely inside the pre-universe "warmup
    desert" and come back empty.
    """
    data, benchmark_df = load_cached_universe(
        cfg["tickers"], cfg["benchmark"], cfg["storage"]["cache_dir"]
    )

    common_start = max(df.index.min() for df in data.values())
    print(f"Trimming pooled dataset to start at {common_start.date()} "
          f"(latest first-available date across the {len(data)} tickers)")

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

        feat = feat.copy()
        feat["label"] = labels
        feat["ticker"] = ticker
        feat["date"] = feat.index
        frames.append(feat)

    pooled = pd.concat(frames, axis=0, ignore_index=True)
    pooled = pooled[pooled["date"] >= common_start]
    pooled = pooled.dropna(subset=["label"])
    pooled = pooled.sort_values("date").reset_index(drop=True)
    return pooled


def evaluate_baseline(y_true: pd.Series) -> float:
    """Up-class base rate: the precision a predictor guessing "up" at its
    natural frequency would get. Used as the beat-the-baseline bar.
    """
    return (y_true == 1).mean()


def run_fold_backtest(test_df: pd.DataFrame, preds: pd.Series, cfg: dict):
    """Run the barrier-mirrored backtest for every ticker in this fold's
    test set, then combine into an equal-weighted portfolio return series.
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


def plot_fold_timeline(fold_windows: list, results_dir: Path):
    """Hand-rolled sanity-check plot: one row per fold showing train / embargo
    / test blocks. No library needed for this -- it's exactly the kind of
    thing best verified visually rather than trusted blindly.
    """
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(fold_windows) + 1))
    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_windows):
        ax.barh(i, train_end - train_start, left=train_start, color="steelblue", label="train" if i == 0 else None)
        ax.barh(i, test_start - train_end, left=train_end, color="lightgray", label="embargo" if i == 0 else None)
        ax.barh(i, test_end - test_start, left=test_start, color="darkorange", label="test" if i == 0 else None)
    ax.set_yticks(range(len(fold_windows)))
    ax.set_yticklabels([f"Fold {i}" for i in range(len(fold_windows))])
    ax.set_xlabel("Row index (positional, within pooled date range)")
    ax.set_title("Walk-forward folds: train / embargo / test")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(results_dir / "fold_timeline.png", dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: list, fold_i: int, results_dir: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Fold {fold_i} confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(results_dir / f"confusion_matrix_fold{fold_i}.png", dpi=120)
    plt.close(fig)


def run_pipeline(cfg: dict = None):
    cfg = cfg or DEFAULT_CONFIG
    results_dir = Path(cfg["storage"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached data and building pooled feature/label set...")
    pooled = build_pooled_dataset(cfg)
    print(f"Pooled dataset: {len(pooled)} rows across {pooled['ticker'].nunique()} tickers, "
          f"{pooled['date'].min().date()} to {pooled['date'].max().date()}")

    feature_cols = [c for c in pooled.columns if c not in FEATURE_COLUMNS_EXCLUDE]

    unique_dates = np.sort(pooled["date"].unique())
    cv_cfg = cfg["cv"]

    fold_summaries = []
    fold_windows = []
    all_portfolio_returns = []
    all_trades = []

    class_labels = [-1, 0, 1]
    class_names = ["down", "flat", "up"]

    for fold_i, (train_date_idx, test_date_idx) in enumerate(
        walk_forward_splits(
            n_samples=len(unique_dates),
            n_folds=cv_cfg["n_folds"],
            embargo=cv_cfg["embargo"],
            min_train_size=cv_cfg["min_train_size"],
            window_type=cv_cfg["window_type"],
        )
    ):
        fold_windows.append((train_date_idx[0], train_date_idx[-1], test_date_idx[0], test_date_idx[-1]))

        train_dates = set(unique_dates[train_date_idx])
        test_dates = set(unique_dates[test_date_idx])

        train_df = pooled[pooled["date"].isin(train_dates)].dropna(subset=feature_cols)
        test_df = pooled[pooled["date"].isin(test_dates)].dropna(subset=feature_cols)

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Fold {fold_i}: skipped (empty after dropna)")
            continue

        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]

        model, scaler = build_model(cfg["model"])
        preds = fit_predict(model, scaler, X_train, y_train, X_test)
        preds = pd.Series(preds, index=test_df.index)

        # --- Classification metrics ---
        report = classification_report(
            y_test, preds, labels=class_labels, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, preds, labels=class_labels)
        plot_confusion_matrix(cm, class_names, fold_i, results_dir)

        up_precision = report["up"]["precision"]
        baseline_precision = evaluate_baseline(y_test)
        beat_baseline = up_precision > baseline_precision

        print(f"\n--- Fold {fold_i} ---")
        print(f"Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)")
        print(f"Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")
        print(f"Up-class precision: {up_precision:.3f} | recall: {report['up']['recall']:.3f} | "
              f"f1: {report['up']['f1-score']:.3f} | baseline: {baseline_precision:.3f} | beats: {beat_baseline}")

        # --- Barrier-mirrored backtest ---
        portfolio_returns, trades_df = run_fold_backtest(test_df, preds, cfg)
        fin_metrics = compute_financial_metrics(portfolio_returns)
        trade_stats = compute_trade_stats(trades_df)

        print(f"Backtest: total_return={fin_metrics['total_return']:.4f}, sharpe={fin_metrics['sharpe']:.2f}, "
              f"max_dd={fin_metrics['max_drawdown']:.4f}, n_trades={trade_stats['n_trades']}, "
              f"trade_win_rate={trade_stats['win_rate']}")

        fold_summaries.append({
            "fold": fold_i,
            "up_precision": up_precision,
            "up_recall": report["up"]["recall"],
            "up_f1": report["up"]["f1-score"],
            "baseline_precision": baseline_precision,
            "beats_baseline": beat_baseline,
            **{f"bt_{k}": v for k, v in fin_metrics.items()},
            **{f"trade_{k}": v for k, v in trade_stats.items()},
        })
        all_portfolio_returns.append(portfolio_returns)
        if not trades_df.empty:
            trades_df["fold"] = fold_i
            all_trades.append(trades_df)

    results_df = pd.DataFrame(fold_summaries)
    print("\n=== Summary across folds ===")
    print(results_df)

    all_folds_beat_baseline = results_df["beats_baseline"].all() if len(results_df) else False
    combined_returns = pd.concat(all_portfolio_returns).sort_index() if all_portfolio_returns else pd.Series(dtype=float)
    overall_profitable = qs.stats.comp(combined_returns) > 0 if len(combined_returns) else False

    print(f"\nSuccess criterion (beats baseline on ALL folds): {all_folds_beat_baseline}")
    print(f"Success criterion (overall backtest profitable after costs): {overall_profitable}")

    # --- Save results to disk ---
    results_df.to_parquet(results_dir / "fold_results.parquet")
    results_df.to_csv(results_dir / "fold_results.csv", index=False)
    combined_returns.to_frame().to_parquet(results_dir / "combined_portfolio_returns.parquet")
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_parquet(results_dir / "all_trades.parquet")

    plot_fold_timeline(fold_windows, results_dir)

    if len(combined_returns) > 0 and combined_returns.abs().sum() > 0:
        try:
            qs.reports.html(
                combined_returns, output=str(results_dir / "tearsheet.html"),
                title="Walk-forward out-of-sample backtest",
            )
            print(f"\nSaved quantstats tearsheet to {results_dir / 'tearsheet.html'}")
        except Exception as e:
            print(f"[warn] quantstats tearsheet generation failed: {e}")

    print(f"All results saved to {results_dir}/")

    return results_df, combined_returns


if __name__ == "__main__":
    run_pipeline()
