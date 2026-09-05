"""
Main pipeline (repeatable): load cached data -> features -> triple-barrier
labels -> walk-forward CV -> baseline model -> classification metrics ->
barrier-mirrored long-only backtest -> permutation-test significance check ->
buy-and-hold benchmark comparison -> save results + plots.

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
    run_fold_backtest,
    compute_financial_metrics,
    compute_trade_stats,
)
from permutation_test import run_permutation_test

FEATURE_COLUMNS_EXCLUDE = {
    "_log_return", "_rolling_vol_for_labeling", "label", "ticker", "date", "forward_return",
}


def build_pooled_dataset(cfg: dict):
    """Load cached data, compute features/labels per ticker, and pool them.

    Fix applied here: the pooled panel is trimmed to start only after every
    ticker in the universe has data (i.e. after the last IPO date among
    them). Pooling dates where only 1-2 of 8 tickers exist yet is not
    meaningful for a cross-sectional model, and it was also causing early
    walk-forward folds to land entirely inside the pre-universe "warmup
    desert" and come back empty.

    Returns:
        pooled: the trimmed, labeled, pooled feature DataFrame
        data: dict[ticker -> raw OHLCV DataFrame] (needed later for the
            equal-weight buy-and-hold benchmark)
        benchmark_df: raw OHLCV DataFrame for the benchmark (e.g. SPY),
            needed later for the SPY buy-and-hold benchmark
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
    return pooled, data, benchmark_df


def evaluate_baseline(y_true: pd.Series) -> float:
    """Up-class base rate: the precision a predictor guessing "up" at its
    natural frequency would get. Used as the beat-the-baseline bar.
    """
    return (y_true == 1).mean()


def build_buy_and_hold_benchmarks(data: dict, benchmark_df: pd.DataFrame, target_index: pd.DatetimeIndex):
    """Build two buy-and-hold benchmark return series, aligned to the
    strategy's out-of-sample date index so quantstats can compare them
    apples-to-apples:
      - SPY buy-and-hold (matches config["benchmark"])
      - equal-weight buy-and-hold across the 8-ticker universe (apples-to-
        apples with the strategy's own universe and weighting convention)
    """
    spy_returns = benchmark_df["close"].pct_change()
    spy_aligned = spy_returns.reindex(target_index).fillna(0)
    spy_aligned.name = "SPY buy-and-hold"

    ticker_returns = {t: df["close"].pct_change() for t, df in data.items()}
    universe_bh = pd.concat(ticker_returns.values(), axis=1).mean(axis=1)
    universe_bh_aligned = universe_bh.reindex(target_index).fillna(0)
    universe_bh_aligned.name = "Equal-weight universe buy-and-hold"

    return spy_aligned, universe_bh_aligned


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


def plot_permutation_null(perm_result: dict, observed_precision: float, observed_sharpe: float,
                           fold_i: int, results_dir: Path):
    """Null-distribution histograms with the observed (real) value marked --
    the fastest way to see visually whether a result looks like skill or luck.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(perm_result["null_precision"], bins=30, color="steelblue", alpha=0.8)
    axes[0].axvline(observed_precision, color="darkorange", linewidth=2, label="observed")
    axes[0].set_title(f"Fold {fold_i}: null precision\n(p={perm_result['p_value_precision']:.3f})")
    axes[0].set_xlabel("Up-class precision")
    axes[0].legend()

    finite_sharpe = perm_result["null_sharpe"][np.isfinite(perm_result["null_sharpe"])]
    axes[1].hist(finite_sharpe, bins=30, color="steelblue", alpha=0.8)
    axes[1].axvline(observed_sharpe, color="darkorange", linewidth=2, label="observed")
    axes[1].set_title(f"Fold {fold_i}: null Sharpe\n(p={perm_result['p_value_sharpe']:.3f})")
    axes[1].set_xlabel("Backtest Sharpe")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / f"permutation_test_fold{fold_i}.png", dpi=120)
    plt.close(fig)


def plot_pooled_permutation_null(pooled_result: dict, results_dir: Path):
    """Null-distribution histograms for the POOLED (all-folds-stitched)
    significance test -- the higher-power counterpart to the per-fold plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    finite_sharpe = pooled_result["pooled_null_sharpe"][np.isfinite(pooled_result["pooled_null_sharpe"])]
    axes[0].hist(finite_sharpe, bins=30, color="steelblue", alpha=0.8)
    axes[0].axvline(pooled_result["observed_sharpe"], color="darkorange", linewidth=2, label="observed")
    axes[0].set_title(f"Pooled null Sharpe (all folds stitched)\n(p={pooled_result['p_value_pooled_sharpe']:.3f})")
    axes[0].set_xlabel("Overall out-of-sample Sharpe")
    axes[0].legend()

    finite_tr = pooled_result["pooled_null_total_return"][np.isfinite(pooled_result["pooled_null_total_return"])]
    axes[1].hist(finite_tr, bins=30, color="steelblue", alpha=0.8)
    axes[1].axvline(pooled_result["observed_total_return"], color="darkorange", linewidth=2, label="observed")
    axes[1].set_title(f"Pooled null total return (all folds stitched)\n(p={pooled_result['p_value_pooled_total_return']:.3f})")
    axes[1].set_xlabel("Overall out-of-sample total return")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "permutation_test_pooled.png", dpi=120)
    plt.close(fig)


def run_pipeline(cfg: dict = None):
    cfg = cfg or DEFAULT_CONFIG
    results_dir = Path(cfg["storage"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached data and building pooled feature/label set...")
    pooled, data, benchmark_df = build_pooled_dataset(cfg)
    print(f"Pooled dataset: {len(pooled)} rows across {pooled['ticker'].nunique()} tickers, "
          f"{pooled['date'].min().date()} to {pooled['date'].max().date()}")

    feature_cols = [c for c in pooled.columns if c not in FEATURE_COLUMNS_EXCLUDE]

    unique_dates = np.sort(pooled["date"].unique())
    cv_cfg = cfg["cv"]
    pt_cfg = cfg["permutation_test"]

    fold_summaries = []
    fold_windows = []
    all_portfolio_returns = []
    all_trades = []
    all_permutation_nulls = []
    fold_null_returns_by_fold = {}  # fold_i -> list of null return Series, for pooled testing

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

        # --- Permutation test: is this result distinguishable from noise? ---
        perm_result = None
        if pt_cfg["enabled"]:
            perm_result = run_permutation_test(
                X_train, y_train, X_test, y_test, test_df,
                observed_precision=up_precision,
                observed_sharpe=fin_metrics["sharpe"],
                observed_total_return=fin_metrics["total_return"],
                cfg=cfg,
                fold_i=fold_i,
            )
            plot_permutation_null(perm_result, up_precision, fin_metrics["sharpe"], fold_i, results_dir)
            all_permutation_nulls.append({
                "fold": fold_i,
                "null_precision": perm_result["null_precision"],
                "null_sharpe": perm_result["null_sharpe"],
                "null_total_return": perm_result["null_total_return"],
            })
            fold_null_returns_by_fold[fold_i] = perm_result["null_returns"]

        fold_summaries.append({
            "fold": fold_i,
            "up_precision": up_precision,
            "up_recall": report["up"]["recall"],
            "up_f1": report["up"]["f1-score"],
            "baseline_precision": baseline_precision,
            "beats_baseline": beat_baseline,
            **{f"bt_{k}": v for k, v in fin_metrics.items()},
            **{f"trade_{k}": v for k, v in trade_stats.items()},
            **({
                "perm_p_value_precision": perm_result["p_value_precision"],
                "perm_p_value_sharpe": perm_result["p_value_sharpe"],
                "perm_p_value_total_return": perm_result["p_value_total_return"],
            } if perm_result else {}),
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
    if pt_cfg["enabled"] and len(results_df):
        print(f"Permutation p-values by fold (precision): {results_df['perm_p_value_precision'].round(3).tolist()}")
        print(f"Permutation p-values by fold (sharpe):    {results_df['perm_p_value_sharpe'].round(3).tolist()}")

    # --- Pooled significance test: stitch each permutation trial's noise-model
    # results across ALL folds into one full out-of-sample null curve, and
    # compare against the real stitched combined_returns. Far higher power
    # than judging each fold's ~500-trade sample in isolation. ---
    pooled_perm_result = None
    if pt_cfg["enabled"] and fold_null_returns_by_fold:
        n_permutations = pt_cfg["n_permutations"]
        fold_order = sorted(fold_null_returns_by_fold.keys())
        overall_metrics = compute_financial_metrics(combined_returns)

        pooled_null_sharpe = np.empty(n_permutations)
        pooled_null_total_return = np.empty(n_permutations)
        for i in range(n_permutations):
            stitched = pd.concat(
                [fold_null_returns_by_fold[f][i] for f in fold_order]
            ).sort_index()
            m = compute_financial_metrics(stitched)
            pooled_null_sharpe[i] = m["sharpe"] if not np.isnan(m["sharpe"]) else -np.inf
            pooled_null_total_return[i] = m["total_return"] if not np.isnan(m["total_return"]) else -np.inf

        p_pooled_sharpe = float(np.mean(pooled_null_sharpe >= overall_metrics["sharpe"]))
        p_pooled_total_return = float(np.mean(pooled_null_total_return >= overall_metrics["total_return"]))

        print(f"\n=== Pooled significance test (all {len(fold_order)} folds stitched, "
              f"{n_permutations} permutations) ===")
        print(f"Observed overall Sharpe: {overall_metrics['sharpe']:.3f} | pooled null p-value: {p_pooled_sharpe:.3f}")
        print(f"Observed overall total return: {overall_metrics['total_return']:.3f} | "
              f"pooled null p-value: {p_pooled_total_return:.3f}")

        pooled_perm_result = {
            "pooled_null_sharpe": pooled_null_sharpe,
            "pooled_null_total_return": pooled_null_total_return,
            "p_value_pooled_sharpe": p_pooled_sharpe,
            "p_value_pooled_total_return": p_pooled_total_return,
            "observed_sharpe": overall_metrics["sharpe"],
            "observed_total_return": overall_metrics["total_return"],
        }
        plot_pooled_permutation_null(pooled_perm_result, results_dir)
        pd.DataFrame({
            "pooled_null_sharpe": pooled_null_sharpe,
            "pooled_null_total_return": pooled_null_total_return,
        }).to_parquet(results_dir / "permutation_pooled_nulls.parquet")

    # --- Buy-and-hold benchmarks, aligned to the same out-of-sample dates ---
    spy_bh, universe_bh = build_buy_and_hold_benchmarks(data, benchmark_df, combined_returns.index)
    spy_metrics = compute_financial_metrics(spy_bh)
    universe_bh_metrics = compute_financial_metrics(universe_bh)
    print("\n=== Buy-and-hold benchmarks (same out-of-sample period) ===")
    print(f"SPY buy-and-hold:            total_return={spy_metrics['total_return']:.4f}, "
          f"sharpe={spy_metrics['sharpe']:.2f}, max_dd={spy_metrics['max_drawdown']:.4f}")
    print(f"Equal-weight universe hold:  total_return={universe_bh_metrics['total_return']:.4f}, "
          f"sharpe={universe_bh_metrics['sharpe']:.2f}, max_dd={universe_bh_metrics['max_drawdown']:.4f}")

    # --- Save results to disk ---
    results_df.to_parquet(results_dir / "fold_results.parquet")
    results_df.to_csv(results_dir / "fold_results.csv", index=False)
    combined_returns.to_frame().to_parquet(results_dir / "combined_portfolio_returns.parquet")
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_parquet(results_dir / "all_trades.parquet")
    if all_permutation_nulls:
        # store as one row per fold with the null arrays as list columns
        pd.DataFrame(all_permutation_nulls).to_parquet(results_dir / "permutation_nulls.parquet")

    benchmark_comparison = pd.DataFrame({
        "strategy": compute_financial_metrics(combined_returns),
        "spy_buy_and_hold": spy_metrics,
        "equal_weight_universe_buy_and_hold": universe_bh_metrics,
    })
    benchmark_comparison.to_csv(results_dir / "benchmark_comparison.csv")

    plot_fold_timeline(fold_windows, results_dir)

    if len(combined_returns) > 0 and combined_returns.abs().sum() > 0:
        try:
            qs.reports.html(
                combined_returns, benchmark=spy_bh,
                output=str(results_dir / "tearsheet_vs_spy.html"),
                title="Strategy vs SPY buy-and-hold",
            )
            qs.reports.html(
                combined_returns, benchmark=universe_bh,
                output=str(results_dir / "tearsheet_vs_universe_bh.html"),
                title="Strategy vs equal-weight universe buy-and-hold",
            )
            print(f"\nSaved tearsheets to {results_dir / 'tearsheet_vs_spy.html'} and "
                  f"{results_dir / 'tearsheet_vs_universe_bh.html'}")
        except Exception as e:
            print(f"[warn] quantstats tearsheet generation failed: {e}")

    print(f"All results saved to {results_dir}/")

    return results_df, combined_returns


if __name__ == "__main__":
    run_pipeline()
