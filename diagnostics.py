"""
Feature diagnostics: correlation matrix and logistic-regression coefficient
stability across folds. Model-agnostic (correlation) / model-specific
(coefficients) checks run before any XGBoost switch, per STATE.md's
Next Steps -> A.

These are diagnostic only. Any feature drops motivated by this output
should be justified directly by the correlation/instability numbers -- not
by re-running the backtest with different feature subsets until something
looks better (same overfitting trap as k_upper/k_lower sweeps).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_correlation_matrix(pooled: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Pearson correlation matrix across the given feature columns, using
    the pooled (post-dropna) dataset actually fed to the model.
    """
    return pooled[feature_cols].corr()


def plot_correlation_matrix(corr: pd.DataFrame, results_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=5, color="white" if abs(val) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature correlation matrix (pooled, post-dropna)")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_correlation_matrix.png", dpi=150)
    plt.close(fig)


def highlight_momentum_cluster(corr: pd.DataFrame, momentum_cols: list) -> pd.DataFrame:
    """Subset of the correlation matrix restricted to the suspected
    momentum-redundant cluster (lagged returns, MA ratio, MA crossover,
    RSI, MACD), for quick inspection without scrolling the full matrix.
    """
    cols = [c for c in momentum_cols if c in corr.columns]
    return corr.loc[cols, cols]


def collect_coefficients(fold_models: list, feature_cols: list) -> pd.DataFrame:
    """fold_models: list of (fold_i, fitted_model, scaler) tuples from
    logistic-regression folds. Coefficients are in *scaled* feature space
    (LR is fit on standardized features), which is what we want for
    cross-feature stability comparison -- unscaled coefficients aren't
    comparable across features with different units anyway.

    model.coef_ has shape (n_classes, n_features). We report the "up"-class
    row (mapped label 2, per models.py's label_map: -1->0, 0->1, 1->2),
    since that's the class the strategy actually trades on.
    """
    rows = []
    for fold_i, model, _scaler in fold_models:
        up_class_mapped = 2
        class_idx = list(model.classes_).index(up_class_mapped)
        coefs = model.coef_[class_idx]
        row = {"fold": fold_i}
        row.update(dict(zip(feature_cols, coefs)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def plot_coefficient_stability(coef_df: pd.DataFrame, results_dir):
    """One line per feature across folds -- flat/consistent lines mean
    stable sign & magnitude; crossing zero or swinging wildly is the
    multicollinearity symptom we're checking for.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in coef_df.columns:
        ax.plot(coef_df.index, coef_df[col], marker="o", label=col)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Coefficient (up-class, scaled feature space)")
    ax.set_title("Logistic regression coefficient stability across folds")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(results_dir / "coefficient_stability.png", dpi=150)
    plt.close(fig)


def coefficient_stability_summary(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature summary: mean, std, |std/mean|, and sign-consistency
    (fraction of folds matching the majority sign). Low sign-consistency
    combined with a high std/mean ratio is the numeric version of "this
    line swings across zero in the plot."
    """
    summary = pd.DataFrame({
        "mean": coef_df.mean(),
        "std": coef_df.std(),
    })
    summary["cv_abs"] = summary["std"] / summary["mean"].abs().replace(0, np.nan)
    signs = np.sign(coef_df)
    majority_sign = signs.sum().apply(lambda s: 1 if s >= 0 else -1)
    summary["sign_consistency"] = signs.eq(majority_sign, axis=1).mean()
    return summary.sort_values("sign_consistency")
