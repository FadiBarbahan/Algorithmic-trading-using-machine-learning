"""
Walk-forward cross-validation with a purge/embargo gap.

Fixed window: train window slides forward, staying constant size.
Expanding window: train start stays fixed at 0, train end grows each fold.

Embargo: a gap of `embargo` periods is left between the end of train and the
start of test, dropped from *both* sides, so that labels whose forward-looking
window (horizon) overlaps the train/test boundary don't leak information
across the split. This matters here specifically because triple-barrier
labels look up to `horizon` days into the future -- a label near the
train/test boundary could otherwise "see" test-period returns.
"""

import numpy as np


def walk_forward_splits(
    n_samples: int,
    n_folds: int,
    embargo: int,
    min_train_size: int,
    window_type: str = "fixed",
):
    """Generate (train_idx, test_idx) index arrays for walk-forward CV.

    Args:
        n_samples: total number of rows in the (already time-sorted) dataset
        n_folds: number of walk-forward folds
        embargo: number of periods purged between train and test on both sides
        min_train_size: minimum training rows before the first fold starts
        window_type: "fixed" (sliding, constant-size train window) or
                     "expanding" (train start fixed at 0, end grows)

    Yields:
        (train_idx, test_idx) as numpy arrays of positional indices
    """
    if window_type not in ("fixed", "expanding"):
        raise ValueError("window_type must be 'fixed' or 'expanding'")

    usable = n_samples - min_train_size
    if usable <= 0:
        raise ValueError("min_train_size exceeds the dataset size.")

    fold_size = usable // n_folds
    if fold_size <= embargo:
        raise ValueError(
            f"fold_size ({fold_size}) must exceed embargo ({embargo}); "
            f"reduce n_folds or embargo, or provide more data."
        )

    for fold in range(n_folds):
        test_start = min_train_size + fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples

        if window_type == "fixed":
            train_start = fold * fold_size  # slides forward each fold
        else:  # expanding
            train_start = 0

        train_end = test_start - embargo
        test_start_embargoed = test_start  # embargo already carved out of train_end
        test_end_embargoed = test_end

        if train_end <= train_start:
            continue  # not enough room for this fold, skip it

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start_embargoed, min(test_end_embargoed, n_samples))

        # Also purge the tail of train that's within `embargo` of test start
        # (already handled by train_end = test_start - embargo above)

        if len(test_idx) == 0:
            continue

        yield train_idx, test_idx
