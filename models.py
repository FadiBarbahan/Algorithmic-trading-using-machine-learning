"""
Model wrapper. Start with logistic regression to validate the pipeline
end-to-end; switch to xgboost via config["model"]["type"] once the plumbing
is confirmed correct.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_model(cfg: dict):
    """Build an (unfitted) model + scaler pair based on config["model"].

    Logistic regression needs scaled features; tree-based models (xgboost)
    don't, so we only scale when needed.
    """
    model_type = cfg["type"]

    if model_type == "logistic_regression":
        params = cfg["logistic_regression"]
        model = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
        )
        scaler = StandardScaler()
        return model, scaler

    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost --break-system-packages"
            ) from e

        params = cfg["xgboost"]
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )
        return model, None  # no scaling needed for tree models

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def fit_predict(model, scaler, X_train, y_train, X_test):
    """Fit on train, predict on test. Handles the optional scaler and remaps
    labels {-1,0,1} <-> {0,1,2} since some classifiers (xgboost) require
    non-negative contiguous class labels.
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    inverse_map = {v: k for k, v in label_map.items()}

    y_train_mapped = y_train.map(label_map)

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train_mapped)
    preds_mapped = model.predict(X_test)
    preds = [inverse_map[p] for p in preds_mapped]

    return preds
