"""Evaluation helpers for regression models."""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import root_mean_squared_error

from src import models


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Evaluate a regression model using RMSE only."""
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"rmse": float(rmse)}


def _build_model(model_name: str, model_params: Dict[str, Any] | None = None):
    """Instantiate a model class defined in src.models by name."""
    normalized_name = model_name.strip().lower()
    params = dict(model_params or {})

    model_map = {
        "ridge": models.RidgeRegressor,
        "knn": models.KNNRegressor,
        "randomforest": models.RandomForestRegressorSA,
        "mlp": models.MLP,
    }

    if normalized_name not in model_map:
        supported = ", ".join(model_map.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of: {supported}")

    if normalized_name == "knn":
        if "k" in params:
            params["n_neighbors"] = params.pop("k")
        if "dist" in params:
            dist_value = str(params.pop("dist")).lower()
            dist_to_p = {
                "manhattan": 1,
                "euclidean": 2,
                "l1": 1,
                "l2": 2,
                "1": 1,
                "2": 2,
            }
            if dist_value in dist_to_p:
                params["p"] = dist_to_p[dist_value]
            else:
                raise ValueError(
                    "Unsupported KNN 'dist'. Use one of: manhattan, euclidean, l1, l2, 1, 2"
                )

    return model_map[normalized_name](**params)


def cross_validate_regression_model(
    model_name: str,
    cv,
    X_train,
    y_train,
    model_params: Dict[str, Any] | None = None,
) -> Dict[str, object]:
    """Run manual CV for a regression model using a provided splitter (e.g., KFold)."""
    fold_rmses: List[float] = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = _build_model(model_name, model_params)
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)

        fold_rmse = root_mean_squared_error(y_fold_val, predictions)
        fold_rmses.append(float(fold_rmse))

    rmse_array = np.array(fold_rmses, dtype=float)
    return {
        "model_name": model_name.strip().lower(),
        "model_params": dict(model_params or {}),
        "rmse_per_fold": fold_rmses,
        "rmse_mean": float(rmse_array.mean()),
        "rmse_std": float(rmse_array.std(ddof=0)),
    }
