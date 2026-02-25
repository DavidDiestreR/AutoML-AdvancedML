"""Evaluation helpers for regression models."""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import root_mean_squared_error

from src import models


def cross_validate_regression_model(
    model_name: str,
    cv,
    X_train,
    y_train,
    model_params: Dict[str, Any] | None = None,
) -> Dict[str, object]:
    """Run manual CV for a regression model using a provided splitter."""
    
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

    model_cls = model_map[normalized_name]
    fold_rmses: List[float] = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = model_cls(**params)
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)

        fold_rmse = root_mean_squared_error(y_fold_val, predictions)
        fold_rmses.append(float(fold_rmse))

    rmse_array = np.array(fold_rmses, dtype=float)
    return {
        "model_name": normalized_name,
        "params": params,
        "rmse": float(rmse_array.mean())
    }
