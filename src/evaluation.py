"""Evaluation helpers for regression models."""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from src import models


def cross_validate_regression_model(
    model_name: str,
    cv,
    X_train,
    y_train,
    model_params: Dict[str, Any] | None = None,
) -> float:
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
    
    return float(rmse_array.mean())


def cross_validate_regression_metrics(
    model_name: str,
    cv,
    X_train,
    y_train,
    model_params: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Run manual CV and return mean regression metrics for a model."""

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
    fold_r2: List[float] = []
    fold_mae: List[float] = []
    fold_mse: List[float] = []
    fold_rmse: List[float] = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = model_cls(**params)
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)

        fold_r2.append(float(r2_score(y_fold_val, predictions)))
        fold_mae.append(float(mean_absolute_error(y_fold_val, predictions)))
        fold_mse.append(float(mean_squared_error(y_fold_val, predictions)))
        fold_rmse.append(float(root_mean_squared_error(y_fold_val, predictions)))

    return {
        "rmse": float(np.mean(fold_rmse)),
        "mse": float(np.mean(fold_mse)),
        "mae": float(np.mean(fold_mae)),
        "r2": float(np.mean(fold_r2)),
    }
