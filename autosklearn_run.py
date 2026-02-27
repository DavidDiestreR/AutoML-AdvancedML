"""auto-sklearn regression run script.

Loads a processed CSV dataset, runs AutoSklearnRegressor with a time budget,
evaluates test RMSE on a hold-out split, and writes artifacts to
`data/results/autosklearn/`.

Designed to mirror the structure and outputs of `tpot_run.py` for fair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _json_default(obj):
    """JSON serializer for NumPy values used in output dumps."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_processed_dataset(data_path: Path, dataset_name: str):
    """Load a processed CSV dataset and split it into X/y."""
    dataset_file = data_path / dataset_name
    if dataset_file.suffix.lower() != ".csv":
        raise ValueError("dataset_name must include the .csv extension.")
    if not dataset_file.exists() or not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in data/processed.")

    df = pd.read_csv(dataset_file)
    if df.shape[1] < 2:
        raise ValueError(
            f"Dataset '{dataset_file.name}' must have at least 2 columns (features + target)."
        )

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, dataset_file, target_col


def _build_autosklearn_regressor(
    execution_time: float,
    k_fold: int,
    seed: int,
    n_jobs: int,
):
    """Build AutoSklearnRegressor with a sensible per-run limit and CV resampling."""
    try:
        import autosklearn.regression
    except ImportError as exc:
        raise ImportError(
            "auto-sklearn is not installed (or not supported on this OS). "
            "Install it in Linux/WSL with `conda install -c conda-forge auto-sklearn`."
        ) from exc

    # Per-model run limit: keep it bounded so one model doesn't eat the whole budget.
    # If execution_time is small, scale down.
    per_run_time_limit = int(min(180, max(30, execution_time / 10)))

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=int(execution_time),
        per_run_time_limit=per_run_time_limit,
        seed=seed,
        n_jobs=n_jobs,
        # Match a "k-fold CV during search" style comparison
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": int(k_fold)},
        # Keep auto-sklearn default ensemble behavior; we log it in the run summary.
        # If you want to disable ensembling, set ensemble_size=0 here.
    )
    return automl, per_run_time_limit


def main(
    dataset_name: str,
    test_size: float = 0.2,
    k_fold: int = 5,
    execution_time: float = 3600.0,
    seed: int | None = None,
    verbosity: int = 2,  # kept for CLI compatibility with tpot_run.py
) -> None:
    if not dataset_name or not str(dataset_name).strip():
        raise ValueError("dataset_name is required and cannot be empty.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    if k_fold < 2:
        raise ValueError("k_fold must be >= 2.")
    if execution_time <= 0:
        raise ValueError("execution_time must be > 0.")
    if seed is None:
        seed = int(time.time())

    start_time = time.perf_counter()
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "processed"
    results_path = project_root / "data" / "results" / "autosklearn"
    results_path.mkdir(parents=True, exist_ok=True)

    X, y, dataset_file, target_col = _load_processed_dataset(data_path, dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    fixed_n_jobs = max(1, int(os.cpu_count() or 1))
    automl, per_run_time_limit = _build_autosklearn_regressor(
        execution_time=execution_time,
        k_fold=k_fold,
        seed=seed,
        n_jobs=fixed_n_jobs,
    )

    # auto-sklearn expects numeric matrices; your processed CSV should already be numeric.
    automl.fit(X_train, y_train)

    y_pred = automl.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    elapsed_seconds = time.perf_counter() - start_time

    # Prepare outputs (mirror tpot_run.py style)
    summary_file = results_path / f"{dataset_file.stem}_run_summary.json"
    models_txt_file = results_path / f"{dataset_file.stem}_show_models.txt"
    stats_txt_file = results_path / f"{dataset_file.stem}_statistics.txt"
    preds_file = results_path / f"{dataset_file.stem}_test_predictions.csv"


    # These are great for your report (they show what it found / ensemble)
    try:
        models_txt_file.write_text(str(automl.show_models()) + "\n", encoding="utf-8")
    except Exception as exc:
        models_txt_file.write_text(
            f"show_models() failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
    try:
        stats_txt_file.write_text(str(automl.sprint_statistics()) + "\n", encoding="utf-8")
    except Exception as exc:
        stats_txt_file.write_text(
            f"sprint_statistics() failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )

    # Save test predictions for later plotting/analysis
    pd.DataFrame({"y_true": y_test.values, "y_pred": np.asarray(y_pred)}).to_csv(
        preds_file, index=False
    )

    run_summary = {
        "elapsed_time": elapsed_seconds,
        "dataset_name": dataset_file.name,
        "target_col": target_col,
        "test_size": test_size,
        "k_fold": k_fold,
        "execution_time": execution_time,
        "per_run_time_limit": per_run_time_limit,
        "seed": seed,
        "verbosity_arg_kept_for_cli_compat": verbosity,
        "n_jobs": fixed_n_jobs,
        "metrics": {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        },
        "show_models_file": str(models_txt_file),
        "statistics_file": str(stats_txt_file),
        "predictions_file": str(preds_file),
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=_json_default)

    print(
        "auto-sklearn run finished.\n"
        f" Project root: {project_root}\n"
        f" Data folder: {data_path}\n"
        f" Results folder: {results_path}\n"
        f" dataset_file: {dataset_file}\n"
        f" target_col: {target_col}\n"
        f" dataset_name: {dataset_name}\n"
        f" k_fold: {k_fold}\n"
        f" execution_time: {execution_time}\n"
        f" per_run_time_limit: {per_run_time_limit}\n"
        f" test_size: {test_size}\n"
        f" seed: {seed}\n"
        f" n_jobs: {fixed_n_jobs} (fixed to cpu_count)\n"
        f" run_summary_file: {summary_file}\n"
        f" show_models_file: {models_txt_file}\n"
        f" statistics_file: {stats_txt_file}\n"
        f" predictions_file: {preds_file}\n"
        f" run_summary:\n{pformat(run_summary, sort_dicts=False)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run auto-sklearn regression for a dataset in data/processed."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name to load from data/processed (must include .csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for test split (default: 0.2).",
    )
    parser.add_argument(
        "-k",
        "--k-fold",
        type=int,
        default=3,
        help="CV folds used by auto-sklearn during search (default: 3).",
    )
    parser.add_argument(
        "-e",
        "--execution-time",
        type=float,
        default=3600.0,
        help="Total auto-sklearn search time in seconds (default: 3600).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, current time is used.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Kept for CLI compatibility with tpot_run.py (auto-sklearn ignores it).",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        k_fold=args.k_fold,
        execution_time=args.execution_time,
        seed=args.seed,
        verbosity=args.verbosity,
    )