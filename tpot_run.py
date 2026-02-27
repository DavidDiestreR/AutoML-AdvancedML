"""TPOT regression run script.

Loads a processed CSV dataset, runs TPOTRegressor with a time budget,
evaluates test RMSE on a hold-out split, and writes artifacts to
`data/results/tpot/`.
"""

from __future__ import annotations

import argparse
import inspect
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


def _build_tpot_regressor(
    max_time_mins: float,
    k_fold: int,
    random_state: int | None,
    verbosity: int,
):
    """
    Build TPOTRegressor with compatibility across TPOT versions.

    The user-facing defaults prefer `max_time_mins` style when available.
    """
    try:
        from tpot import TPOTRegressor
    except ImportError as exc:
        raise ImportError(
            "TPOT is not installed. Install with `pip install tpot` (or project requirements)."
        ) from exc

    signature = inspect.signature(TPOTRegressor.__init__)
    fixed_n_jobs = max(1, int(os.cpu_count() or 1))
    kwargs = {
        "cv": k_fold,
        "random_state": random_state,
        "n_jobs": fixed_n_jobs,
    }
    # TPOT versions differ here: some use `verbosity`, others `verbose`.
    if "verbosity" in signature.parameters:
        kwargs["verbosity"] = verbosity
    elif "verbose" in signature.parameters:
        kwargs["verbose"] = verbosity
    if "max_time_mins" in signature.parameters:
        kwargs["max_time_mins"] = max_time_mins
    else:
        # Legacy fallback: approximate a time-budgeted run.
        kwargs["generations"] = 100
        kwargs["population_size"] = 50

    return TPOTRegressor(**kwargs), fixed_n_jobs


def main(
    dataset_name: str,
    test_size: float = 0.2,
    k_fold: int = 3,
    execution_time: float = 3600.0,
    seed: int | None = None,
    verbosity: int = 2,
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
    results_path = project_root / "data" / "results" / "tpot"
    results_path.mkdir(parents=True, exist_ok=True)

    X, y, dataset_file, target_col = _load_processed_dataset(data_path, dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    max_time_mins = float(execution_time) / 60.0
    tpot, fixed_n_jobs = _build_tpot_regressor(
        max_time_mins=max_time_mins,
        k_fold=k_fold,
        random_state=seed,
        verbosity=verbosity,
    )
    tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    elapsed_seconds = time.perf_counter() - start_time
    run_summary = {
        "elapsed_time": elapsed_seconds,
        "dataset_name": dataset_file.name,
        "target_col": target_col,
        "test_size": test_size,
        "k_fold": k_fold,
        "execution_time": execution_time,
        "max_time_mins": max_time_mins,
        "seed": seed,
        "metrics": {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        },
    }

    summary_file = results_path / f"{dataset_file.stem}_run_summary.json"
    pipeline_file = results_path / f"{dataset_file.stem}_best_pipeline.py"
    pipeline_fallback_file = results_path / f"{dataset_file.stem}_best_pipeline.txt"

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=_json_default)

    pipeline_export_mode = "none"
    exported_pipeline_path = None
    if hasattr(tpot, "export"):
        tpot.export(str(pipeline_file))
        pipeline_export_mode = "python_export"
        exported_pipeline_path = str(pipeline_file)
    else:
        fitted_pipeline = getattr(tpot, "fitted_pipeline_", None)
        pipeline_repr = repr(fitted_pipeline) if fitted_pipeline is not None else "None"
        pipeline_fallback_file.write_text(pipeline_repr + "\n", encoding="utf-8")
        pipeline_export_mode = "repr_fallback"
        exported_pipeline_path = str(pipeline_fallback_file)

    run_summary["pipeline_export_mode"] = pipeline_export_mode
    run_summary["pipeline_file"] = exported_pipeline_path
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=_json_default)

    print(
        "TPOT run finished.\n"
        f" Project root: {project_root}\n"
        f" Data folder: {data_path}\n"
        f" Results folder: {results_path}\n"
        f" dataset_file: {dataset_file}\n"
        f" target_col: {target_col}\n"
        f" dataset_name: {dataset_name}\n"
        f" k_fold: {k_fold}\n"
        f" execution_time: {execution_time}\n"
        f" max_time_mins: {max_time_mins}\n"
        f" test_size: {test_size}\n"
        f" seed: {seed}\n"
        f" n_jobs: {fixed_n_jobs} (fixed to cpu_count)\n"
        f" run_summary_file: {summary_file}\n"
        f" pipeline_file: {exported_pipeline_path}\n"
        f" run_summary:\n{pformat(run_summary, sort_dicts=False)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TPOT regression for a dataset in data/processed."
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
        help="CV folds used by TPOT during search (default: 3).",
    )
    parser.add_argument(
        "-e",
        "--execution-time",
        type=float,
        default=3600.0,
        help="Total TPOT search time in seconds (default: 3600).",
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
        help="TPOT verbosity level (default: 2).",
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
