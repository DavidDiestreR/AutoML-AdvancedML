"""Custom AutoML search with simulated annealing.

This script loads a processed regression dataset, runs a local AutoML search
over multiple model families and their hyperparameters, and saves the search
trajectory/results in `data/results/automl/`.

The search uses simulated annealing acceptance, with a separate temperature
scaling factor for model-change proposals (to encourage switching between
model families independently from hyperparameter moves). Temperature is
decreased with a time-based schedule, so it cools as execution time advances
towards the configured runtime budget.
"""

import argparse
import copy
import inspect
import json
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import models
from src.evaluation import (
    cross_validate_regression_metrics,
    cross_validate_regression_model,
)


def _json_default(obj):
    """JSON serializer for NumPy values used in metrics/state dumps."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _default_init_params(model_cls) -> dict:
    """Return only mutable default params (the keys produced by neighbour())."""
    defaults = {}
    signature = inspect.signature(model_cls.__init__)
    for name, param in signature.parameters.items():
        if name == "self" or param.default is inspect._empty:
            continue
        defaults[name] = copy.deepcopy(param.default)
    model = model_cls(**defaults)
    probe = model.neighbour(np.random.default_rng(0))
    if not isinstance(probe, dict):
        raise TypeError(f"{model_cls.__name__}.neighbour(...) must return a dict.")
    return {key: copy.deepcopy(defaults[key]) for key in probe.keys()}


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


def _decrease_temperature(
    initial_temperature: float,
    elapsed_time: float,
    execution_time: float = 3600.0,
    min_temperature: float = 1e-4,
) -> float:
    """Time-based exponential cooling: reach T_min at `execution_time` seconds."""
    if execution_time <= 0:
        raise ValueError("execution_time must be > 0.")
    if elapsed_time <= 0:
        return float(max(min_temperature, initial_temperature))
    if elapsed_time >= execution_time:
        return float(min_temperature)
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0.")
    if initial_temperature < min_temperature:
        raise ValueError("initial_temperature must be >= min_temperature.")
    progress = float(np.clip(elapsed_time / execution_time, 0.0, 1.0))
    # Exponential interpolation in log-space keeps exploration higher early on.
    temperature = initial_temperature * ((min_temperature / initial_temperature) ** progress)
    return float(max(min_temperature, temperature))


def _generate_neighboring_state(
    current_iteration_row: dict[str, dict[str, object]],
    p_change_model: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    """
    Propose a neighboring state:
    - with probability p_change_model, switch model only (reuse params from current row)
    - otherwise, keep model and propose neighboring hyperparameters

    Returns: {"model": <model_name>, "params": <dict>}
    """
    model_class_map = {
        "ridge": models.RidgeRegressor,
        "knn": models.KNNRegressor,
        "randomforest": models.RandomForestRegressorSA,
        "mlp": models.MLP,
    }

    current_models = [
        model_name
        for model_name, model_state in current_iteration_row.items()
        if bool(model_state.get("current_model", False))
    ]
    if len(current_models) != 1:
        raise ValueError(
            "current_iteration_row must contain exactly one model with current_model=True."
        )
    current_model = current_models[0]

    current_model_params = current_iteration_row[current_model]["params"]

    should_switch_model = rng.random() < p_change_model and len(current_iteration_row) > 1

    chosen_model = current_model
    if should_switch_model:
        candidates = [name for name in current_iteration_row.keys() if name != current_model]
        chosen_model = str(rng.choice(candidates))
        # Model switch proposal: reuse the existing params in this iteration row.
        return {
            "model": chosen_model,
            "params": copy.deepcopy(current_iteration_row[chosen_model]["params"]),
        }

    # Hyperparameter proposal: keep the current model and perturb its params locally.
    base_params = copy.deepcopy(current_model_params)
    model_instance = model_class_map[current_model](**copy.deepcopy(base_params))
    neighbor_update = model_instance.neighbour(rng)

    new_params = copy.deepcopy(base_params)
    new_params.update(neighbor_update)
    return {"model": current_model, "params": new_params}


def main(
    dataset_name: str,
    k_fold: int = 3,
    temperature: float = 0.05,
    model_temperature_factor: float = 4.0,
    p_change_model: float = 0.35,
    execution_time: float = 3600.0,
    min_temperature: float = 1e-4,
    seed: int | None = None,
) -> None:
    if not dataset_name or not str(dataset_name).strip():
        raise ValueError("dataset_name is required and cannot be empty.")
    if k_fold < 2:
        raise ValueError("k_fold must be >= 2.")

    if seed is None:
        seed = int(time.time())

    if not (0.0 <= p_change_model <= 1.0):
        raise ValueError("p_change_model must be between 0 and 1.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if execution_time <= 0:
        raise ValueError("execution_time must be > 0.")
    if min_temperature <= 0:
        raise ValueError("min_temperature must be > 0.")
    if temperature < min_temperature:
        raise ValueError("temperature must be >= min_temperature.")
    if model_temperature_factor <= 0:
        raise ValueError("model_temperature_factor must be > 0.")
    rng = np.random.default_rng(seed)
    initial_temperature = float(temperature)

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "processed"
    results_path = project_root / "data" / "results" / "automl"
    results_path.mkdir(parents=True, exist_ok=True)


    # =========================== INITIALIZATION ===========================
    init_start_time = time.perf_counter()
    X_train, y_train, dataset_file, target_col = _load_processed_dataset(data_path, dataset_name)
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=seed)

    model_map = {
        "ridge": models.RidgeRegressor,
        "knn": models.KNNRegressor,
        "randomforest": models.RandomForestRegressorSA,
        "mlp": models.MLP,
    }

    state_matrix: dict[int, dict[str, dict[str, object]]] = {0: {}}
    # time step (int) -> model name (str) ->
    # {"params": dict, "rmse": float, "current_model": bool}
    for model_name, model_cls in model_map.items():
        default_params = _default_init_params(model_cls)
        rmse = cross_validate_regression_model(
            model_name=model_name,
            cv=cv,
            X_train=X_train,
            y_train=y_train,
            model_params=default_params,
        )
        state_matrix[0][model_name] = {
            "params": default_params,
            "rmse": rmse,
            "current_model": False,
        }

    current_model = str(rng.choice(list(model_map.keys())))
    for model_name in state_matrix[0]:
        state_matrix[0][model_name]["current_model"] = model_name == current_model
    if sum(int(v["current_model"]) for v in state_matrix[0].values()) != 1:
        raise ValueError("state_matrix row must contain exactly one current_model=True.")
    t = 0


    # =========================== AUTOML ===========================
    automl_start_time = time.perf_counter()
    while temperature > min_temperature:
        current_row = state_matrix[t]
        current_rmse = float(current_row[current_model]["rmse"])

        proposed_state = _generate_neighboring_state(
            current_iteration_row=current_row,
            p_change_model=p_change_model,
            rng=rng,
        )
        proposed_model = str(proposed_state["model"])
        proposed_params = copy.deepcopy(proposed_state["params"])
        is_model_change = proposed_model != current_model

        # If the proposal is a pure model switch, reuse the RMSE already stored in the row.
        if is_model_change:
            proposed_rmse = float(current_row[proposed_model]["rmse"])
        else:
            proposed_rmse = cross_validate_regression_model(
                model_name=proposed_model,
                cv=cv,
                X_train=X_train,
                y_train=y_train,
                model_params=proposed_params,
            )

        delta_rmse = proposed_rmse - current_rmse
        effective_temperature = (
            temperature * model_temperature_factor if is_model_change else temperature
        )

        if delta_rmse < 0:
            accept = True
        else:
            acceptance_prob = float(np.exp(-delta_rmse / effective_temperature))
            accept = rng.random() < acceptance_prob

        next_row = copy.deepcopy(current_row)
        for model_name in next_row:
            next_row[model_name]["current_model"] = False

        if accept:
            next_row[proposed_model]["params"] = proposed_params
            next_row[proposed_model]["rmse"] = proposed_rmse
            next_row[proposed_model]["current_model"] = True
            current_model = proposed_model
        else:
            next_row[current_model]["current_model"] = True

        if sum(int(v["current_model"]) for v in next_row.values()) != 1:
            raise ValueError("state_matrix row must contain exactly one current_model=True.")

        t += 1
        state_matrix[t] = next_row
        temperature = _decrease_temperature(
            initial_temperature=initial_temperature,
            elapsed_time=time.perf_counter() - automl_start_time,
            execution_time=execution_time,
            min_temperature=min_temperature,
        )


    # =========================== RESULTS ===========================
    # Save initialization state + summary derived from the last state-matrix iteration.
    last_iteration = max(state_matrix.keys())
    last_iteration_models = state_matrix[last_iteration]
    if not last_iteration_models:
        raise ValueError("Last state_matrix iteration is empty; cannot compute best model summary.")

    best_model_name, best_model_state = min(
        last_iteration_models.items(), key=lambda item: float(item[1]["rmse"])
    )
    elapsed_seconds = time.perf_counter() - init_start_time

    best_metrics = cross_validate_regression_metrics(
        model_name=best_model_name,
        cv=cv,
        X_train=X_train,
        y_train=y_train,
        model_params=best_model_state["params"],
    )

    run_summary = {
        "elapsed_time": elapsed_seconds,
        "num_iter": int(last_iteration),
        "best_model": best_model_name,
        "params": best_model_state["params"],
        "rmse": best_metrics["rmse"],
        "metrics": best_metrics,
    }

    state_matrix_file = results_path / f"{dataset_file.stem}_state_matrix.json"
    summary_file = results_path / f"{dataset_file.stem}_run_summary.json"

    with state_matrix_file.open("w", encoding="utf-8") as f:
        json.dump(state_matrix, f, indent=2, default=_json_default)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=_json_default)

    print(
        "AutoML run finished.\n"
        f" Project root: {project_root}\n"
        f" Data folder: {data_path}\n"
        f" Results folder: {results_path}\n"
        f" dataset_file: {dataset_file}\n"
        f" target_col: {target_col}\n"
        f" dataset_name: {dataset_name}\n"
        f" k_fold: {k_fold}\n"
        f" final_temperature: {temperature}\n"
        f" model_temperature_factor: {model_temperature_factor}\n"
        f" p_change_model: {p_change_model}\n"
        f" execution_time: {execution_time}\n"
        f" min_temperature: {min_temperature}\n"
        f" seed: {seed}\n"
        f" current_model (final): {current_model}\n"
        f" t (final): {t}\n"
        f" state_matrix_file: {state_matrix_file}\n"
        f" run_summary_file: {summary_file}\n"
        f" run_summary:\n{pformat(run_summary, sort_dicts=False)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AutoML search for a given dataset in data/results/automl/."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name to load from data/processed (mandatory).",
    )
    parser.add_argument(
        "-k",
        "--k-fold",
        type=int,
        default=3,
        help="Number of CV folds (default: 3).",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=0.05,
        help="Initial temperature for the search (default: 0.05).",
    )
    parser.add_argument(
        "-m",
        "--model-temperature-factor",
        type=float,
        default=4.0,
        help="Positive factor (>0) to scale model-change exploration temperature (default: 4.0).",
    )
    parser.add_argument(
        "-p",
        "--p-change-model",
        type=float,
        default=0.35,
        help="Probability of proposing a model change (default: 0.35).",
    )
    parser.add_argument(
        "-e",
        "--execution-time",
        type=float,
        default=3600.0,
        help="Target runtime in seconds to cool temperature down to Tmin (default: 3600).",
    )
    parser.add_argument(
        "-t",
        "--min-temperature",
        type=float,
        default=1e-4,
        help="Minimum temperature stopping threshold (default: 1e-4).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for RNG. If omitted, current time is used.",
    )

    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        k_fold=args.k_fold,
        temperature=args.temperature,
        model_temperature_factor=args.model_temperature_factor,
        p_change_model=args.p_change_model,
        execution_time=args.execution_time,
        min_temperature=args.min_temperature,
        seed=args.seed,
    )
