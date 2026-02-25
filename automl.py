"""AutoML training (e.g., FLAML).

Short idea: load the processed dataset, train AutoML, and save metrics
+ model/results in `data/results/automl/`.
"""

from pathlib import Path
import time


def main(
    temperature: float = 1.0,
    model_temperature_factor: float = 1.5,
    p_change_model: float = 0.2,
    min_temperature: float = 1e-3,
    seed: int | None = None,
) -> None:
    if seed is None:
        seed = int(time.time())

    if not (0.0 <= p_change_model <= 1.0):
        raise ValueError("p_change_model must be between 0 and 1.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if min_temperature <= 0:
        raise ValueError("min_temperature must be > 0.")
    if model_temperature_factor <= 0:
        raise ValueError("model_temperature_factor must be > 0.")

    p_change_hyperparams = 1.0 - p_change_model

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "processed"
    results_path = project_root / "data" / "results" / "automl"
    results_path.mkdir(parents=True, exist_ok=True)

    # TODO: load X_train, X_test, y_train, y_test from `data/processed/`
    # TODO: train AutoML (for example, FLAML)
    # TODO: save metrics/predictions to `results_path`
    print(
        "Implementation pending.\n"
        f" Project root: {project_root}\n"
        f" Data folder: {data_path}\n"
        f" Results folder: {results_path}\n"
        f" temperature: {temperature}\n"
        f" model_temperature_factor: {model_temperature_factor}\n"
        f" p_change_model: {p_change_model}\n"
        f" p_change_hyperparams: {p_change_hyperparams}\n"
        f" min_temperature: {min_temperature}\n"
        f" seed: {seed}"
    )


if __name__ == "__main__":
    main()
