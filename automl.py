"""AutoML training (e.g., FLAML).

Short idea: load the processed dataset, train AutoML, and save metrics
+ model/results in `data/results/automl/`.
"""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "processed"
    results_path = project_root / "data" / "results" / "automl"
    results_path.mkdir(parents=True, exist_ok=True)

    # TODO: load X_train, X_test, y_train, y_test from `data/processed/`
    # TODO: train AutoML (for example, FLAML)
    # TODO: save metrics/predictions to `results_path`
    print(f"Implementation pending. Data folder: {data_path}")


if __name__ == "__main__":
    main()
