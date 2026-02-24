"""TPOT run script.

Short idea: use the same processed dataset for a fair comparison
and export metrics/pipeline to `data/results/tpot/`.
"""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_path = project_root / "data" / "results" / "tpot"
    results_path.mkdir(parents=True, exist_ok=True)

    # TODO: load processed data
    # TODO: configure TPOTClassifier/TPOTRegressor
    # TODO: train and save results / exported pipeline
    print(f"TPOT implementation pending. Output path: {results_path}")


if __name__ == "__main__":
    main()
