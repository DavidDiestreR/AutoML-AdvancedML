"""auto-sklearn run script.

Short idea: replicate the same workflow as the other scripts for a fair comparison.
Recommended to run it on Linux/WSL if you are on Windows.
"""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_path = project_root / "data" / "results" / "autosklearn"
    results_path.mkdir(parents=True, exist_ok=True)

    # TODO: load processed data
    # TODO: train auto-sklearn with a reasonable time limit
    # TODO: save metrics/model/predictions
    print(f"auto-sklearn implementation pending. Output path: {results_path}")


if __name__ == "__main__":
    main()
