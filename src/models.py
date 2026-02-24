"""Model helpers / common configurations.

Idea: centralize model creation, parameter spaces,
and/or helper functions that return baseline pipelines to avoid duplicated logic.
"""

from __future__ import annotations

from typing import Any, Dict


def get_baseline_config(task_type: str = "classification") -> Dict[str, Any]:
    """Return a minimal baseline config for quick experiments."""
    if task_type == "regression":
        return {
            "metric": "rmse",
            "cv_folds": 5,
        }
    return {
        "metric": "f1_weighted",
        "cv_folds": 5,
    }
