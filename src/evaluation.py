"""Evaluation and comparison helpers.

Idea: keep metrics, summary tables, and comparison functions here
to evaluate FLAML / TPOT / auto-sklearn using the same criteria.
"""

from __future__ import annotations

from typing import Dict

from sklearn.metrics import accuracy_score, f1_score


def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    """Evaluate a classification model with basic metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
