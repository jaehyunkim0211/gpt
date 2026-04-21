"""Evaluation metrics for forecasting and classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, label_names: dict | None = None):
    """Return dict of acc, macro-F1, confusion matrix, sklearn text report."""
    target_names = None
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if label_names is not None:
        target_names = [label_names[i] for i in labels]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
        "report": classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=3),
    }
