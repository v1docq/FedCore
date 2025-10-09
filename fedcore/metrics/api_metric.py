"""
Public API functions for computing metrics on regression, forecasting,
classification and computational tasks.

Each function returns a one-row pandas DataFrame with rounded metric values.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from fedcore.metrics.metric_impl import (
    Accuracy,
    F1,
    Precision,
    Logloss,
    ROCAUC,
    smape,
    mase,
    mape,
)


def _to_df(values: dict, rounding: int = 3) -> pd.DataFrame:
    """Convert dict of metric values into a one-row DataFrame with rounding."""
    return pd.DataFrame(values, index=[0]).round(rounding)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def calculate_regression_metric(
    target,
    predict,
    rounding_order: int = 3,
    metric_names: tuple[str, ...] = ("r2", "rmse", "mae"),
) -> pd.DataFrame:
    """
    Compute regression metrics.

    Args:
        target: Ground truth array.
        predict: Predictions array.
        rounding_order: Decimal places for rounding.
        metric_names: Metrics to compute.
            Options: "r2", "mse", "rmse", "mae", "msle", "mape".
    """
    target = np.asarray(target, dtype=float)
    predict = np.asarray(predict, dtype=float)

    if "msle" in metric_names and ((target < 0).any() or (predict < 0).any()):
        raise ValueError("MSLE requires non-negative target and prediction.")

    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    metrics = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "rmse": rmse,
        "mae": mean_absolute_error,
        "msle": mean_squared_log_error,
        "mape": mean_absolute_percentage_error,
    }
    values = {n: metrics[n](target, predict) for n in metric_names if n in metrics}
    return _to_df(values, rounding_order)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def calculate_forecasting_metric(
    target,
    predict,
    rounding_order: int = 3,
    metric_names: tuple[str, ...] = ("rmse", "mae", "smape"),
) -> pd.DataFrame:
    """
    Compute forecasting (time series) metrics.

    Args:
        target: Ground truth series.
        predict: Predicted series.
        rounding_order: Decimal places for rounding.
        metric_names: Metrics to compute.
            Options: "rmse", "mae", "smape", "mase", "mape".
    """
    target = np.asarray(target, dtype=float)
    predict = np.asarray(predict, dtype=float)

    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    metrics = {
        "rmse": rmse,
        "mae": mean_absolute_error,
        "smape": smape,
        "mase": mase,
        "mape": mape,
    }
    values = {n: metrics[n](target, predict) for n in metric_names if n in metrics}
    return _to_df(values, rounding_order)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def calculate_classification_metric(
    target,
    labels,
    probs=None,
    rounding_order: int = 3,
    metric_names: tuple[str, ...] = ("f1", "accuracy"),
) -> pd.DataFrame:
    """
    Compute classification metrics.

    Args:
        target: Ground truth labels.
        labels: Predicted labels.
        probs: Predicted probabilities (needed for logloss, roc_auc).
        rounding_order: Decimal places for rounding.
        metric_names: Metrics to compute.
            Options: "accuracy", "f1", "precision", "logloss", "roc_auc".
    """
    metrics = {
        "accuracy": Accuracy().metric,
        "f1": F1().metric,
        "precision": Precision().metric,
        "logloss": Logloss().metric,
        "roc_auc": ROCAUC().metric,
    }
    values = {}
    for name in metric_names:
        if name in ("logloss", "roc_auc"):
            if probs is None:
                raise ValueError(f"{name} requires `probs` argument.")
            values[name] = metrics[name](target, probs)
        elif name in metrics:
            values[name] = metrics[name](target, labels)
    return _to_df(values, rounding_order)


# ---------------------------------------------------------------------------
# Computational / CV metrics
# ---------------------------------------------------------------------------

def calculate_computational_metric(model, dataset, model_regime):
    """
    Compute computational metrics such as latency or throughput.

    Requires optional CV dependencies. Raises ImportError if unavailable.
    """
    try:
        from fedcore.metrics.cv_metrics import CV_quality_metric
    except Exception as e:
        raise ImportError(
            "Computational metrics require additional dependencies."
        ) from e

    return CV_quality_metric().metric(model, dataset, model_regime)
