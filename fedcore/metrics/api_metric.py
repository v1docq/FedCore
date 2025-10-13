import torch
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)
from fedcore.metrics.metric_impl import SMAPE, MASE, MAPE

from fedcore.metrics.metric_impl import QualityMetric
from fedcore.metrics.metric_impl import (
    Accuracy,
    F1,
    Precision,
    Logloss,
    ROCAUC,
)
from typing import Optional

# -------------------- Define Metric Calculation Function --------------------

def calculate_metric(a: torch.Tensor, b: torch.Tensor, metric_type: str) -> float:
    """
    Compute various metrics based on the metric_type.

    Args:
        a (torch.Tensor): Ground truth values.
        b (torch.Tensor): Predicted values.
        metric_type (str): The type of metric to compute.

    Returns:
        float: The computed metric value.
    """
    a, b = a.float(), b.float()

    if metric_type == "mse":
        return torch.mean((a - b) ** 2).item()
    elif metric_type == "rmse":
        return torch.sqrt(torch.mean((a - b) ** 2)).item()
    elif metric_type == "mae":
        return torch.mean(torch.abs(a - b)).item()
    elif metric_type == "msle":
        return torch.mean(torch.log1p(a) - torch.log1p(b)).item()
    elif metric_type == "mape":
        return torch.mean(torch.abs((a - b) / a)).item()
    elif metric_type == "r2":
        # Calculate R2 score using PyTorch
        ss_total = torch.sum((a - torch.mean(a)) ** 2)
        ss_residual = torch.sum((a - b) ** 2)
        return 1 - (ss_residual / ss_total).item()
    elif metric_type == "accuracy":
        # Accuracy calculation using PyTorch
        return float(torch.mean((a == b).float()).item())
    elif metric_type == "precision":
        # Precision calculation using PyTorch
        tp = torch.sum((a == 1) & (b == 1))
        fp = torch.sum((a == 0) & (b == 1))
        return float(tp / (tp + fp + 1e-8))
    elif metric_type == "f1":
        # F1 calculation using PyTorch
        tp = torch.sum((a == 1) & (b == 1))
        fp = torch.sum((a == 0) & (b == 1))
        fn = torch.sum((a == 1) & (b == 0))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * (precision * recall) / (precision + recall + 1e-8)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

# -------------------- Metric Names Dictionary --------------------

# Map string names to the appropriate metric calculation function
METRIC_NAMES = {
    "r2": "r2",
    "mse": "mse",
    "rmse": "rmse",
    "mae": "mae",
    "msle": "msle",
    "mape": "mape",
    "accuracy": "accuracy",  # Add accuracy metric here
    "precision": "precision",  # Add precision metric here
    "f1": "f1",  # Add F1 metric here
}

# -------------------- Utility Function: Convert to DataFrame --------------------

def _to_df(values: dict, rounding: int = 3) -> pd.DataFrame:
    """
    Convert a dictionary of metric values into a DataFrame with one row.
    
    Args:
        values (dict): A dictionary of metric names and values.
        rounding (int): The number of decimal places to round the metric values.
    
    Returns:
        pd.DataFrame: A DataFrame with one row of rounded metric values.
    """
    return pd.DataFrame(values, index=[0]).round(rounding)

# -------------------- General Metric Calculation Function --------------------

def calculate_metrics(
    target: torch.Tensor,
    predict: torch.Tensor,
    rounding_order: int = 3,
    metric_names: tuple[str, ...] = ("r2", "rmse", "mae"),
    is_forecasting: bool = False
) -> pd.DataFrame:
    """
    Compute metrics for regression or forecasting (time series) tasks.

    Args:
        target (torch.Tensor): Ground truth values.
        predict (torch.Tensor): Predicted values.
        rounding_order (int): Decimal places for rounding the results.
        metric_names (tuple): Metrics to compute.
        is_forecasting (bool): Flag to indicate if it is forecasting or not.

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics.
    """
    target = target.float()
    predict = predict.float()

    # Check if MSLE can be used (non-negative values only)
    if "msle" in metric_names and ((target < 0).any() or (predict < 0).any()):
        raise ValueError("MSLE requires non-negative values for both target and prediction.")

    values = {name: calculate_metric(target, predict, name) for name in metric_names if name in METRIC_NAMES}

    return _to_df(values, rounding_order)

# -------------------- Classification Metrics Calculation --------------------

def calculate_classification_metrics(
    target: torch.Tensor,
    labels: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    rounding_order: int = 3,
    metric_names: tuple[str, ...] = ("f1", "accuracy")
) -> pd.DataFrame:
    """
    Compute classification metrics such as Accuracy, F1, Logloss, etc.

    Args:
        target (torch.Tensor): Ground truth labels.
        labels (torch.Tensor): Predicted labels.
        probs (torch.Tensor, optional): Predicted probabilities (needed for logloss and ROC AUC).
        rounding_order (int): Decimal places for rounding the results.
        metric_names (tuple): Metrics to compute.

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics.
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

# -------------------- Computational Metrics Calculation --------------------

def calculate_computational_metrics(model, dataset, model_regime: str) -> float:
    """
    Compute computational metrics like latency and throughput.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset: The dataset to evaluate on.
        model_regime (str): The regime of the model.

    Returns:
        float: The computed computational metric (e.g., throughput or latency).
    """
    try:
        from fedcore.metrics.cv_metrics import CV_quality_metric
    except ImportError:
        raise ImportError("Computational metrics require additional dependencies.")
    
    return CV_quality_metric().metric(model, dataset, model_regime)
