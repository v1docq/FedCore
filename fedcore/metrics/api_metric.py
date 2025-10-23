import torch
import pandas as pd

from typing import Optional

from fedcore.metrics.metric_impl import CLASSIFICATION_METRICS, REGRESSION_METRICS


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

def calculate_regression_metrics(
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

    values = {name: calculate_metric(target, predict, name) for name in metric_names if name in REGRESSION_METRICS}

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
    values = {}
    for name in metric_names:
        metric = CLASSIFICATION_METRICS[name]
        if name in ("logloss", "roc_auc"):
            if probs is None:
                raise ValueError(f"{name} requires `probs` argument.")
            values[name] = metric(target, probs)
        else:
            values[name] = metric(target, labels)

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

    ###################### TODO Here should be called the fedcore.tools.ruler.PerformanceEvaluator
    pass 
