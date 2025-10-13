import importlib
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
from typing import Optional, Any, Dict, Sequence

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
        ss_total = torch.sum((a - torch.mean(a)) ** 2)
        ss_residual = torch.sum((a - b) ** 2)
        return 1 - (ss_residual / ss_total).item()
    elif metric_type == "accuracy":
        # Accuracy calculation using PyTorch (convert boolean to float)
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

# ============================== Lazy Loading of `evaluate` =====================

_EVALUATE = None

def _get_evaluate():
    """Lazy load the `evaluate` package."""
    global _EVALUATE
    if _EVALUATE is None:
        try:
            _EVALUATE = importlib.import_module("evaluate")
        except ImportError as e:
            raise ImportError("The 'evaluate' package is required for NLP metrics. Install it using `pip install evaluate`.") from e
    return _EVALUATE

# =============================== NLP Metrics ================================

class EvaluateMetric(QualityMetric):
    """Base class for NLP metrics powered by HuggingFace `evaluate`, compatible with QualityMetric."""
    default_value: float = 0.0
    need_to_minimize: bool = False
    split: str = "val"
    output_mode: str = "texts"  # default for text generation

    metric_name: str = ""
    load_kwargs: Dict[str, Any] = {}
    result_key: Optional[str] = None

    @classmethod
    def _hf_metric(cls):
        """Lazy load the specific HuggingFace metric."""
        evaluate = _get_evaluate()
        return evaluate.load(cls.metric_name, **cls.load_kwargs)

    @classmethod
    def metric(cls, target: Sequence[Any] | None, predict: Sequence[Any] | None, **kwargs: Any) -> float:
        references = kwargs.pop("references", None)
        predictions = kwargs.pop("predictions", None)
        if references is None:
            references = target
        if predictions is None:
            predictions = predict
        if references is None or predictions is None:
            raise ValueError("Both references and predictions are required.")

        res = cls._hf_metric().compute(references=references, predictions=predictions, **kwargs)

        # Normalize evaluate output to float
        if isinstance(res, dict):
            if cls.result_key is not None:
                val = res[cls.result_key]
            elif "score" in res:
                val = res["score"]
            elif len(res) == 1:
                val = next(iter(res.values()))
            else:
                raise ValueError(
                    f"{cls.__name__}: specify result_key (multiple fields found: {list(res.keys())})"
                )
        else:
            val = res
        return float(val)

    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        out = pipeline.predict(reference_data, output_mode=cls.output_mode)
        preds = out.predict.predict

        try:
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().tolist()
        except Exception:
            pass

        loader = getattr(reference_data.features, f"{cls.split}_dataloader")
        ds = loader.dataset

        if hasattr(ds, "references"):
            refs = ds.references
        elif hasattr(ds, "targets"):
            refs = ds.targets
        elif hasattr(ds, "labels"):
            refs = ds.labels
        else:
            it = iter(ds)
            refs = [ex[1] for ex in it]

        return cls.metric(refs, preds)

    def __init__(self, **override_load_kwargs: Any) -> None:
        params = dict(self.load_kwargs)
        params.update(override_load_kwargs)
        self._metric_inst = _get_evaluate().load(self.metric_name, **params)

    def compute(self, y_true: Sequence[Any] | None = None, y_pred: Sequence[Any] | None = None, *, references: Sequence[Any] | None = None, predictions: Sequence[Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
        if references is None and y_true is not None:
            references = y_true
        if predictions is None and y_pred is not None:
            predictions = y_pred
        if references is None or predictions is None:
            raise ValueError("Both references and predictions are required.")
        return self._metric_inst.compute(predictions=predictions, references=references, **kwargs)


class NLPAccuracy(EvaluateMetric):
    """Classification accuracy for NLP tasks."""
    metric_name = "accuracy"
    result_key = "accuracy"
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        evaluate_module = _get_evaluate()
        accuracy_metric = evaluate_module.load("accuracy")
        result = accuracy_metric.compute(references=target, predictions=predict)
        return result["accuracy"]
    
# =============================== Additional NLP Metrics ================================

class NLPPrecision(EvaluateMetric):
    """Precision for NLP classification tasks."""
    metric_name = "precision"
    result_key = "precision"
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        evaluate_module = _get_evaluate()
        precision_metric = evaluate_module.load("precision")
        result = precision_metric.compute(references=target, predictions=predict, average='macro')
        return result["precision"]

class NLPRecall(EvaluateMetric):
    """Recall for NLP classification tasks."""
    metric_name = "recall"
    result_key = "recall"
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        evaluate_module = _get_evaluate()
        recall_metric = evaluate_module.load("recall")
        result = recall_metric.compute(references=target, predictions=predict, average='macro')
        return result["recall"]

class NLPF1(EvaluateMetric):
    """F1 score for NLP classification tasks."""
    metric_name = "f1"
    result_key = "f1"
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        evaluate_module = _get_evaluate()
        f1_metric = evaluate_module.load("f1")
        result = f1_metric.compute(references=target, predictions=predict, average='macro')
        return result["f1"]