"""
Core implementations of regression, classification, and time series metrics.

All metrics inherit from `QualityMetric` base class and provide a unified
interface: `metric(target, predict) -> float` plus `.get_value(...)` to
integrate with pipelines.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from torch import Tensor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    roc_auc_score,
    precision_recall_fscore_support,  # used in counters
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QualityMetric:
    """
    Base metric that can be computed directly or via pipeline evaluation.

    Subclasses override `metric(target, predict)` to return a scalar score.
    """

    default_value: float = 0.0
    need_to_minimize: bool = False
    output_mode: str = "compress"   # 'labels', 'probs', 'raw', 'compress'
    split: str = "val"              # 'val' or 'test'

    @classmethod
    def metric(cls, target, predict) -> float:
        """Compute metric value from arrays of target and prediction."""
        raise NotImplementedError

    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        """
        Run `pipeline.predict` and compute the metric on reference_data.

        Uses `features.<split>_dataloader` to obtain true labels.
        """
        results = pipeline.predict(reference_data, output_mode=cls.output_mode)
        loader = getattr(reference_data.features, f"{cls.split}_dataloader")

        prediction = results.predict.predict
        if isinstance(prediction, Tensor):
            prediction = prediction.cpu().detach().numpy().flatten()

        dataset = loader.dataset
        if hasattr(dataset, "targets"):
            true_target = dataset.targets
        else:
            iter_object = iter(dataset)
            true_target = np.array([batch[1] for batch in iter_object])

        return cls.metric(target=true_target, predict=prediction)

    @staticmethod
    def _get_least_frequent_val(array: np.ndarray):
        """Return the least frequent value in array (for binary F1)."""
        unique_vals, count = np.unique(np.ravel(array), return_counts=True)
        return unique_vals[np.argmin(count)]


# ---------------------------------------------------------------------------
# Regression and time series metrics
# ---------------------------------------------------------------------------

class RMSE(QualityMetric):
    """Root Mean Squared Error (lower is better)."""
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(mean_squared_error(y_true=target, y_pred=predict, squared=False))


class MSE(QualityMetric):
    """Mean Squared Error (lower is better)."""
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(mean_squared_error(y_true=target, y_pred=predict, squared=True))


class MSLE(QualityMetric):
    """Mean Squared Logarithmic Error (non-negative inputs required)."""
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(mean_squared_log_error(y_true=target, y_pred=predict))


class MAE(QualityMetric):
    """Mean Absolute Error (lower is better)."""
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(mean_absolute_error(y_true=target, y_pred=predict))


class MAPE(QualityMetric):
    """Mean Absolute Percentage Error, expressed as a fraction."""
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(mean_absolute_percentage_error(y_true=target, y_pred=predict))


class SMAPE(QualityMetric):
    """Symmetric MAPE in percentage, robust to scale differences."""
    @classmethod
    def metric(cls, target, predict) -> float:
        t = np.asarray(target).ravel()
        p = np.asarray(predict).ravel()
        return float(np.mean(2.0 * np.abs(t - p) / (np.abs(t) + np.abs(p) + 1e-12)) * 100.0)


class R2(QualityMetric):
    """Coefficient of determination R² (higher is better)."""
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(r2_score(y_true=target, y_pred=predict))


def smape(target, predict) -> float:
    """SMAPE as a functional alias (percent)."""
    return SMAPE.metric(target, predict)


def mape(target, predict) -> float:
    """MAPE as a functional alias (fraction)."""
    return MAPE.metric(target, predict)


def mase(target, predict, seasonal_period: int = 1) -> float:
    """
    Mean Absolute Scaled Error using naive seasonal lag as scale.

    Args:
        target: Ground truth series.
        predict: Predicted series.
        seasonal_period: Lag for naive forecast (1, 7, 12, etc.).
    """
    y = np.asarray(target).ravel()
    p = np.asarray(predict).ravel()
    if y.size <= seasonal_period:
        return float("inf")
    mae_model = np.mean(np.abs(y - p))
    mae_naive = np.mean(np.abs(y[seasonal_period:] - y[:-seasonal_period]))
    return float(mae_model / (mae_naive if mae_naive != 0 else np.finfo(float).eps))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class Accuracy(QualityMetric):
    """Classification accuracy on discrete labels."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        return float(accuracy_score(target, predict))


class Precision(QualityMetric):
    """Macro-averaged precision for classification."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        return float(precision_score(target, predict, average="macro", zero_division=0))


class F1(QualityMetric):
    """F1 score; macro for multiclass, minority-positive for binary."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        uniq_t = np.unique(target)
        if uniq_t.size > 2:
            return float(f1_score(target, predict, average="macro", zero_division=0))
        pos_label = QualityMetric._get_least_frequent_val(target)
        return float(
            f1_score(target, predict, average="binary", pos_label=pos_label, zero_division=0)
        )


class Logloss(QualityMetric):
    """Cross-entropy loss (log loss) on predicted probabilities."""
    output_mode = "probs"

    @classmethod
    def metric(cls, target, predict) -> float:
        return float(log_loss(y_true=target, y_pred=predict))


class ROCAUC(QualityMetric):
    """ROC-AUC score; macro OVR for multiclass, standard binary otherwise."""
    output_mode = "probs"

    @classmethod
    def metric(cls, target, predict) -> float:
        t = np.asarray(target)
        p = np.asarray(predict)
        if np.unique(t).size > 2:
            score = roc_auc_score(y_true=t, y_score=p, multi_class="ovr", average="macro")
        else:
            y_score = p[:, 1] if p.ndim == 2 else p
            score = roc_auc_score(y_true=t, y_score=y_score)
        return float(score)