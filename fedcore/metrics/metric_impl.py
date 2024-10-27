from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.functional import softmax
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from typing import Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from golem.core.dag.graph import Graph
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
)
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error


class MetricCounter(ABC):
    """Generalized class for calculating metrics"""

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Have to implement updating, taking model outputs as input."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Have to implement metrics computing."""
        raise NotImplementedError


class ClassificationMetricCounter(MetricCounter):
    """Calculates metrics for classification task.

    Args:
        class_metrics:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates predictions and targets."""
        self.y_true.extend(targets.tolist())
        self.y_score.extend(softmax(predictions, dim=1).tolist())
        self.y_pred.extend(predictions.argmax(1).tolist())

    def compute(self) -> Dict[str, float]:
        """Compute accuracy, precision, recall, f1, roc auc metrics.

        Returns:
             Dictionary: `{metric: score}`.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average="macro"
        )

        scores = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if self.class_metrics:
            f1s = f1_score(self.y_true, self.y_pred, average=None)
            scores.update({f"f1_for_class_{i}": s for i, s in enumerate(f1s)})
        return scores


class SegmentationMetricCounter(MetricCounter):
    """Calculates metrics for semantic segmentation task.

    Args:
        class_metrics:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.iou = []
        self.dice = []
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates iou and dice."""
        masks = torch.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            masks[:, i, :, :] = torch.squeeze(targets == i)
        self.iou.append(iou_score(predictions, masks))
        self.dice.append(dice_score(predictions, masks))

    def compute(self) -> Dict[str, float]:
        """Compute average metrics.

        Returns:
             Dictionary: `{metric: score}`.
        """
        iou = torch.cat(self.iou).T
        dice = torch.cat(self.dice).T

        scores = {
            "iou": iou[1:][iou[1:] >= 0].mean().item(),
            "dice": dice[1:][dice[1:] >= 0].mean().item(),
        }
        if self.class_metrics:
            scores.update(
                {
                    f"iou_for_class_{i}": s[s >= 0].mean().item()
                    for i, s in enumerate(iou)
                }
            )
            scores.update(
                {
                    f"dice_for_class_{i}": s[s >= 0].mean().item()
                    for i, s in enumerate(dice)
                }
            )
        return scores


class ObjectDetectionMetricCounter(MetricCounter):
    """Calculates metrics for object detection task.

    Args:
        class_metrics:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.map = MeanAveragePrecision(class_metrics=class_metrics)
        self.class_metrics = class_metrics

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Accumulates predictions and targets."""
        self.map.update(preds=predictions, target=targets)

    def compute(self) -> Dict[str, float]:
        """Compute MAP, MAR metrics.

        Returns:
             Dictionary: `{metric: score}`.
        """

        scores = self.map.compute()
        if self.class_metrics:
            scores.update(
                {f"map_for_class_{i}": s for i, s in enumerate(scores["map_per_class"])}
            )
            scores.update(
                {
                    f"mar_100_for_class_{i}": s
                    for i, s in enumerate(scores["mar_100_per_class"])
                }
            )
        del scores["map_per_class"]
        del scores["mar_100_per_class"]
        return scores


class LossesAverager(MetricCounter):
    """Calculates the average loss."""

    def __init__(self) -> None:
        super().__init__()
        self.losses = None
        self.counter = 0

    def update(self, losses: Dict[str, torch.Tensor]) -> None:
        """Accumulates losses"""
        self.counter += 1
        if self.losses is None:
            self.losses = {k: v.item() for k, v in losses.items()}
        else:
            for key, value in losses.items():
                self.losses[key] += value.item()

    def compute(self) -> Dict[str, float]:
        """Compute average losses.

        Returns:
            Dictionary: `{metric: score}`.
        """
        return {k: v / self.counter for k, v in self.losses.items()}


def iou_score(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-10,
) -> torch.Tensor:
    """Computes intersection over union (masks) on batch.

    Args:
        outputs: Output from semantic segmentation model.
        masks: True masks.
        threshold: Binarization threshold for output.
        smooth: Additional constant to avoid division by zero.

    Returns:
        Intersection over union for batch.
    """
    outputs = (outputs > threshold).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    union = torch.logical_or(outputs, masks).float().sum((2, 3))
    iou = (intersection + smooth) / (union + smooth)
    iou[union == 0] = -1
    return iou


def dice_score(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-10,
) -> torch.Tensor:
    """Computes dice coefficient (masks) on batch.

    Args:
        outputs: Output from semantic segmentation model.
        masks: True masks.
        threshold: Binarization threshold for output.
        smooth: Additional constant to avoid division by zero.

    Returns:
        Dice for batch.
    """
    outputs = (outputs > threshold).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    total = (outputs + masks).sum((2, 3))
    dice = (2 * intersection + smooth) / (total + smooth)
    dice[total == 0] = -1
    return dice


class ParetoMetrics:
    def pareto_metric_list(
        self, costs: Union[list, np.ndarray], maximise: bool = True
    ) -> np.ndarray:
        """Calculates the pareto front for a list of costs.

        Args:
            costs: list of costs. An (n_points, n_costs) array.
            maximise: flag for maximisation or minimisation.

        Returns:
            A (n_points, ) boolean array, indicating whether each point is Pareto efficient

        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                if maximise:
                    is_efficient[is_efficient] = np.any(
                        costs[is_efficient] >= c, axis=1
                    )  # Remove dominated points
                else:
                    is_efficient[is_efficient] = np.any(
                        costs[is_efficient] <= c, axis=1
                    )  # Remove dominated points
        return is_efficient


class QualityMetric:
    def __init__(
        self,
        target,
        predicted_labels,
        predicted_probs=None,
        metric_list: list = ("f1", "roc_auc", "accuracy", "logloss", "precision"),
        default_value: float = 0.0,
    ):
        self.predicted_probs = predicted_probs
        labels_as_matrix = len(predicted_labels.shape) >= 2
        labels_as_one_dim = min(predicted_labels.shape) == 1
        if labels_as_matrix and not labels_as_one_dim:
            self.predicted_labels = np.argmax(predicted_labels, axis=1)
        else:
            self.predicted_labels = np.array(predicted_labels).flatten()
        self.target = np.array(target).flatten()
        self.metric_list = metric_list
        self.default_value = default_value

    def metric(self) -> float:
        pass

    @staticmethod
    def _get_least_frequent_val(array: np.ndarray):
        """Returns the least frequent value in a flattened numpy array."""
        unique_vals, count = np.unique(np.ravel(array), return_counts=True)
        least_frequent_idx = np.argmin(count)
        least_frequent_val = unique_vals[least_frequent_idx]
        return least_frequent_val


class RMSE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_error(
            y_true=self.target, y_pred=self.predicted_labels, squared=False
        )


class SMAPE(QualityMetric):
    def metric(self):
        return (
            1
            / len(self.predicted_labels)
            * np.sum(
                2
                * np.abs(self.target - self.predicted_labels)
                / (np.abs(self.predicted_labels) + np.abs(self.target))
                * 100
            )
        )


class MSE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_error(
            y_true=self.target, y_pred=self.predicted_labels, squared=True
        )


class MSLE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_log_error(y_true=self.target, y_pred=self.predicted_labels)


class MAPE(QualityMetric):
    def metric(self) -> float:
        return mean_absolute_percentage_error(
            y_true=self.target, y_pred=self.predicted_labels
        )


class F1(QualityMetric):
    output_mode = "labels"

    def metric(self) -> float:
        n_classes = len(np.unique(self.target))
        n_classes_pred = len(np.unique(self.predicted_labels))

        try:
            if n_classes > 2 or n_classes_pred > 2:
                return f1_score(
                    y_true=self.target, y_pred=self.predicted_labels, average="weighted"
                )
            else:
                pos_label = QualityMetric._get_least_frequent_val(self.target)
                return f1_score(
                    y_true=self.target,
                    y_pred=self.predicted_labels,
                    average="binary",
                    pos_label=pos_label,
                )
        except ValueError:
            return self.default_value


class MAE(QualityMetric):
    def metric(self) -> float:
        return mean_absolute_error(y_true=self.target, y_pred=self.predicted_labels)


class R2(QualityMetric):
    def metric(self) -> float:
        return r2_score(y_true=self.target, y_pred=self.predicted_labels)


def maximised_r2(graph: Graph, reference_data: InputData, **kwargs):
    result = graph.predict(reference_data)
    r2_value = r2_score(y_true=reference_data.target, y_pred=result.predict)
    return 1 - r2_value


class ROCAUC(QualityMetric):
    def metric(self) -> float:
        n_classes = len(np.unique(self.target))

        if n_classes > 2:
            target = pd.get_dummies(self.target)
            additional_params = {"multi_class": "ovr", "average": "macro"}
            if self.predicted_probs is None:
                prediction = pd.get_dummies(self.predicted_labels)
            else:
                prediction = self.predicted_probs
        else:
            target = self.target
            additional_params = {}
            prediction = self.predicted_probs

        score = roc_auc_score(
            y_score=prediction,
            y_true=target,
            labels=np.unique(target),
            **additional_params,
        )
        score = round(score, 3)

        return score


class Precision(QualityMetric):
    output_mode = "labels"

    def metric(self) -> float:
        n_classes = np.unique(self.target)
        if n_classes.shape[0] >= 2:
            additional_params = {"average": "macro"}
        else:
            additional_params = {}

        score = precision_score(
            y_pred=self.predicted_labels, y_true=self.target, **additional_params
        )
        score = round(score, 3)
        return score


class Logloss(QualityMetric):
    def metric(self) -> float:
        return log_loss(y_true=self.target, y_pred=self.predicted_probs)


class Accuracy(QualityMetric):
    output_mode = "labels"

    def metric(self) -> float:
        return accuracy_score(y_true=self.target, y_pred=self.predicted_labels)


def mase(A, F, y_train):
    return mean_absolute_scaled_error(A, F, y_train=y_train)


def smape(a, f, _=None):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


def mape(A, F):
    return mean_absolute_percentage_error(A, F)
