from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.functional import softmax
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from typing import Union
from torch import Tensor
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
from fedot.core.composer.metrics import Metric

from fedcore.architecture.computational.devices import default_device


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


class QualityMetric(Metric):
    default_value = 0
    need_to_minimize = False
    output_mode = 'compress'

    @classmethod
    @abstractmethod
    def metric(cls, target, predict) -> float:
        pass

    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        """ Get metric value based on pipeline, reference data, and number of validation blocks.
        Args:
            pipeline: a :class:`Pipeline` instance for evaluation.
            reference_data: :class:`InputData` for evaluation.
            validation_blocks: number of validation blocks. Used only for time series forecasting.
                If ``None``, data separation is not performed.
        """
        metric = cls.default_value
        results = pipeline.predict(reference_data, output_mode=cls.output_mode)
        # get true targets from test/calib dataloader
        test_dataset = reference_data.val_dataloader.dataset
        # get predction from result.predict (CompressionOutputData)
        prediction = results.predict
        print(f"DEBUG: results type: {type(results)}, results.predict type: {type(prediction)}")
        if prediction.__class__.__name__ == "PredictionOutput":
            prediction = prediction.predictions.max(axis=2).flatten()
        if isinstance(prediction, Tensor):
            prediction = prediction.cpu().detach().numpy().flatten()
        if hasattr(test_dataset, 'targets'):
            true_target = reference_data.features.val_dataloader.dataset.targets
        else:
            iter_object = iter(test_dataset)
            true_target = np.array([batch[1] for batch in iter_object]).flatten()
        
        # print(f"DEBUG: true_target shape: {true_target.shape if hasattr(true_target, 'shape') else len(true_target) if hasattr(true_target, '__len__') else 'no shape'}, true_target size: {len(true_target) if hasattr(true_target, '__len__') else 'no len'}")
        # print(f"DEBUG: true_target type: {type(true_target)}, true_target content sample: {true_target[:5] if hasattr(true_target, '__getitem__') and len(true_target) > 0 else true_target}")
        true_target = true_target.astype(np.float32)
        prediction = prediction.astype(np.float32)
        return cls.metric(cls, target=true_target, predict=prediction)

    @staticmethod
    def _get_least_frequent_val(array: np.ndarray):
        """Returns the least frequent value in a flattened numpy array."""
        unique_vals, count = np.unique(np.ravel(array), return_counts=True)
        least_frequent_idx = np.argmin(count)
        least_frequent_val = unique_vals[least_frequent_idx]
        return least_frequent_val


class RMSE(QualityMetric):
    def metric(cls, target, predict) -> float:
        return mean_squared_error(y_true=target, y_pred=predict, squared=False)


class SMAPE(QualityMetric):

    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        """ Get metric value based on pipeline, reference data, and number of validation blocks.
        Args:
            pipeline: a :class:`Pipeline` instance for evaluation.
            reference_data: :class:`InputData` for evaluation.
            validation_blocks: number of validation blocks. Used only for time series forecasting.
                If ``None``, data separation is not performed.
        """
        true_pred = pipeline.predict(reference_data, output_mode=cls.output_mode).predict
        # get true targets from test dataloader
        target_list = []
        for batch in reference_data.features.test_dataloader:
            x_hist, x_fut, y = [b.to(default_device()) for b in batch]
            target_list.append(y.cpu().detach().numpy().squeeze())
        true_target = np.concatenate(target_list).ravel()
        # get predction from result.predict (OutputData)
        return cls.metric(cls, target=true_target, predict=true_pred)

    def metric(cls, target, predict) -> float:
        return (1 / len(predict) * np.sum(2 * np.abs(target - predict)
                                          / (np.abs(predict) + np.abs(target)) * 100))


class MSE(QualityMetric):
    def metric(cls, target, predict) -> float:
        return mean_squared_error(
            y_true=target, y_pred=predict, squared=True
        )


class MSLE(QualityMetric):
    def metric(cls, target, predict) -> float:
        return mean_squared_log_error(y_true=target, y_pred=predict)


class MAPE(QualityMetric):
    def metric(cls, target, predict) -> float:
        return mean_absolute_percentage_error(
            y_true=target, y_pred=predict
        )


class F1(QualityMetric):
    output_mode = "labels"

    def metric(cls, target, predict) -> float:
        n_classes = len(np.unique(target))
        n_classes_pred = len(np.unique(predict))

        try:
            if n_classes > 2 or n_classes_pred > 2:
                return f1_score(
                    y_true=target, y_pred=predict, average="weighted"
                )
            else:
                pos_label = QualityMetric._get_least_frequent_val(target)
                return f1_score(
                    y_true=target,
                    y_pred=predict,
                    average="binary",
                    pos_label=pos_label,
                )
        except ValueError:
            return cls.default_value


class MAE(QualityMetric):
    def metric(cls, target, predict) -> float:
        return mean_absolute_error(y_true=target, y_pred=predict)


class R2(QualityMetric):
    def metric(self) -> float:
        return r2_score(y_true=self.target, y_pred=self.predicted_labels)


def maximised_r2(graph: Graph, reference_data: InputData, **kwargs):
    result = graph.predict(reference_data)
    r2_value = r2_score(y_true=reference_data.target, y_pred=result.predict)
    return 1 - r2_value


class ROCAUC(QualityMetric):
    def metric(cls, target, predict) -> float:
        n_classes = len(np.unique(target))

        if n_classes > 2:
            target = pd.get_dummies(target)
            additional_params = {"multi_class": "ovr", "average": "macro"}
            prediction = pd.get_dummies(predict)
        else:
            target = target
            additional_params = {}
            prediction = predict

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

    def metric(cls, target, predict) -> float:
        n_classes = np.unique(target)
        if n_classes.shape[0] >= 2:
            additional_params = {"average": "macro"}
        else:
            additional_params = {}

        score = precision_score(y_pred=predict, y_true=target, **additional_params)
        score = round(score, 3)
        return score


class Logloss(QualityMetric):
    def metric(cls, target, predict) -> float:
        return log_loss(y_true=target, y_pred=predict)


class Accuracy(QualityMetric):
    output_mode = "labels"

    def metric(cls, target, predict) -> float:
        return accuracy_score(y_true=target, y_pred=predict)


def mase(A, F, y_train):
    return mean_absolute_scaled_error(A, F, y_train=y_train)


def smape(a, f, _=None):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


def mape(A, F):
    return mean_absolute_percentage_error(A, F)
