import torch
from torch import Tensor
from torch.nn.functional import softmax
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from abc import ABC, abstractmethod
from typing import List, Dict, Union

from fedcore.metrics.quality import QualityMetric

from fedot.core.composer.metrics import Metric
from fedot.core.data.data import InputData
from golem.core.dag.graph import Graph

# ============================== Counters =====================================

class MetricCounter(ABC):
    """Base class for streaming metric accumulation."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update internal state with a new batch."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Return computed metrics."""
        raise NotImplementedError


class ClassificationMetricCounter(MetricCounter):
    """Accumulates logits/targets and computes macro metrics (+ ROC-AUC)."""

    def __init__(self, class_metrics: bool = False) -> None:
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_score: List[Tensor] = []
        self.class_metrics = class_metrics

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Add a batch of logits and targets."""
        logits = logits.detach().cpu()
        targets = targets.detach().cpu()
        self.y_true.extend(targets.tolist())
        self.y_pred.extend(torch.argmax(logits, dim=-1).tolist())
        self.y_score.extend(softmax(logits, dim=-1))

    def compute(self) -> Dict[str, float]:
        """Compute macro metrics; add ROC-AUC if y_score is available."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average="macro"
        )

        scores = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        # Compute ROC-AUC
        try:
            y_true = torch.tensor(self.y_true)
            y_score = torch.stack(self.y_score)
            if y_score.ndimension() == 2 and y_score.size(1) > 1:
                auc = torchmetrics.functional.roc_auc_score(
                    y_true, y_score
                )
            else:
                pos = y_score[:, 1] if y_score.ndimension() == 2 else y_score
                auc = torchmetrics.functional.roc_auc_score(y_true, pos)
            scores["roc_auc"] = round(float(auc), 3)
        except Exception:
            pass

        if self.class_metrics:
            f1s = f1_score(self.y_true, self.y_pred, average=None)
            scores.update({f"f1_for_class_{i}": float(s) for i, s in enumerate(f1s)})

        return scores


class SegmentationMetricCounter(MetricCounter):
    """IoU/Dice for semantic segmentation."""

    def __init__(self, class_metrics: bool = False) -> None:
        self.iou: List[Tensor] = []
        self.dice: List[Tensor] = []
        self.class_metrics = class_metrics

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Accumulate predictions and masks."""
        masks = torch.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            masks[:, i, :, :] = (targets == i).float()
        self.iou.append(iou_score(predictions, masks))
        self.dice.append(dice_score(predictions, masks))

    def compute(self) -> Dict[str, float]:
        """Return mean IoU/Dice (ignoring empty masks)."""
        iou = torch.cat(self.iou).T
        dice = torch.cat(self.dice).T

        scores = {
            "iou": iou[1:][iou[1:] >= 0].mean().item(),
            "dice": dice[1:][dice[1:] >= 0].mean().item(),
        }
        if self.class_metrics:
            scores.update({f"iou_for_class_{i}": s[s >= 0].mean().item() for i, s in enumerate(iou)})
            scores.update({f"dice_for_class_{i}": s[s >= 0].mean().item() for i, s in enumerate(dice)})

        return scores


class ObjectDetectionMetricCounter(MetricCounter):
    """mAP/mAR for object detection."""

    def __init__(self, class_metrics: bool = False) -> None:
        self.map = MeanAveragePrecision(class_metrics=class_metrics)
        self.class_metrics = class_metrics

    def update(
        self,
        predictions: List[Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]],
    ) -> None:
        self.map.update(preds=predictions, target=targets)

    def compute(self) -> Dict[str, float]:
        """Return mAP/mAR (optionally per-class)."""
        scores = self.map.compute()
        if self.class_metrics:
            scores.update({f"map_for_class_{i}": s for i, s in enumerate(scores["map_per_class"])})

            scores.update({f"mar_100_for_class_{i}": s for i, s in enumerate(scores["mar_100_per_class"])})
        scores.pop("map_per_class", None)
        scores.pop("mar_100_per_class", None)
        return scores


class LossesAverager(MetricCounter):
    """Average any dict of losses."""

    def __init__(self) -> None:
        self.losses: Dict[str, float] = {}
        self.counter = 0

    def update(self, losses: Dict[str, Tensor]) -> None:
        self.counter += 1
        if not self.losses:
            self.losses = {k: v.item() for k, v in losses.items()}
        else:
            for key, value in losses.items():
                self.losses[key] += float(value.item())

    def compute(self) -> Dict[str, float]:
        return {k: v / max(self.counter, 1) for k, v in self.losses.items()}


# ============================= Helpers =======================================

def iou_score(outputs: Tensor, masks: Tensor, threshold: float = 0.5, smooth: float = 1e-10) -> Tensor:
    """Batch IoU for one-hot masks."""
    outputs = (outputs > threshold).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    union = torch.logical_or(outputs, masks).float().sum((2, 3))
    iou = (intersection + smooth) / (union + smooth)
    iou[union == 0] = -1
    return iou


def dice_score(outputs: Tensor, masks: Tensor, threshold: float = 0.5, smooth: float = 1e-10) -> Tensor:
    """Batch Dice for one-hot masks."""
    outputs = (outputs > threshold).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    total = (outputs + masks).sum((2, 3))
    dice = (2 * intersection + smooth) / (total + smooth)
    dice[total == 0] = -1
    return dice


from fedcore.metrics.pareto import ParetoMetrics

class MASE(QualityMetric):
    """MAE scaled by a fixed training-series seasonal difference along axis 0.

    Pass training_target or a precomputed training_scale on every call. A zero
    scale yields 0 for perfect predictions and +inf otherwise.
    """
    need_to_minimize = True
    @staticmethod
    def training_scale(training_target, seasonal_factor=1):
        values = torch.as_tensor(training_target).double()
        if seasonal_factor < 1 or len(values) <= seasonal_factor:
            raise ValueError('Training series must exceed positive seasonal_factor')
        return (values[seasonal_factor:] - values[:-seasonal_factor]).abs().mean()

    @classmethod
    def metric(cls, target, predict, seasonal_factor=1, training_target=None, training_scale=None):
        if training_scale is None:
            if training_target is None:
                raise ValueError('MASE requires an explicit training_target or training_scale')
            training_scale = cls.training_scale(training_target, seasonal_factor)
        scale = float(training_scale)
        if scale < 0 or not torch.isfinite(torch.tensor(scale)):
            raise ValueError('Training scale must be finite and nonnegative')
        error = float((target.double() - predict.double()).abs().mean())
        return error / scale if scale > 0 else (0.0 if error == 0 else float('inf'))


class SMAPE(QualityMetric):
    need_to_minimize = True
    """Symmetric Mean Absolute Percentage Error (SMAPE)."""

    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        """
        Compute SMAPE (Symmetric Mean Absolute Percentage Error).

        Args:
            target (torch.Tensor): Ground truth values.
            predict (torch.Tensor): Predicted values.

        Returns:
            float: The SMAPE value.
        """
        t = target.ravel()
        p = predict.ravel()
        return float(torch.mean(2.0 * torch.abs(t - p) / (torch.abs(t) + torch.abs(p) + 1e-12)) * 100.0)


class MSE(QualityMetric):
    need_to_minimize = True
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        return float(torch.mean((target - predict) ** 2))


class MSLE(QualityMetric):
    need_to_minimize = True
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        return float(torch.mean((torch.log1p(target) - torch.log1p(predict)) ** 2))


class MAPE(QualityMetric):
    need_to_minimize = True
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        error = (target.double() - predict.double()).abs()
        if (target == 0).any():
            if (error[target == 0] > 0).any():
                return float('inf')
            ratio = torch.where(target != 0, error / target.double().abs().clamp_min(torch.finfo(torch.double).tiny), 0)
            return float(ratio.mean())
        return float((error / target.double().abs()).mean())


class MAE(QualityMetric):
    need_to_minimize = True
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        return float(torch.mean(torch.abs(target - predict)))


class R2(QualityMetric):
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        target, predict = target.double(), predict.double()
        denominator = (target - target.mean()).square().sum()
        error = (target - predict).square().sum()
        if denominator == 0:
            return 1.0 if error == 0 else 0.0
        return float(1 - error / denominator)


# --------------------------- Classification -----------------------------------

class Accuracy(QualityMetric):
    """Accuracy on label predictions."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        return float((target == predict).to(torch.float64).mean())


class Precision(QualityMetric):
    """Macro precision on labels."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        values = []
        for label in torch.unique(torch.cat((target.reshape(-1), predict.reshape(-1)))):
            tp = ((target == label) & (predict == label)).sum()
            predicted = (predict == label).sum()
            values.append(tp.double() / predicted if predicted else tp.double() * 0)
        return float(torch.stack(values).mean())


class F1(QualityMetric):
    """Macro F1 over the union of observed and predicted class labels."""
    output_mode = "labels"
    @classmethod
    def metric(cls, target, predict):
        values = []
        for label in torch.unique(torch.cat((target.reshape(-1), predict.reshape(-1)))):
            tp = ((target == label) & (predict == label)).sum()
            denominator = (target == label).sum() + (predict == label).sum()
            values.append(2 * tp.double() / denominator if denominator else tp.double() * 0)
        return float(torch.stack(values).mean())


class Logloss(QualityMetric):
    """Log loss on probabilities."""
    output_mode = "probs"

    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        return float(torch.mean(-target * torch.log(predict) - (1 - target) * torch.log(1 - predict)))


class ROCAUC(QualityMetric):
    """ROC-AUC; multiclass uses macro OVR."""
    output_mode = "probs"

    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        t = target
        p = predict
        if torch.unique(t).size(0) > 2:
            score = torchmetrics.functional.roc_auc_score(t, p)
        else:
            score = torchmetrics.functional.roc_auc_score(t, p[:, 1] if p.ndimension() == 2 else p)
        return round(score, 3)
