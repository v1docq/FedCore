import torch
from torch import Tensor
from torch.nn.functional import softmax
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from abc import ABC, abstractmethod
from typing import List, Dict, Union

# Import necessary libraries
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


class ParetoMetrics:
    def pareto_metric_list(self, costs: Union[list, torch.Tensor], maximise: bool = True) -> torch.Tensor:
        """Return mask of Pareto-efficient points."""
        costs = torch.tensor(costs)
        is_efficient = torch.ones(costs.shape[0], dtype=torch.bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                if maximise:
                    is_efficient[is_efficient] = torch.all(costs[is_efficient] >= c, dim=1)
                else:
                    is_efficient[is_efficient] = torch.all(costs[is_efficient] <= c, dim=1)
        return is_efficient


# ============================ Generic metrics =================================

class QualityMetric(Metric):
    """Base metric computed via pipeline.predict()."""
    default_value = 0
    need_to_minimize = False
    output_mode = "compress"  # 'labels' | 'probs' | 'raw' | 'compress'
    split = "val"             # 'val' | 'test'

    @classmethod
    def metric(cls, target, predict) -> float:
        raise NotImplementedError

    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        """Compute metric on features.<split> using pipeline.predict(output_mode)."""
        results = pipeline.predict(reference_data, output_mode=cls.output_mode)
        loader = getattr(reference_data.features, f"{cls.split}_dataloader")

        prediction = results.predict.predict
        if isinstance(prediction, Tensor):
            prediction = prediction.cpu().detach()

        dataset = loader.dataset
        if hasattr(dataset, "targets"):
            true_target = dataset.targets
        else:
            iter_object = iter(dataset)
            true_target = torch.tensor([batch[1] for batch in iter_object])

        return cls.metric(target=true_target, predict=prediction)

    @staticmethod
    def _get_least_frequent_val(array: torch.Tensor):
        """Return least frequent value in array."""
        unique_vals, count = torch.unique(array, return_counts=True)
        return unique_vals[torch.argmin(count)]


# --------------------------- Regression / TS ----------------------------------

class RMSE(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean((target - predict) ** 2).sqrt())


class SMAPE(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        t = target.ravel()
        p = predict.ravel()
        return float(torch.mean(2.0 * torch.abs(t - p) / (torch.abs(t) + torch.abs(p) + 1e-12)) * 100.0)


class MSE(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean((target - predict) ** 2))


class MSLE(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean((torch.log1p(target) - torch.log1p(predict)) ** 2))


class MAPE(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean(torch.abs((target - predict) / target)))


class MAE(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean(torch.abs(target - predict)))


class R2(QualityMetric):
    @classmethod
    def metric(cls, target, predict) -> float:
        return float(1 - torch.sum((target - predict) ** 2) / torch.sum((target - target.mean()) ** 2))


# --------------------------- Classification -----------------------------------

class Accuracy(QualityMetric):
    """Accuracy on label predictions."""  
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean(target == predict).item())


class Precision(QualityMetric):
    """Macro precision on labels."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        tp = torch.sum((target == 1) & (predict == 1))
        fp = torch.sum((target == 0) & (predict == 1))
        return float(tp / (tp + fp + 1e-8))


class F1(QualityMetric):
    """F1; macro for multiclass, binary uses minority class as positive."""
    output_mode = "labels"

    @classmethod
    def metric(cls, target, predict) -> float:
        tp = torch.sum((target == 1) & (predict == 1))
        fp = torch.sum((target == 0) & (predict == 1))
        fn = torch.sum((target == 1) & (predict == 0))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * (precision * recall) / (precision + recall + 1e-8)


class Logloss(QualityMetric):
    """Log loss on probabilities."""
    output_mode = "probs"

    @classmethod
    def metric(cls, target, predict) -> float:
        return float(torch.mean(-target * torch.log(predict) - (1 - target) * torch.log(1 - predict)))


class ROCAUC(QualityMetric):
    """ROC-AUC; multiclass uses macro OVR."""
    output_mode = "probs"

    @classmethod
    def metric(cls, target, predict) -> float:
        t = target
        p = predict
        if torch.unique(t).size(0) > 2:
            score = torchmetrics.functional.roc_auc_score(t, p)
        else:
            score = torchmetrics.functional.roc_auc_score(t, p[:, 1] if p.ndimension() == 2 else p)
        return round(score, 3)
    
class MASE(QualityMetric):
    """Mean Absolute Scaled Error (MASE)."""
    
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor, seasonal_factor: int = 1) -> float:
        """
        Compute Mean Absolute Scaled Error (MASE).

        Args:
            target (torch.Tensor): Ground truth values.
            predict (torch.Tensor): Predicted values.
            seasonal_factor (int): Seasonal factor (e.g., number of periods per year).

        Returns:
            float: The MASE value.
        """
        # Compute the scale based on the seasonal difference (for example, yearly data)
        scale = torch.mean(torch.abs(target[seasonal_factor:] - target[:-seasonal_factor]))

        # Calculate the MASE
        mase_value = torch.mean(torch.abs(target - predict)) / scale
        return mase_value.item()
