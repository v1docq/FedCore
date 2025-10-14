import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import List, Dict, Union

# Import necessary libraries
from fedot.core.composer.metrics import Metric

# ============================== Pareto =====================================


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

REGRESSION_METRICS = [
    'R2',
    'MAE',
    'MAPE',
    'MSLE',
    'RMSE',
    'SMAPE',
    'MSE', 
    'MASE'
]

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
    
CLASSIFICATION_METRICS = [
    'ROCAUC',
    'LogLoss',
    'F1',
    'Precision',
    'Accuracy'
]

__all__ = [
    *CLASSIFICATION_METRICS, 
    *REGRESSION_METRICS,
    'ParetoMetrics'
]