import torch
import pandas as pd

from typing import Optional

from fedot.core.composer.metrics import Metric

import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import List, Dict, Union
import torchmetrics 

# Import necessary libraries

from importlib import import_module

from fedcore.repository.constanst_repository import FedotTaskEnum

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
        if isinstance(prediction, torch.Tensor):
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

PROBLEM_MAPPING = {
    'ts_forecasting': 'regression'
}

ATTRIBUTE_MAPPING = {
    'higher_is_better': 'need_to_minimize', 
    'is_differentiable': 'is_differentiable' 
}

LOADED_METRICS = {}

__problems = [PROBLEM_MAPPING.get(problem.name, problem.name) for problem in FedotTaskEnum]
__problem_to_metric = {
    problem: getattr(torchmetrics, problem).__all__ for problem in __problems
}
__METRICS_TO_PROBLEM = {
    metric: problem for problem, metrics in __problem_to_metric.items() for metric in metrics
}

def get_available_metrics(problem): 
    module = import_module(f'torchmetrics.{PROBLEM_MAPPING.get(problem, problem)}')
    return module.__all__


def metric_factory(metric_name, problem=None) -> QualityMetric:
    if problem is None:
        problem = __METRICS_TO_PROBLEM[metric_name]
    if metric_name in LOADED_METRICS:
        return LOADED_METRICS[metric_name]
    
    module = import_module(f'torchmetrics.{PROBLEM_MAPPING.get(problem, problem)}')
    parent_cls = getattr(module, metric_name)
    attributes = {
        child_attr: getattr(parent_cls, parent_attr) for parent_attr, child_attr in ATTRIBUTE_MAPPING.items()
    }
    # special cases
    attributes['need_to_minimize'] = not attributes['need_to_minimize'] 

    @classmethod
    def metric(cls: torchmetrics.Metric, target, predict, **metric_kw) -> torch.Tensor:
        instance = cls(**metric_kw)
        instance.update(predict, target)
        result = instance.compute()
        del instance
        return result
    
    attributes['metric'] = metric
    new_metric = type(
        metric_name, (parent_cls, QualityMetric), attributes
    )
    LOADED_METRICS[metric_name] = new_metric 
    return new_metric


def calculate_metrics(
    metric_names: tuple[str, ...],
    target: torch.Tensor,
    predict: torch.Tensor,
    rounding_order: int = 3,
    ):
    values = {metric_name: metric_factory(metric_name).metric(target, predict).item() for metric_name in metric_names}
    return _to_df(values, rounding_order)

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

# def calculate_regression_metrics(
#     target: torch.Tensor,
#     predict: torch.Tensor,
#     rounding_order: int = 3,
#     metric_names: tuple[str, ...] = ("r2", "rmse", "mae"),
#     is_forecasting: bool = False
# ) -> pd.DataFrame:
#     """
#     Compute metrics for regression or forecasting (time series) tasks.

#     Args:
#         target (torch.Tensor): Ground truth values.
#         predict (torch.Tensor): Predicted values.
#         rounding_order (int): Decimal places for rounding the results.
#         metric_names (tuple): Metrics to compute.
#         is_forecasting (bool): Flag to indicate if it is forecasting or not.

#     Returns:
#         pd.DataFrame: A DataFrame containing the computed metrics.
#     """

#     values = {name: calculate_metrics(target, predict, name) for name in metric_names if name in REGRESSION_METRICS}

#     return _to_df(values, rounding_order)

# # -------------------- Classification Metrics Calculation --------------------

# def calculate_classification_metrics(
#     target: torch.Tensor,
#     labels: torch.Tensor,
#     probs: Optional[torch.Tensor] = None,
#     rounding_order: int = 3,
#     metric_names: tuple[str, ...] = ("f1", "accuracy")
# ) -> pd.DataFrame:
#     """
#     Compute classification metrics such as Accuracy, F1, Logloss, etc.

#     Args:
#         target (torch.Tensor): Ground truth labels.
#         labels (torch.Tensor): Predicted labels.
#         probs (torch.Tensor, optional): Predicted probabilities (needed for logloss and ROC AUC).
#         rounding_order (int): Decimal places for rounding the results.
#         metric_names (tuple): Metrics to compute.

#     Returns:
#         pd.DataFrame: A DataFrame containing the computed metrics.
#     """
#     values = {}
#     for name in metric_names:
#         metric = CLASSIFICATION_METRICS[name]
#         if name in ("logloss", "roc_auc"):
#             if probs is None:
#                 raise ValueError(f"{name} requires `probs` argument.")
#             values[name] = metric(target, probs)
#         else:
#             values[name] = metric(target, labels)

#     return _to_df(values, rounding_order)

# # -------------------- Computational Metrics Calculation --------------------

# def calculate_computational_metrics(model, dataset, model_regime: str) -> float:
#     """
#     Compute computational metrics like latency and throughput.

#     Args:
#         model (torch.nn.Module): The model to evaluate.
#         dataset: The dataset to evaluate on.
#         model_regime (str): The regime of the model.

#     Returns:
#         float: The computed computational metric (e.g., throughput or latency).
#     """    

#     ###################### TODO Here should be called the fedcore.tools.ruler.PerformanceEvaluator
#     pass 
