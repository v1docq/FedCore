from __future__ import annotations
from abc import abstractmethod
from typing import Optional
import torch
from torch import nn
from fedot.core.composer.metrics import Metric
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.tools.ruler import PerformanceEvaluator
from fedcore.metrics.metric_impl import SMAPE as smape
# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CompressionMetric(Metric):
    """
    Base class for compression- and computation-oriented metrics.
    """

    default_value: float = 0.0
    need_to_minimize: bool = False

    @staticmethod
    @abstractmethod
    def metric(**kwargs) -> float:
        """Compute the metric from provided arguments."""
        pass

    def simple_prediction(self, pipeline, reference_data):
        """Convenience wrapper to get pipeline predictions and pass-through data."""
        predict = pipeline.predict(reference_data)
        return predict, reference_data

    @classmethod
    def get_value(
        cls,
        pipeline: Pipeline,
        reference_data: InputData,
        validation_blocks: Optional[int] = None,
    ) -> float:
        """
        Compute metric value on a pipeline and reference data.
        """
        value = cls.metric(pipeline, reference_data.features.val_dataloader)
        if cls.need_to_minimize and isinstance(value, (int, float, torch.Tensor)):
            value = -value
        return value


# ---------------------------------------------------------------------------
# Distillation losses (student vs teacher)
# ---------------------------------------------------------------------------

class IntermediateAttention(CompressionMetric):
    """KL-divergence loss between student and teacher attentions."""
    @classmethod
    def metric(
        cls,
        student_attentions,
        teacher_attentions,
        weights,
        student_teacher_attention_mapping,
    ):
        loss = 0
        for i in range(len(student_attentions)):
            loss += weights[i] * nn.KLDivLoss(reduction="batchmean")(
                student_attentions[i],
                teacher_attentions[student_teacher_attention_mapping[i]],
            )
        return loss


class IntermediateFeatures(CompressionMetric):
    """MSE loss between intermediate student and teacher feature maps."""
    @classmethod
    def metric(cls, student_feats, teacher_feats, weights):
        loss = 0
        for i in range(len(student_feats)):
            loss += weights[i] * nn.MSELoss()(student_feats[i], teacher_feats[i])
        return loss


class LastLayer(CompressionMetric):
    """MSE loss between student and teacher logits."""
    @classmethod
    def metric(cls, student_logits, teacher_logits, weight):
        return weight * nn.MSELoss()(student_logits, teacher_logits)


# ---------------------------------------------------------------------------
# Runtime metrics
# ---------------------------------------------------------------------------

class Throughput(CompressionMetric):
    """Average number of samples processed per second (higher is better)."""
    need_to_minimize = True

    @classmethod
    def metric(cls, model, dataset, device=default_device(), batch_size: int = 32):
        evaluator = PerformanceEvaluator(model=model, data=dataset, device=device, batch_size=batch_size)
        throughputs = evaluator.throughput_eval()
        return float(torch.mean(torch.tensor(throughputs)))


class Latency(CompressionMetric):
    """Average per-batch latency in seconds (lower is better)."""
    need_to_minimize = True

    @classmethod
    def metric(
        cls,
        model,
        dataset,
        model_regime: str = "model_after",
        device=default_device(),
        batch_size: int = 32,
    ):
        evaluator = PerformanceEvaluator(
            model=model,
            model_regime=model_regime,
            data=dataset,
            device=device,
            batch_size=batch_size,
        )
        latency_list = evaluator.latency_eval()
        return float(torch.mean(torch.tensor(latency_list)))


class CV_quality_metric(CompressionMetric):
    """
    Aggregate quality metric wrapper using PerformanceEvaluator.
    """
    default_clf_metric = "accuracy"
    need_to_minimize = True

    def __repr__(self):
        return "Fedcore_compression_quality_metric"

    @classmethod
    def metric(
        cls,
        model,
        dataset,
        model_regime: str = "model_after",
        device=default_device(),
        batch_size: int = 32,
    ):
        evaluator = PerformanceEvaluator(
            model=model,
            model_regime=model_regime,
            data=dataset,
            device=device,
            batch_size=batch_size,
        )
        return evaluator.eval()
