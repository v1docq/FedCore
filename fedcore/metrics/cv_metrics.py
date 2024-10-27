from abc import abstractmethod
from typing import Optional

from fedot.core.composer.metrics import Metric
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.custom_errors import AbstractMethodNotImplementError
from torch import nn

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.tools.ruler import PerformanceEvaluator


class CompressionMetric(Metric):
    default_value = 0
    need_to_minimize = False

    @staticmethod
    @abstractmethod
    def metric(**kwargs) -> float:
        raise AbstractMethodNotImplementError

    def simple_prediction(self, pipeline, reference_data):
        predict = pipeline.predict(reference_data)
        return predict, reference_data

    @classmethod
    def get_value(
        cls,
        pipeline: Pipeline,
        reference_data: InputData,
        validation_blocks: Optional[int] = None,
    ) -> float:
        """Get metric value based on pipeline, reference data, and number of validation blocks.
        Args:
            pipeline: a :class:`Pipeline` instance for evaluation.
            reference_data: :class:`InputData` for evaluation.
            validation_blocks: number of validation blocks. Used only for time series forecasting.
                If ``None``, data separation is not performed.
        """
        metric = cls.default_value
        metric = cls.metric(pipeline, reference_data.features.calib_dataloader)
        if cls.need_to_minimize:
            metric = -metric
        return metric


class IntermediateAttention(CompressionMetric):
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
    @classmethod
    def metric(cls, student_feats, teacher_feats, weights):
        loss = 0
        for i in range(len(student_feats)):
            loss += weights[i] * nn.MSELoss()(student_feats[i], teacher_feats[i])
        return loss


class LastLayer(CompressionMetric):
    @classmethod
    def metric(cls, student_logits, teacher_logits, weight):
        return weight * nn.MSELoss()(student_logits, teacher_logits)


class Throughput(CompressionMetric):
    need_to_minimize = True

    @classmethod
    def metric(cls, model, dataset, device=default_device(), batch_size=32):
        evaluator = PerformanceEvaluator(model, dataset, device, batch_size)
        return evaluator.throughput_eval()


class Latency(CompressionMetric):
    need_to_minimize = False

    @classmethod
    def metric(cls, model, dataset, device=default_device(), batch_size=32):
        evaluator = PerformanceEvaluator(model, dataset, device, batch_size)
        return evaluator.latency_eval()


class CV_quality_metric(CompressionMetric):
    default_clf_metric = "accuracy"
    need_to_minimize = True

    def __repr__(self):
        """Fedcore_compression_quality_metric"""

    @classmethod
    def metric(cls, model, dataset, device=default_device(), batch_size=32):
        evaluator = PerformanceEvaluator(model, dataset, device, batch_size)
        metric_dict = evaluator.eval()
        return metric_dict
