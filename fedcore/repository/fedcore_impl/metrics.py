from typing import Dict
from fedot.core.repository.metrics_repository import (
    ClassificationMetricsEnum,
    RegressionMetricsEnum,
    MetricsEnum,
    MetricCallable,
)
from fedot.core.composer.metrics import (
    Accuracy,
    ComplexityMetric,
    F1,
    Logloss,
    MAE,
    MAPE,
    MSE,
    MSLE,
    Precision,
    QualityMetric,
    R2,
    RMSE,
    ROCAUC,
    SMAPE,
)
from typing import Union

from fedcore.metrics.cv_metrics import (
    LastLayer,
    IntermediateFeatures,
    IntermediateAttention,
    Latency,
    Throughput,
    CV_quality_metric,
)
from fedcore.repository.constanst_repository import (
    DistilationMetricsEnum,
    InferenceMetricsEnum,
    CVMetricsEnum,
)


class MetricsRepository:
    _metrics_implementations: Dict[MetricsEnum, MetricCallable] = {
        # classification
        ClassificationMetricsEnum.ROCAUC: ROCAUC.get_value,
        ClassificationMetricsEnum.ROCAUC_penalty: ROCAUC.get_value_with_penalty,
        ClassificationMetricsEnum.f1: F1.get_value,
        ClassificationMetricsEnum.precision: Precision.get_value,
        ClassificationMetricsEnum.accuracy: Accuracy.get_value,
        ClassificationMetricsEnum.logloss: Logloss.get_value,
        # regression
        RegressionMetricsEnum.MAE: MAE.get_value,
        RegressionMetricsEnum.MSE: MSE.get_value,
        RegressionMetricsEnum.MSLE: MSLE.get_value,
        RegressionMetricsEnum.MAPE: MAPE.get_value,
        RegressionMetricsEnum.SMAPE: SMAPE.get_value,
        RegressionMetricsEnum.RMSE: RMSE.get_value,
        RegressionMetricsEnum.RMSE_penalty: RMSE.get_value_with_penalty,
        RegressionMetricsEnum.R2: R2.get_value,
        # Distilation metric
        DistilationMetricsEnum.last_layer: LastLayer.get_value,
        DistilationMetricsEnum.intermediate_layers_attention: IntermediateAttention.get_value,
        DistilationMetricsEnum.intermediate_layers_feature: IntermediateFeatures.get_value,
        # Inference metric
        InferenceMetricsEnum.latency: Latency.get_value,
        InferenceMetricsEnum.throughput: Throughput.get_value,
        # CV metric
        CVMetricsEnum.cv_clf_metric: CV_quality_metric.get_value,
    }

    _metrics_classes = {
        metric_id: getattr(metric_func, "__self__")
        for metric_id, metric_func in _metrics_implementations.items()
    }

    @staticmethod
    def get_metric(metric_name: MetricsEnum) -> MetricCallable:
        return MetricsRepository._metrics_implementations[metric_name]

    @staticmethod
    def get_metric_class(
        metric_name: MetricsEnum,
    ) -> Union[QualityMetric, ComplexityMetric]:
        return MetricsRepository._metrics_classes[metric_name]
