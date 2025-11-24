import importlib
import torch
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)
from fedcore.metrics.metric_impl import SMAPE, MASE, MAPE
from fedcore.metrics.metric_impl import QualityMetric

from fedcore.metrics.metric_impl import (
    Accuracy,
    F1,
    Precision,
    Logloss,
    ROCAUC,
)
from typing import Optional, Any, Dict, Sequence


_EVALUATE = None

def _get_evaluate():
    """Lazy load the `evaluate` package."""
    global _EVALUATE
    if _EVALUATE is None:
        try:
            _EVALUATE = importlib.import_module("evaluate")
        except ImportError as e:
            raise ImportError("The 'evaluate' package is required for NLP metrics. Install it using `pip install evaluate`.") from e
    return _EVALUATE

_get_evaluate()

# =============================== NLP Metrics ================================

class EvaluateMetric(QualityMetric):
    """Base class for NLP metrics powered by HuggingFace `evaluate`, compatible with QualityMetric."""
    default_value: float = 0.0
    need_to_minimize: bool = False
    split: str = "val"
    output_mode: str = "texts"  # default for text generation

    metric_name: str = ""
    result_key: Optional[str] = None
    _metric = _EVALUATE.load(metric_name)

    @classmethod
    def metric(cls, target: Sequence[Any] | None, predict: Sequence[Any] | None, **kwargs: Any) -> float:
        references = kwargs.pop("references", None)
        predictions = kwargs.pop("predictions", None)
        if references is None:
            references = target
        if predictions is None:
            predictions = predict
        if references is None or predictions is None:
            raise ValueError("Both references and predictions are required.")

        res = cls._metric.compute(references=references, predictions=predictions, **kwargs)

        # Normalize evaluate output to float
        if isinstance(res, dict):
            if cls.result_key is not None:
                val = res[cls.result_key]
            elif "score" in res:
                val = res["score"]
            elif len(res) == 1:
                val = next(iter(res.values()))
            else:
                raise ValueError(
                    f"{cls.__name__}: specify result_key (multiple fields found: {list(res.keys())})"
                )
        else:
            val = res
        return float(val)

    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        out = pipeline.predict(reference_data, output_mode=cls.output_mode)
        preds = out.predict.predict
        preds = preds.detach().cpu().tolist()

        loader = getattr(reference_data.features, f"{cls.split}_dataloader")
        ds = loader.dataset

        if hasattr(ds, "references"):
            refs = ds.references
        elif hasattr(ds, "targets"):
            refs = ds.targets
        elif hasattr(ds, "labels"):
            refs = ds.labels
        else:
            it = iter(ds)
            refs = [ex[1] for ex in it]

        return cls.metric(refs, preds)

    def compute(self, y_true: Sequence[Any] | None = None, y_pred: Sequence[Any] | None = None, *, references: Sequence[Any] | None = None, predictions: Sequence[Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
        if references is None and y_true is not None:
            references = y_true
        if predictions is None and y_pred is not None:
            predictions = y_pred
        if references is None or predictions is None:
            raise ValueError("Both references and predictions are required.")
        return self._metric.compute(predictions=predictions, references=references, **kwargs)


class NLPAccuracy(EvaluateMetric):
    """Classification accuracy for NLP tasks."""
    metric_name = "accuracy"
    result_key = "accuracy"
    output_mode = "labels"
    _metric = _EVALUATE.load(metric_name)
    
    
# =============================== Additional NLP Metrics ================================

class NLPPrecision(EvaluateMetric):
    """Precision for NLP classification tasks."""
    metric_name = "precision"
    result_key = "precision"
    output_mode = "labels"
    _metric = _EVALUATE.load(metric_name)

class NLPRecall(EvaluateMetric):
    """Recall for NLP classification tasks."""
    metric_name = "recall"
    result_key = "recall"
    output_mode = "labels"
    _metric = _EVALUATE.load(metric_name)

class NLPF1(EvaluateMetric):
    """F1 score for NLP classification tasks."""
    metric_name = "f1"
    result_key = "f1"
    output_mode = "labels"
    _metric = _EVALUATE.load(metric_name)
