import importlib
import torch
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)
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
    _metric = None  # Will be loaded lazily
    
    @classmethod
    def _load_metric(cls):
        """Lazy load the metric."""
        if cls._metric is None and cls.metric_name:
            cls._metric = _EVALUATE.load(cls.metric_name)
        return cls._metric

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

        metric = cls._load_metric()
        if metric is None:
            raise ValueError(f"Metric {cls.metric_name} not loaded. Ensure metric_name is set correctly.")
        
        res = metric.compute(references=references, predictions=predictions, **kwargs)

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
    def get_value(cls, pipeline, reference_data, validation_blocks=None, **generation_kwargs) -> float:
        out = pipeline.predict(reference_data, output_mode=cls.output_mode, **generation_kwargs)
        preds = out.predict.predict
        
        # Fix: if predict is already a list of strings (for output_mode="texts")
        if isinstance(preds, list):
            # Check if first element is a string
            if len(preds) > 0 and isinstance(preds[0], str):
                # This is already texts, no need to decode
                predictions = preds
            else:
                # Try to process as tensors
                if len(preds) > 0 and hasattr(preds[0], 'detach'):
                    preds = [p.detach().cpu().tolist() if isinstance(p, torch.Tensor) else p for p in preds]
                predictions = preds
        elif isinstance(preds, torch.Tensor):
            predictions = preds.detach().cpu().tolist()
        else:
            predictions = preds

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

        return cls.metric(refs, predictions)

    def compute(self, y_true: Sequence[Any] | None = None, y_pred: Sequence[Any] | None = None, *, references: Sequence[Any] | None = None, predictions: Sequence[Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
        if references is None and y_true is not None:
            references = y_true
        if predictions is None and y_pred is not None:
            predictions = y_pred
        if references is None or predictions is None:
            raise ValueError("Both references and predictions are required.")
        
        metric = self._load_metric()
        if metric is None:
            raise ValueError(f"Metric {self.metric_name} not loaded. Ensure metric_name is set correctly.")
        
        return metric.compute(predictions=predictions, references=references, **kwargs)


class NLPAccuracy(EvaluateMetric):
    """Classification accuracy for NLP tasks."""
    metric_name = "accuracy"
    result_key = "accuracy"
    output_mode = "labels"
    
    
# =============================== Additional NLP Metrics ================================

class NLPPrecision(EvaluateMetric):
    """Precision for NLP classification tasks."""
    metric_name = "precision"
    result_key = "precision"
    output_mode = "labels"

class NLPRecall(EvaluateMetric):
    """Recall for NLP classification tasks."""
    metric_name = "recall"
    result_key = "recall"
    output_mode = "labels"

class NLPF1(EvaluateMetric):
    """F1 score for NLP classification tasks."""
    metric_name = "f1"
    result_key = "f1"
    output_mode = "labels"
