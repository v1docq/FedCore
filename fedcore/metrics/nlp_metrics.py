"""
NLP metrics integrated with HuggingFace `evaluate`, unified with FedCore QualityMetric API.

- Fully compatible with QualityMetric interface (classmethod metric(...), get_value(...))
- Supports both y_true/y_pred and references/predictions aliases
- Converts HuggingFace evaluate outputs (dicts) to float via result_key / score / single-key dicts
"""

from __future__ import annotations
from typing import Any, Dict, Sequence, Optional
import importlib
import numpy as np

from fedcore.metrics.metric_impl import QualityMetric

# -----------------------------------------------------------------------------
# Lazy import for HuggingFace evaluate
# -----------------------------------------------------------------------------
_EVALUATE = None

def _get_evaluate():
    """Lazy import of the `evaluate` package."""
    global _EVALUATE
    if _EVALUATE is None:
        try:
            _EVALUATE = importlib.import_module("evaluate")
        except Exception as e:
            raise ImportError(
                "The 'evaluate' package is required for NLP metrics. "
                "Install it with: pip install evaluate"
            ) from e
    return _EVALUATE


# -----------------------------------------------------------------------------
# Base wrapper compatible with QualityMetric
# -----------------------------------------------------------------------------
class EvaluateMetric(QualityMetric):
import torch
from torch import nn
from typing import Any, Dict, Sequence, Optional

from fedcore.metrics.metric_impl import QualityMetric


# ---------------------------------------------------------------------------
# Base wrapper compatible with QualityMetric
# ---------------------------------------------------------------------------

class EvaluateMetric(QualityMetric):
    """
    Base class for NLP metrics powered by HuggingFace `evaluate`, compatible with QualityMetric.

    Subclasses must define:
      - metric_name: str                 (metric name in evaluate)
      - load_kwargs: Dict[str, Any]      (optional extra arguments for evaluate.load)
      - result_key: Optional[str]        (key in the evaluate output dict to extract)
      - output_mode: str                 ('texts' for generation, 'labels' for classification)
    """

    # ---- QualityMetric contract attributes ----
    default_value: float = 0.0
    need_to_minimize: bool = False
    split: str = "val"
    output_mode: str = "texts"  # default for text generation

    # ---- HuggingFace evaluate config ----
    metric_name: str = ""
    load_kwargs: Dict[str, Any] = {}
    result_key: Optional[str] = None

    @classmethod
    def _hf_metric(cls):
        """Lazy load the specific HuggingFace metric."""
        evaluate = _get_evaluate()
        return evaluate.load(cls.metric_name, **cls.load_kwargs)

    # ---- Required method: QualityMetric.metric ----
    @classmethod
    def metric(
        cls,
        target: Sequence[Any] | None,
        predict: Sequence[Any] | None,
        **kwargs: Any,
    ) -> float:
        """
        Unified calculation entry point.
        Accepts (target, predict) or aliases (references, predictions).
        Returns a float value extracted from evaluate's output.
        """
        references = kwargs.pop("references", None)
        predictions = kwargs.pop("predictions", None)
        if references is None:
            references = target
        if predictions is None:
            predictions = predict
        if references is None or predictions is None:
            raise ValueError("Both references (y_true) and predictions (y_pred) are required.")

        res = cls._hf_metric().compute(references=references, predictions=predictions, **kwargs)

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

    # ---- Required method: QualityMetric.get_value ----
    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        """
        Calls pipeline.predict(..., output_mode=cls.output_mode)
        and computes the metric on (references, predictions) from the given dataset split.
        """
        out = pipeline.predict(reference_data, output_mode=cls.output_mode)
        preds = out.predict.predict

        # Convert torch.Tensor to list if necessary
        try:
            import torch
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().tolist()
        except Exception:
            pass

        loader = getattr(reference_data.features, f"{cls.split}_dataloader")
        ds = loader.dataset

        # Common dataset field names for targets in NLP
        if hasattr(ds, "references"):
            refs = ds.references
        elif hasattr(ds, "targets"):
            refs = ds.targets
        elif hasattr(ds, "labels"):
            refs = ds.labels
        else:
            # Fallback: take second element from dataset iterator
            it = iter(ds)
            refs = [ex[1] for ex in it]

        return cls.metric(refs, preds)

    # ---- Optional: instance API for direct usage ----
    def __init__(self, **override_load_kwargs: Any) -> None:
        params = dict(self.load_kwargs)
        params.update(override_load_kwargs)
        self._metric_inst = _get_evaluate().load(self.metric_name, **params)

    def compute(
        self,
        y_true: Sequence[Any] | None = None,
        y_pred: Sequence[Any] | None = None,
        *,
        references: Sequence[Any] | None = None,
        predictions: Sequence[Any] | None = None,
    Base class for NLP metrics powered by PyTorch.
    """

    # ---- QualityMetric contract attributes ----
    default_value: float = 0.0
    need_to_minimize: bool = False
    split: str = "val"
    output_mode: str = "texts"  # default for text generation

    # ---- Required method: QualityMetric.metric ----
    @classmethod
    def metric(
        cls,
        target: Sequence[Any] | None,
        predict: Sequence[Any] | None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Direct HF evaluate compute() wrapper (kept for backward compatibility)."""
        if references is None and y_true is not None:
            references = y_true
        if predictions is None and y_pred is not None:
            predictions = y_pred
    ) -> float:
        """
        Unified calculation entry point.
        Accepts (target, predict) or aliases (references, predictions).
        Returns a float value.
        """
        references = kwargs.pop("references", None)
        predictions = kwargs.pop("predictions", None)
        if references is None:
            references = target
        if predictions is None:
            predictions = predict
        if references is None or predictions is None:
            raise ValueError("Both references and predictions are required.")
        return self._metric_inst.compute(predictions=predictions, references=references, **kwargs)
            raise ValueError("Both references (y_true) and predictions (y_pred) are required.")

        # Convert to torch tensors
        references = torch.tensor(references, dtype=torch.float32)
        predictions = torch.tensor(predictions, dtype=torch.float32)

        return cls._compute_metric(references, predictions, **kwargs)

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs):
        raise NotImplementedError

    # ---- Required method: QualityMetric.get_value ----
    @classmethod
    def get_value(cls, pipeline, reference_data, validation_blocks=None) -> float:
        """
        Calls pipeline.predict(..., output_mode=cls.output_mode)
        and computes the metric on (references, predictions) from the given dataset split.
        """
        out = pipeline.predict(reference_data, output_mode=cls.output_mode)
        preds = out.predict.predict

        # Convert torch.Tensor to list if necessary
        try:
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().tolist()
        except Exception:
            pass

        loader = getattr(reference_data.features, f"{cls.split}_dataloader")
        ds = loader.dataset

        # Common dataset field names for targets in NLP
        if hasattr(ds, "references"):
            refs = ds.references
        elif hasattr(ds, "targets"):
            refs = ds.targets
        elif hasattr(ds, "labels"):
            refs = ds.labels
        else:
            # Fallback: take second element from dataset iterator
            it = iter(ds)
            refs = [ex[1] for ex in it]

        return cls.metric(refs, preds)


# -----------------------------------------------------------------------------
# Concrete metrics
# -----------------------------------------------------------------------------
class NLPAccuracy(EvaluateMetric):
    """Classification accuracy for NLP tasks."""
    metric_name = "accuracy"
    result_key = "accuracy"
    output_mode = "labels"
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        correct = (references == predictions).sum().item()
        return correct / references.size(0)


class NLPPrecision(EvaluateMetric):
    """Macro-averaged precision for NLP classification."""
    metric_name = "precision"
    result_key = "precision"
    output_mode = "labels"
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        tp = (references * predictions).sum().item()
        fp = ((1 - references) * predictions).sum().item()
        return tp / (tp + fp)


class NLPRecall(EvaluateMetric):
    """Macro-averaged recall for NLP classification."""
    metric_name = "recall"
    result_key = "recall"
    output_mode = "labels"
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        tp = (references * predictions).sum().item()
        fn = (references * (1 - predictions)).sum().item()
        return tp / (tp + fn)


class NLPF1(EvaluateMetric):
    """Macro-averaged F1 score for NLP classification."""
    metric_name = "f1"
    result_key = "f1"
    output_mode = "labels"
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        tp = (references * predictions).sum().item()
        fp = ((1 - references) * predictions).sum().item()
        fn = (references * (1 - predictions)).sum().item()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)


class SacreBLEU(EvaluateMetric):
    """SacreBLEU metric for machine translation and text generation."""
    metric_name = "sacrebleu"
    result_key = "score"
    output_mode = "texts"
    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement SacreBLEU calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


class BLEU(SacreBLEU):
    """BLEU alias (maps to SacreBLEU)."""
    # Inherits everything from SacreBLEU
    pass


class ROUGE(EvaluateMetric):
    """ROUGE metric (L/SU/F variants) for text summarization."""
    metric_name = "rouge"
    # By default, we return rougeLsum (commonly used for summarization)
    result_key = "rougeLsum"
    output_mode = "texts"
    """ROUGE metric (L/SU/F variants) for text summarization."""
    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement ROUGE calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


class METEOR(EvaluateMetric):
    """METEOR metric for machine translation quality."""
    metric_name = "meteor"
    result_key = "meteor"
    output_mode = "texts"
    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement METEOR calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


class BERTScore(EvaluateMetric):
    """BERTScore metric based on contextual embeddings."""
    metric_name = "bertscore"
    output_mode = "texts"

    @classmethod
    def metric(cls, target, predict, **kwargs) -> float:
        """
        HuggingFace BERTScore output example:
        {'precision': [...], 'recall': [...], 'f1': [...]}
        Returns the mean F1 value.
        """
        res = cls._hf_metric().compute(references=target, predictions=predict, **kwargs)
        f1 = res.get("f1")
        if isinstance(f1, (list, tuple, np.ndarray)):
            return float(np.mean(f1))
        return float(f1)

    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement BERTScore calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


__all__ = [
    "EvaluateMetric",
    "NLPAccuracy",
    "NLPPrecision",
    "NLPRecall",
    "NLPF1",
    "SacreBLEU",
    "BLEU",
    "ROUGE",
    "METEOR",
    "BERTScore",
]

# Backward-compatibility alias
SacreBleu = SacreBLEU
