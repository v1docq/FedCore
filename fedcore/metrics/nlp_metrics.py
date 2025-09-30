"""
NLP-specific metrics implemented as thin wrappers over the HuggingFace `evaluate` package.

Each metric is exposed as a class with a unified interface:
    m = SacreBLEU()
    result = m.compute(y_true=references, y_pred=predictions)

The wrapper ensures consistent naming and lightweight lazy import of `evaluate`.
"""

from __future__ import annotations
from typing import Any, Dict, Sequence
import importlib

# Cache for lazy import
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


class EvaluateMetric:
    """
    Base class for NLP metrics powered by HuggingFace `evaluate`.

    Subclasses must define `metric_name` and may override `load_kwargs`.
    Provides a unified `.compute(y_true, y_pred, ...)` method.
    """

    metric_name: str = ""
    load_kwargs: dict[str, Any] = {}

    def __init__(self, **override_load_kwargs: Any) -> None:
        evaluate = _get_evaluate()
        params = dict(self.load_kwargs)
        params.update(override_load_kwargs)
        self._metric = evaluate.load(self.metric_name, **params)

    def compute(
        self,
        y_true: Sequence[Any] | None = None,
        y_pred: Sequence[Any] | None = None,
        *,
        references: Sequence[Any] | None = None,
        predictions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute the metric.

        Accepts both (y_true, y_pred) and (references, predictions).
        Extra keyword args are passed through to the underlying `evaluate` metric.
        """
        if references is None and y_true is not None:
            references = y_true
        if predictions is None and y_pred is not None:
            predictions = y_pred
        if references is None or predictions is None:
            raise ValueError("Both references (y_true) and predictions (y_pred) must be provided.")
        return self._metric.compute(predictions=predictions, references=references, **kwargs)


# ---------------------------------------------------------------------------
# Concrete metrics
# ---------------------------------------------------------------------------

class NLPAccuracy(EvaluateMetric):
    """Classification accuracy for NLP tasks."""
    metric_name = "accuracy"


class NLPPrecision(EvaluateMetric):
    """Macro-averaged precision for NLP classification."""
    metric_name = "precision"


class NLPRecall(EvaluateMetric):
    """Macro-averaged recall for NLP classification."""
    metric_name = "recall"


class NLPF1(EvaluateMetric):
    """Macro-averaged F1 score for NLP classification."""
    metric_name = "f1"


class SacreBLEU(EvaluateMetric):
    """SacreBLEU metric for machine translation and text generation."""
    metric_name = "sacrebleu"


class BLEU(SacreBLEU):
    """BLEU alias (maps to SacreBLEU)."""
    # Inherits metric_name = "sacrebleu"


class ROUGE(EvaluateMetric):
    """ROUGE metric (L/SU/F variants) for text summarization overlap."""
    metric_name = "rouge"


class METEOR(EvaluateMetric):
    """METEOR metric for machine translation quality."""
    metric_name = "meteor"


class BERTScore(EvaluateMetric):
    """BERTScore metric based on contextual embeddings."""
    metric_name = "bertscore"

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
