import torch
from torch import nn
from typing import Any, Dict, Sequence, Optional

from fedcore.metrics.metric_impl import QualityMetric


# ---------------------------------------------------------------------------
# Base wrapper compatible with QualityMetric
# ---------------------------------------------------------------------------

class EvaluateMetric(QualityMetric):
    """
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


# ---------------------------------------------------------------------------
# Concrete metrics
# ---------------------------------------------------------------------------

class NLPAccuracy(EvaluateMetric):
    """Classification accuracy for NLP tasks."""
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        correct = (references == predictions).sum().item()
        return correct / references.size(0)


class NLPPrecision(EvaluateMetric):
    """Macro-averaged precision for NLP classification."""
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        tp = (references * predictions).sum().item()
        fp = ((1 - references) * predictions).sum().item()
        return tp / (tp + fp)


class NLPRecall(EvaluateMetric):
    """Macro-averaged recall for NLP classification."""
    output_mode = "labels"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        tp = (references * predictions).sum().item()
        fn = (references * (1 - predictions)).sum().item()
        return tp / (tp + fn)


class NLPF1(EvaluateMetric):
    """Macro-averaged F1 score for NLP classification."""
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
    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement SacreBLEU calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


class BLEU(SacreBLEU):
    """BLEU alias (maps to SacreBLEU)."""
    pass


class ROUGE(EvaluateMetric):
    """ROUGE metric (L/SU/F variants) for text summarization."""
    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement ROUGE calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


class METEOR(EvaluateMetric):
    """METEOR metric for machine translation quality."""
    output_mode = "texts"

    @classmethod
    def _compute_metric(cls, references, predictions, **kwargs) -> float:
        # Implement METEOR calculation here (can use an external package or custom code)
        return float(0.0)  # Placeholder


class BERTScore(EvaluateMetric):
    """BERTScore metric based on contextual embeddings."""
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
