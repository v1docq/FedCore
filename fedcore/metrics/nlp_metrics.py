# fedcore/metrics/nlp.py
from __future__ import annotations
from typing import Any, Callable, Dict, Sequence
import importlib

_EVALUATE = None  # lazy
def _get_evaluate():
    global _EVALUATE
    if _EVALUATE is None:
        try:
            _EVALUATE = importlib.import_module("evaluate")
        except Exception as e:
            raise ImportError(
                "Библиотека 'evaluate' не установлена. "
                "Установи: pip install evaluate"
            ) from e
    return _EVALUATE

class EvaluateMetric:
    """
    Универсальная обёртка над любой метрикой из evaluate.
    Пример:
        acc = EvaluateMetric("accuracy")
        acc.compute(y_true=[0,1], y_pred=[0,1]) -> {"accuracy": 1.0}
    """
    def __init__(self, name: str, **load_kwargs: Any) -> None:
        self.name = name
        self.load_kwargs = load_kwargs
        evaluate = _get_evaluate()
        self._module = evaluate.load(self.name, **self.load_kwargs)  # type: ignore

    def compute(
        self,
        y_true: Sequence[Any] | None = None,
        y_pred: Sequence[Any] | None = None,
        *,
        predictions: Sequence[Any] | None = None,
        references: Sequence[Any] | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        if predictions is None and y_pred is not None:
            predictions = y_pred
        if references is None and y_true is not None:
            references = y_true
        if predictions is None or references is None:
            raise ValueError("Нужно передать predictions/y_pred и references/y_true")
        return self._module.compute(predictions=predictions, references=references, **kwargs)  # type: ignore

def available_nlp_metrics() -> Dict[str, Callable[..., EvaluateMetric]]:
    return {
        "accuracy":  lambda **kw: EvaluateMetric("accuracy", **kw),
        "precision": lambda **kw: EvaluateMetric("precision", **kw),
        "recall":    lambda **kw: EvaluateMetric("recall", **kw),
        "f1":        lambda **kw: EvaluateMetric("f1", **kw),
        "sacrebleu": lambda **kw: EvaluateMetric("sacrebleu", **kw),
        "bleu":      lambda **kw: EvaluateMetric("sacrebleu", **kw),  # алиас
        "rouge":     lambda **kw: EvaluateMetric("rouge", **kw),
        "meteor":    lambda **kw: EvaluateMetric("meteor", **kw),
        "bertscore": lambda **kw: EvaluateMetric("bertscore", **kw),
    }
