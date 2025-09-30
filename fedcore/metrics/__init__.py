# fedcore/metrics/__init__.py
from __future__ import annotations
from typing import Any, Callable, Dict

# ---- Простой реестр метрик ----
REGISTRY: Dict[str, Callable[..., Any]] = {}

def register(name: str, factory: Callable[..., Any]) -> None:
    """Регистрация метрики по имени. factory должен возвращать объект-метрику."""
    REGISTRY[name] = factory

def get(name: str, **kwargs: Any):
    """Получить инстанс метрики по имени (с возможными kwargs в конструктор)."""
    try:
        factory = REGISTRY[name]
    except KeyError as e:
        avail = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Metric '{name}' is not registered. Available: {avail}") from e
    return factory(**kwargs)

# ---- Автоподключение NLP-метрик из evaluate (опциональная зависимость) ----
try:
    from .nlp_metrics import available_nlp_metrics
    for _name, _factory in available_nlp_metrics().items():
        # и без префикса, и с префиксом "nlp:"
        register(_name, _factory)
        register(f"nlp:{_name}", _factory)
except Exception:
    # evaluate может быть не установлен — регистрация пропускается
    pass

