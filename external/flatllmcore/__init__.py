"""Calibration prototype exports; LLM dependencies load on demand."""
from importlib import import_module

__version__ = "0.1.0"
_EXPORTS = {'FlatLLMPruner': '.core.prune', 'ImportancePreservingRankSelector': '.core.rank_allocation', 'FlatLlamaAttention': '.layers.attention_layers', 'FlatLlamaDecoderLayer': '.layers.attention_layers', 'FlatMistralAttention': '.layers.attention_layers', 'FlatMistralDecoderLayer': '.layers.attention_layers'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
