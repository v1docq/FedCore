"""Model reassembly; transformer backends load only when requested."""
from importlib import import_module
from .core_reassemblers import Reassembler, ParentalReassembler, get_reassembler, REASSEMBLERS
from .config_mixins import ConfigAnalysisMixin
from .decomposed_recreation import RECREATION_FUNCTIONS

_LAZY_EXPORTS = {
    "TransMLA": ".transmla_reassembler",
    "TransMLAConfig": ".transmla_reassembler",
    "get_transmla_status": ".transmla_reassembler",
    "FlatLLM": ".flatllm_reassembler",
    "FlatLLMConfig": ".flatllm_reassembler",
    "get_flatllm_status": ".flatllm_reassembler",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = ["Reassembler", "ParentalReassembler", "ConfigAnalysisMixin",
           "get_reassembler", "REASSEMBLERS", "RECREATION_FUNCTIONS", *_LAZY_EXPORTS]
