"""
Custom attention layers with FLAT-LLM modifications.
"""

from .attention_layers import (
    FlatLlamaAttention,
    FlatLlamaDecoderLayer,
    FlatMistralAttention,
    FlatMistralDecoderLayer
)

__all__ = [
    "FlatLlamaAttention",
    "FlatLlamaDecoderLayer", 
    "FlatMistralAttention",
    "FlatMistralDecoderLayer"
]
