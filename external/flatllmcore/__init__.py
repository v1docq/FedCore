"""
FLAT-LLM Core Module

Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression.
Minimal implementation for FedCore integration.

Based on: "FLAT-LLM: Fine-grained Low-rank Activation Space Transformation 
for Large Language Model Compression" (arXiv:2505.23966)
"""

from .core.prune import FlatLLMPruner
from .core.rank_allocation import ImportancePreservingRankSelector
from .layers.attention_layers import (
    FlatLlamaAttention, 
    FlatLlamaDecoderLayer,
    FlatMistralAttention,
    FlatMistralDecoderLayer
)

__version__ = "0.1.0"
__all__ = [
    "FlatLLMPruner",
    "ImportancePreservingRankSelector",
    "FlatLlamaAttention",
    "FlatLlamaDecoderLayer",
    "FlatMistralAttention", 
    "FlatMistralDecoderLayer"
]
