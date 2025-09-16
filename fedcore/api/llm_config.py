from dataclasses import dataclass
from typing import (
    Any, Dict, Optional,
)

from fedcore.api.api_configs import NeuralModelConfigTemplate

@dataclass
class LLMConfigTemplate(NeuralModelConfigTemplate):
    """Configuration template for LLM-specific parameters"""
    is_llm: bool = True  # Default to True for LLM
    model: Any = None
    tokenizer: Any = None