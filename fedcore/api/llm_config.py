from dataclasses import dataclass
from typing import (
    Any, Dict, Optional,
)

from fedcore.api.api_configs import TrainingTemplate

@dataclass
class LLMConfigTemplate(TrainingTemplate):
    """Configuration template for LLM-specific parameters"""
    is_llm: bool = True
    model: Any = None
    tokenizer: Any = None
    fedcore_id: Optional[str] = None