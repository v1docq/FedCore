from dataclasses import dataclass
from typing import (
    Any, Dict, Optional,
)

from fedcore.api.api_configs import NeuralModelConfigTemplate

@dataclass
class QAConfigTemplate(NeuralModelConfigTemplate):
    model: Any = None
    tokenizer: Any = None


@dataclass
class SummarizationConfigTemplate(NeuralModelConfigTemplate):
    model: Any = None
    tokenizer: Any = None