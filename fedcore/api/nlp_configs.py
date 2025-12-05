from dataclasses import dataclass
from typing import (
    Any
)

from fedcore.api.api_configs import TrainingTemplate

@dataclass
class QAConfigTemplate(TrainingTemplate):
    model: Any = None
    tokenizer: Any = None


@dataclass
class SummarizationConfigTemplate(TrainingTemplate):
    model: Any = None
    tokenizer: Any = None