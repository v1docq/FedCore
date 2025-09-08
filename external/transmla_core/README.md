# TransMLA Core

Minimal Python module for Multi-head Latent Attention (MLA) transformations.

## Overview

This is a cleaned-up, minimal version of TransMLA containing only the essential components needed for MLA model conversions. It removes all unnecessary files and focuses on core functionality.

## Components

- **`utils.py`** - Dataset loading and evaluation utilities
- **`partial_rope.py`** - Partial RoPE (Rotary Position Embedding) transformations
- **`lora_qkv.py`** - Low-rank QKV (Query-Key-Value) operations
- **`modify_config.py`** - Model configuration modifications

## Usage

```python
from external.transmla_core import (
    get_dataset, 
    prepare_dataloader,
    partial_rope,
    low_rank_qkv,
    modify_config
)

# Example usage in FedCore
from fedcore.algorithm.quantization.utils import TransMLA

model = TransMLA.convert(model, tokenizer=tokenizer)
```

## Integration with FedCore

This module is automatically used by `fedcore.algorithm.quantization.utils` when TransMLA functionality is needed:

```python
from fedcore.algorithm.quantization.utils import AttentionReassembler

# Standard conversion
model = AttentionReassembler.convert(model, mode='standard')

# TransMLA conversion (uses transmla_core)
model = AttentionReassembler.convert(model, mode='trans-mla', tokenizer=tokenizer)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.52.0
- datasets
- numpy
- tqdm
