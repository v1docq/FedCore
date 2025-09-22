# FLAT-LLM Core Module

**Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression**

This module provides a minimal implementation of FLAT-LLM algorithm integrated with FedCore's reassembly framework.

## Overview

FLAT-LLM applies fine-grained low-rank transformations in the activation space of large language models, using:

- **Importance-Preserving Rank Selection (IPRS)**: Adaptive rank allocation based on layer importance
- **Head-wise PCA**: Individual PCA transformations for each attention head  
- **Selective Pruning**: Preserves Q,K projections while compressing V,O,MLP layers

## Features

- ✅ Support for Llama-2, Llama-3, and Mistral architectures
- ✅ Configurable compression ratios (20% to 90%)
- ✅ Head-wise attention transformations
- ✅ Integration with FedCore reassembly framework
- ✅ Calibration-based importance scoring
- ✅ Memory-efficient layer-by-layer processing

## Quick Start

### Basic Usage

```python
from flatllmcore import FlatLLMReassembler, FlatLLMConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create configuration
config = FlatLLMConfig(
    target_sparsity=0.5,    # 50% compression
    tolerance=0.96,         # 96% variance preservation
    cal_nsamples=128        # Calibration samples
)

# Apply FLAT-LLM transformation
compressed_model = FlatLLMReassembler.reassemble(model, tokenizer, config)
```

### Convenience Functions

```python
from flatllmcore import apply_flat_llm, compute_layer_importance

# Quick compression
compressed_model = apply_flat_llm(
    model, tokenizer, 
    target_sparsity=0.6,
    cal_dataset="wikitext2"
)

# Analyze layer importance
importance_scores = compute_layer_importance(model, tokenizer)
print(f"Layer importance scores: {importance_scores}")
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_sparsity` | Target compression ratio (0.0-1.0) | 0.5 |
| `tolerance` | Eigenvalue preservation threshold | 0.96 |
| `cal_dataset` | Calibration dataset name | "wikitext2" |
| `cal_nsamples` | Number of calibration samples | 128 |
| `importance_method` | Importance computation method | "angular" |
| `preserve_qk_layers` | Preserve Q,K layers (FLAT-LLM strategy) | True |

## Architecture Components

### Core Algorithms (`core/`)

- **`FlatLLMPruner`**: Main pruning algorithm implementation
- **`ImportancePreservingRankSelector`**: IPRS algorithm for rank allocation

### Custom Layers (`layers/`)

- **`FlatLlamaAttention`**: FLAT-LLM optimized Llama attention
- **`FlatMistralAttention`**: FLAT-LLM optimized Mistral attention
- **Decoder Layers**: Corresponding decoder layer implementations

### Utilities (`utils/`)

- **`layer_utils.py`**: Layer discovery and analysis functions
- **`data_utils.py`**: Calibration data preparation and statistics

## Integration with FedCore

The `FlatLLMReassembler` class extends FedCore's `Reassembler` base class:

```python
from fedcore.algorithm.low_rank.reassembly.core_reassemblers import get_reassembler

# Register FLAT-LLM reassembler (add to core_reassemblers.py)
REASSEMBLERS['flat-llm'] = FlatLLMReassembler

# Use via FedCore interface
reassembler = get_reassembler('flat-llm')
compressed_model = reassembler.reassemble(model, tokenizer=tokenizer)
```

## Model Support

### Supported Architectures

- **Llama-2** (7B, 13B, 70B)
- **Llama-3** (8B)
- **Mistral** (7B)

### Architecture Detection

The module automatically detects model architecture and applies appropriate transformations:

```python
# Automatic detection based on model.config
architecture = FlatLLMReassembler._detect_model_architecture(model)
print(f"Detected: {architecture}")  # "llama" or "mistral"
```

## Advanced Usage

### Custom Importance Computation

```python
# Compute importance scores separately
importance_scores = FlatLLMReassembler.compute_importance_scores(
    model, tokenizer,
    config=FlatLLMConfig(importance_method="gradient")
)

# Allocate ranks based on importance
rank_allocation = FlatLLMReassembler.allocate_ranks(
    importance_scores,
    target_sparsity=0.4
)

print(f"Rank allocation per layer: {rank_allocation}")
```

### Layer-specific Analysis

```python
from external.flatllmcore.utils import check_sparsity, get_layer_dimensions

# Analyze current sparsity
sparsity = check_sparsity(model)
print(f"Current sparsity: {sparsity:.4f}")

# Get model dimensions
dimensions = get_layer_dimensions(model, "llama")
print(f"Model dimensions: {dimensions}")
```

### Compression Statistics

```python
# After applying FLAT-LLM
pruner = FlatLLMPruner(model, tokenizer, target_sparsity=0.5)
pruned_model = pruner.prune_model()

stats = pruner.get_compression_stats()
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Parameter reduction: {stats['sparsity']:.1%}")
```

## Performance Considerations

### Memory Optimization

- Processes layers individually to minimize memory usage
- Moves layers to CPU during processing
- Supports multi-GPU models via device mapping

### Calibration Efficiency

- Uses 128 samples by default (adjustable)
- Supports batch processing for efficiency
- Caches calibration data between layers

## Implementation Notes

### Key Differences from Original

1. **Modular Design**: Separated into reusable components
2. **FedCore Integration**: Implements `Reassembler` interface
3. **Architecture Agnostic**: Supports multiple model types
4. **Memory Efficient**: Layer-by-layer processing
5. **Configurable**: Extensive configuration options

### FLAT-LLM Algorithm Steps

1. **Importance Scoring**: Compute layer-wise importance using angular method
2. **Rank Allocation**: Apply IPRS algorithm for adaptive rank distribution  
3. **Layer Replacement**: Replace standard attention with FLAT-LLM versions
4. **Head-wise Pruning**: Apply PCA transformations per attention head
5. **Validation**: Ensure device consistency and compression targets

## Citation

If you use this implementation, please cite the original FLAT-LLM paper:

```bibtex
@article{tian2025flat,
    title={FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression},
    author={Tian, Jiayi and Solgi, Ryan and Lu, Jinming and Yang, Yifan and Li, Hai and Zhang, Zheng},
    journal={arXiv preprint arXiv:2505.23966},
    year={2025}
}
```

## License

This implementation follows the MIT License of the original FLAT-LLM repository.
