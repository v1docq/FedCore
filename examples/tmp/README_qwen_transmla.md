# Qwen2.5-0.5B TransMLA Integration Example

This example demonstrates how to use TransMLA (Multi-head Latent Attention) with Qwen2.5-0.5B model using the FedCore framework.

## Overview

TransMLA is an advanced attention mechanism that can significantly reduce memory usage and computational overhead while maintaining model performance. This example shows three different ways to apply TransMLA to Qwen2.5-0.5B:

1. **Deferred Conversion**: Create a conversion task that can be executed later
2. **Immediate Conversion**: Apply TransMLA conversion immediately
3. **AttentionReassembler**: Use the flexible reassembler interface

## Requirements

```bash
pip install transformers>=4.40.0
pip install torch>=2.0.0
pip install packaging
```

## Quick Start

### Basic Usage

```bash
# Run all demos
python examples/qwen_transmla_demo.py

# Run only deferred conversion
python examples/qwen_transmla_demo.py --mode deferred

# Run with model saving
python examples/qwen_transmla_demo.py --save-path ./qwen_mla_model
```

### Programmatic Usage

```python
from fedcore.algorithm.quantization.utils import TransMLA, TransMLAConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Create configuration
config = TransMLAConfig(
    qk_mqa_dim=64,        # Match Qwen2.5-0.5B head_dim
    kv_lora_rank=128,     # Must be < 2*latent_dim - qk_mqa_dim = 2*128-64 = 192
    cal_nsamples=32,      # Calibration samples
)

# Method 1: Deferred conversion
deferred = TransMLA.convert(model, tokenizer=tokenizer, config=config, deferred=True)
converted_model = deferred.execute()  # Execute when ready

# Method 2: Immediate conversion
converted_model = TransMLA.convert(model, tokenizer=tokenizer, config=config, deferred=False)

# Method 3: Using AttentionReassembler
from fedcore.algorithm.quantization.utils import AttentionReassembler
converted_model = AttentionReassembler.convert(model, mode='trans-mla', tokenizer=tokenizer, config=config)
```

## Configuration Options

### TransMLAConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `qk_mqa_dim` | 64 | Query-Key MQA dimension (should match head_dim) |
| `kv_lora_rank` | 128 | Key-Value LoRA rank (must be < 2*latent_dim - qk_mqa_dim) |
| `collapse` | "auto" | Collapse factor (auto-calculated) |
| `cal_nsamples` | 32 | Number of calibration samples |
| `cal_batch_size` | 2 | Calibration batch size |
| `cal_max_seqlen` | 128 | Maximum sequence length for calibration |

### Model-Specific Optimizations for Qwen2.5-0.5B

```python
config = TransMLAConfig(
    qk_mqa_dim=64,        # Matches Qwen2.5-0.5B head_dim (896/14=64)
    kv_lora_rank=128,     # Must be < 2*latent_dim - qk_mqa_dim = 2*128-64 = 192
    cal_nsamples=32,      # Sufficient for 0.5B model calibration
    cal_batch_size=2,     # Memory-efficient for testing
    cal_max_seqlen=128,   # Shorter sequences for faster processing
    deepseek_style=True,  # Use DeepSeek-style implementation
    dtype="fp16",         # Use half precision if CUDA available
)
```

## Expected Output

The demo will show:

1. **Model Loading**: Qwen2.5-0.5B model with ~494M parameters
2. **Original Performance**: Text generation before conversion
3. **Conversion Process**: TransMLA calibration and conversion
4. **Converted Performance**: Text generation after conversion
5. **Memory Usage**: Comparison of memory usage (if applicable)

## Architecture Details

### Qwen2.5-0.5B Specifications

- **Parameters**: ~494M
- **Hidden Size**: 896
- **Attention Heads**: 14
- **Head Dimension**: 64 (896/14)
- **Layers**: 24
- **Vocabulary**: 151,936 tokens

### TransMLA Benefits

- **Memory Reduction**: Significant reduction in attention memory usage
- **Speed Improvement**: Faster attention computation
- **Quality Preservation**: Maintains text generation quality
- **Flexibility**: Works with existing Qwen models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `cal_batch_size` or `cal_max_seqlen`
2. **Slow Conversion**: Reduce `cal_nsamples` for faster testing
3. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

- Use GPU if available for faster conversion
- Start with smaller `cal_nsamples` for initial testing
- Use `deferred=True` for batch processing scenarios
- Save converted models to avoid re-conversion

## Advanced Usage

### Custom Configuration

```python
# High-performance configuration
config = TransMLAConfig(
    qk_mqa_dim=64,
    kv_lora_rank=512,     # Higher rank for better quality
    cal_nsamples=128,     # More samples for better calibration
    cal_batch_size=8,     # Larger batch if memory allows
    cal_max_seqlen=512,   # Longer sequences
    use_qkv_norm=True,    # Enable QKV normalization
)

# Memory-efficient configuration
config = TransMLAConfig(
    qk_mqa_dim=32,        # Smaller dimension
    kv_lora_rank=128,     # Lower rank
    cal_nsamples=16,      # Fewer samples
    cal_batch_size=1,     # Smallest batch
    cal_max_seqlen=64,    # Shorter sequences
)
```

### Batch Processing

```python
# Process multiple models with deferred conversion
models_to_convert = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
deferred_conversions = []

for model_name in models_to_convert:
    model, tokenizer = load_model(model_name)
    deferred = TransMLA.convert(model, tokenizer=tokenizer, config=config, deferred=True)
    deferred_conversions.append(deferred)

# Execute all conversions
converted_models = [d.execute() for d in deferred_conversions]
```

## Integration with FedCore

This example integrates seamlessly with the FedCore ecosystem:

- **Quantization**: Combine with other FedCore quantization techniques
- **Pruning**: Apply TransMLA after model pruning
- **Federated Learning**: Use converted models in federated scenarios
- **Model Compression**: Part of comprehensive compression pipeline

For more information, see the main FedCore documentation.
