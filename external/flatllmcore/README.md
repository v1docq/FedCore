# FLAT-LLM Quick Start Guide

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from external.flatllmcore.core import AbsorptionCompressor

# 1. Load model
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Prepare calibration data
texts = ["Hello, how are you?", "The quick brown fox...", ...]  # 8-16 texts are enough
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
calibration_inputs = inputs.input_ids

# 3. Create compressor
compressor = AbsorptionCompressor(
    model=model,
    target_sparsity=0.7,  # 70% retention = 30% compression
    tolerance=0.96,
    device="cuda"
)

# 4. Choose layers to compress (every 3rd layer)
num_layers = len(model.model.layers)
layers_to_compress = list(range(0, num_layers, 3))

# 5. Collect activations (one forwarded pass)
compressor.collect_all_activations(
    layer_indices=layers_to_compress,
    calibration_input_ids=calibration_inputs
)

# 6. Apply compression
for layer_idx in layers_to_compress:
    compressor.apply_absorption_mlp(layer_idx, sparsity_ratio=0.7)
    compressor.apply_absorption_attention(layer_idx, sparsity_ratio=0.7)

# 7. Patch for inference
compressor.patch_compressed_layers(layers_to_compress)

# 8. Generation
output = model.generate(**tokenizer("Hello, I am", return_tensors="pt"), max_length=50)
print(tokenizer.decode(output[0]))
```

## Expexted Results

### TinyLlama 1.1B
- **Reduction**: 8-10% of the initial size
- **Quality**: Slightly worse than an original model
- **Time**: ~90 seconds

### Llama-2 7B (assumption)
- **Reduction**: 15-20% of the initial size
- **Quality**: <2% degradation on benchmarks
- **Time**: ~5-10 minutes

## Recommended params

| Model | Layers | Sparsity | Tolerance | Expected Reduction |
|-------|--------|----------|-----------|--------------------|
| 1B    | 30-40% | 0.7      | 0.96      | 8-12%              |
| 7B    | 25-35% | 0.8      | 0.98      | 15-20%             |
| 13B+  | 20-30% | 0.85     | 0.99      | 12-18%             |

## Main files

- `external/flatllmcore/core/absorption.py` - main realization
- `apply_flatllm_absorption.py` - working example


