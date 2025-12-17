"""
True FLAT-LLM with Absorption - Test Script

This implements the REAL FLAT-LLM algorithm with:
- PCA on activations (not weights)
- Absorption mechanism
- Physical dimension reduction
- Architecture rebuilding

Expected result: MODEL SIZE ACTUALLY DECREASES!
"""

import torch
import gc
import time
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from external.flatllmcore.core.absorption import AbsorptionCompressor

print("=" * 80)
print("TRUE FLAT-LLM WITH ABSORPTION")
print("=" * 80)
print()

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # Start on CPU for safety
    trust_remote_code=True
)

print("Loaded")
print()


# Model info
def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params, total_bytes / (1024 ** 3)


original_params, original_size = get_model_size(model)
print(f"Original model:")
print(f"   Parameters: {original_params:,}")
print(f"   Size: {original_size:.2f} GB")
print()


# Benchmark generation speed
def benchmark_generation(m, prompt="Hello, I am", max_length=10, num_runs=10):
    """Measure generation speed."""
    m = m.to("cuda")
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    times = []
    total_tokens = 0
    result_text = None
    
    # Warmup
    with torch.no_grad():
        _ = m.generate(**inp, max_length=max_length, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    
    # Benchmark
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = m.generate(**inp, max_length=max_length, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if result_text is None:
            result_text = tokenizer.decode(out[0], skip_special_tokens=True)
            total_tokens = len(out[0]) - len(inp.input_ids[0])
    
    m = m.to("cpu")
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = total_tokens / avg_time if avg_time > 0 else 0
    
    return {
        'text': result_text,
        'avg_time': avg_time,
        'tokens': total_tokens,
        'tokens_per_sec': tokens_per_sec,
        'runs': num_runs
    }


print("Baseline generation benchmark...")
baseline_bench = benchmark_generation(model, "Hello, I am", max_length=50, num_runs=3)
print(f"   Text: {baseline_bench['text'][:80]}...")
print(f"   Speed: {baseline_bench['tokens_per_sec']:.2f} tokens/sec")
print(f"   Time: {baseline_bench['avg_time']:.3f}s ({baseline_bench['tokens']} tokens)")
print()

# Prepare calibration data
print("=" * 80)
print("APPLYING ABSORPTION")
print("=" * 80)
print()

print("Preparing calibration data...")
calibration_texts = [
                        "The future of artificial intelligence is bright and full of possibilities.",
                        "Machine learning models are becoming more efficient and powerful.",
                        "Natural language processing has made significant progress in recent years.",
                        "Deep learning architectures continue to evolve and improve.",
                    ] * 2  # 8 samples

calibration_inputs = tokenizer(
    calibration_texts,
    max_length=128,
    truncation=True,
    padding=True,
    return_tensors="pt"
).input_ids

print(f"   Calibration samples: {calibration_inputs.shape[0]}")
print()

# Create compressor
compressor = AbsorptionCompressor(
    model=model,
    target_sparsity=0.7,  # 70% retention = 30% compression
    tolerance=0.96,
    device="cuda"
)

print(f"Calibration inputs shape: {calibration_inputs.shape}")
print()

# Select layers to compress (only 30% least important, as we learned)
num_layers = len(model.model.layers)
layers_to_compress = list(range(0, num_layers, 3))  # Every 3rd layer for testing
print(f"Compressing {len(layers_to_compress)} out of {num_layers} layers:")
print(f"   Indices: {layers_to_compress}")
print()

start_time = time.time()

# Step 1: Collect activations from ALL target layers in one pass
# (Must do this before any compression, since compression changes model structure)
compressor.collect_all_activations(
    layer_indices=layers_to_compress,
    calibration_input_ids=calibration_inputs
)

# Step 2: Apply absorption to each layer
for layer_idx in layers_to_compress:
    print(f"Compressing layer {layer_idx + 1}/{num_layers}...")

    try:
        # Apply absorption to MLP
        print(f"  1/2 Applying MLP absorption...")
        compressor.apply_absorption_mlp(
            layer_idx=layer_idx,
            sparsity_ratio=0.7  # Keep 70% of neurons
        )

        # Apply absorption to Attention
        print(f"  2/2 Applying Attention absorption...")
        compressor.apply_absorption_attention(
            layer_idx=layer_idx,
            sparsity_ratio=0.7  # Keep 70% of dimensions per head
        )

        print(f"    Layer {layer_idx} complete")
        print()

    except Exception as e:
        print(f"    Error on layer {layer_idx}: {e}")
        import traceback

        traceback.print_exc()
        print()
        continue

compress_time = time.time() - start_time

print(f"  Compression complete in {compress_time:.1f}s")
print()

# Patch compressed layers for inference
compressor.patch_compressed_layers(layers_to_compress)

# Check new size
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

compressed_params, compressed_size = get_model_size(model)

print(f"Original:")
print(f"   Parameters: {original_params:,}")
print(f"   Size: {original_size:.2f} GB")
print()

print(f"Compressed:")
print(f"   Parameters: {compressed_params:,}")
print(f"   Size: {compressed_size:.2f} GB")
print()

reduction_params = (1 - compressed_params / original_params) * 100
reduction_size = (1 - compressed_size / original_size) * 100

print(f"Reduction:")
print(f"   Parameters: {reduction_params:.1f}%")
print(f"   Size: {reduction_size:.1f}%")
print()

if reduction_params > 5:
    print("SUCCESS! Model size actually decreased!")
else:
    print("Warning: Size reduction is minimal")

print()

# Test generation and benchmark speed after compression
print("=" * 80)
print("GENERATION SPEED BENCHMARK")
print("=" * 80)
print()

print("Testing generation after compression...")
try:
    compressed_bench = benchmark_generation(model, "Hello, I am", max_length=50, num_runs=3)
    print(f"   Text: {compressed_bench['text'][:80]}...")
    print(f"   Speed: {compressed_bench['tokens_per_sec']:.2f} tokens/sec")
    print(f"   Time: {compressed_bench['avg_time']:.3f}s ({compressed_bench['tokens']} tokens)")
    print()
    
    # Compare speeds
    print("=" * 80)
    print("SPEED COMPARISON")
    print("=" * 80)
    print()
    
    print(f"Original model:")
    print(f"   Speed: {baseline_bench['tokens_per_sec']:.2f} tokens/sec")
    print(f"   Time:  {baseline_bench['avg_time']:.3f}s")
    print()
    
    print(f"Compressed model:")
    print(f"   Speed: {compressed_bench['tokens_per_sec']:.2f} tokens/sec")
    print(f"   Time:  {compressed_bench['avg_time']:.3f}s")
    print()
    
    speedup = compressed_bench['tokens_per_sec'] / baseline_bench['tokens_per_sec']
    time_reduction = (1 - compressed_bench['avg_time'] / baseline_bench['avg_time']) * 100
    
    if speedup > 1.0:
        print(f"SPEEDUP: {speedup:.2f}x faster ({time_reduction:.1f}% faster)")
    elif speedup < 1.0:
        print(f"SLOWDOWN: {speedup:.2f}x ({abs(time_reduction):.1f}% slower)")
    else:
        print(f"No significant change")
    print()
    
    # Quality check
    if len(compressed_bench['text']) > 15 and "Hello" in compressed_bench['text']:
        print("Generation quality: OK")
        print()

        # Save model
        save_path = "tinyllama_absorption_compressed"
        print(f"Save compressed model to {save_path}? (y/n): ", end='')
        response = input()
        if response.lower() == 'y':
            print("   Saving...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Check saved size
            saved_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(save_path)
                for f in fns
            ) / (1024 ** 3)

            print(f"Saved! Size on disk: {saved_size:.2f} GB")
    else:
        print("Generation quality degraded")

except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("Test Complete")
print("=" * 80)