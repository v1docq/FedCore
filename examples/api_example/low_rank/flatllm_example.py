"""
Example of using FLAT-LLM reassembler for model compression.

This example demonstrates how to use the FlatLLM reassembler to compress
a Large Language Model using the absorption mechanism.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import FlatLLM from fedcore
from fedcore.algorithm.low_rank.reassembly import FlatLLM, FlatLLMConfig, get_flatllm_status


def main():
    print("=" * 80)
    print("FLAT-LLM Compression Example")
    print("=" * 80)
    print()
    
    # Check FLAT-LLM availability
    status = get_flatllm_status()
    print(f"FLAT-LLM Status:")
    print(f"  Available: {status['available']}")
    print(f"  Path: {status['path']}")
    if not status['available']:
        print(f"  Error: {status['error']}")
        return
    print()
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Test generation before compression
    print("Testing generation before compression...")
    test_prompt = "Hello, I am"
    model_device = next(model.parameters()).device
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model_device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
    
    original_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {test_prompt}")
    print(f"  Output: {original_text}")
    print()
    
    # Example 1: Auto configuration
    print("=" * 80)
    print("Example 1: Auto Configuration")
    print("=" * 80)
    print()
    
    config = FlatLLMConfig.auto_from_model(
        model,
        priority="balanced",
        compression="medium"
    )
    
    print("Auto-generated configuration:")
    print(f"  target_sparsity: {config.target_sparsity}")
    print(f"  tolerance: {config.tolerance}")
    print(f"  compression_ratio: {config.compression_ratio}")
    print(f"  cal_nsamples: {config.cal_nsamples}")
    print()
    
    # Compress model
    print("Compressing model...")
    compressed_model = FlatLLM.reassemble(model, tokenizer, config)
    
    print(f"Compression complete!")
    print(f"  Parameters: {sum(p.numel() for p in compressed_model.parameters()):,}")
    print()
    
    # Test generation after compression
    print("Testing generation after compression...")
    compressed_device = next(compressed_model.parameters()).device
    inputs = tokenizer(test_prompt, return_tensors="pt").to(compressed_device)
    
    with torch.no_grad():
        outputs = compressed_model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
    
    compressed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {test_prompt}")
    print(f"  Output: {compressed_text}")
    print()
    
    # Example 2: Manual configuration
    print("=" * 80)
    print("Example 2: Manual Configuration")
    print("=" * 80)
    print()
    
    # Reload model for the second example
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    config = FlatLLMConfig(
        target_sparsity=0.7,  # Keep 70% = 30% reduction
        tolerance=0.96,        # Keep 96% variance
        layer_selection=[0, 3, 6, 9],  # Explicit layer indices
        cal_nsamples=8,
        verbose=True
    )
    
    print("Manual configuration:")
    print(f"  target_sparsity: {config.target_sparsity}")
    print(f"  layer_selection: {config.layer_selection}")
    print()
    
    # Compress with manual config
    print("Compressing model with manual config...")
    compressed_model = FlatLLM.reassemble(model, tokenizer, config)
    
    print(f"Compression complete!")
    print(f"  Parameters: {sum(p.numel() for p in compressed_model.parameters()):,}")
    print()
    
    # Example 3: Using kwargs to override config
    print("=" * 80)
    print("Example 3: Override Config with Kwargs")
    print("=" * 80)
    print()
    
    # Reload model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Use default config but override specific parameters
    compressed_model = FlatLLM.reassemble(
        model,
        tokenizer,
        config=None,  # Use default config
        target_sparsity=0.8,  # Override: less compression
        cal_nsamples=4,       # Override: fewer samples (faster)
        verbose=True
    )
    
    print(f"Compression complete!")
    print(f"  Parameters: {sum(p.numel() for p in compressed_model.parameters()):,}")
    print()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

