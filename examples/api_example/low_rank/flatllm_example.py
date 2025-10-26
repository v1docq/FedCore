"""
Example of using FLAT-LLM reassembler for model compression.

This example demonstrates how to use the FlatLLM reassembler to compress
a Large Language Model using the absorption mechanism.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import FlatLLM from fedcore
from fedcore.algorithm.low_rank.reassembly import FlatLLM, FlatLLMConfig, get_flatllm_status


def load_model(model_name: str):
    """Load model and tokenizer."""
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

    return model, tokenizer


def test_generation(model, tokenizer, test_prompt: str = "Hello, I am"):
    """Test model generation with a prompt."""
    model_device = next(model.parameters()).device
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model_device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {test_prompt}")
    print(f"  Output: {generated_text}")
    print()

    return generated_text


def print_config_details(config: FlatLLMConfig):
    """Print configuration details."""
    print("Configuration details:")
    print(f"  target_sparsity: {config.target_sparsity}")
    print(f"  tolerance: {config.tolerance}")
    if hasattr(config, 'compression_ratio'):
        print(f"  compression_ratio: {config.compression_ratio}")
    if config.layer_selection:
        print(f"  layer_selection: {config.layer_selection}")
    print(f"  cal_nsamples: {config.cal_nsamples}")
    print()


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

    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    test_prompt = "Hello, I am"

    # Load model once for initial testing
    model, tokenizer = load_model(model_name)

    # Test generation before compression
    print("Testing generation before compression...")
    original_text = test_generation(model, tokenizer, test_prompt)

    # Define parameterized examples
    examples = [
        {
            "name": "Auto Configuration",
            "description": "Using auto_from_model() with balanced priority and medium compression",
            "config_fn": lambda m: FlatLLMConfig.auto_from_model(
                m, priority="balanced", compression="medium", cal_dataset="c4"
            ),
            "kwargs": {},
            "test_generation": True,
        },
        {
            "name": "Manual Configuration",
            "description": "Using explicit parameters with layer selection",
            "config_fn": lambda m: FlatLLMConfig(
                target_sparsity=0.7,  # Keep 70% = 30% reduction
                tolerance=0.96,        # Keep 96% variance
                layer_selection=[0, 3, 6, 9],  # Explicit layer indices
                cal_nsamples=8,
                verbose=True
            ),
            "kwargs": {},
            "test_generation": False,
        },
        {
            "name": "Override Config with Kwargs",
            "description": "Using default config but overriding specific parameters via kwargs",
            "config_fn": lambda m: None,  # Use default config
            "kwargs": {
                "target_sparsity": 0.8,  # Override: less compression
                "cal_nsamples": 4,       # Override: fewer samples (faster)
                "verbose": True
            },
            "test_generation": False,
        },
    ]

    # Run all examples in a loop
    for i, example in enumerate(examples, 1):
        print("=" * 80)
        print(f"Example {i}: {example['name']}")
        print("=" * 80)
        print(f"{example['description']}")
        print()

        # Reload model for each example (to ensure clean state)
        model, tokenizer = load_model(model_name)

        # Create config
        config = example['config_fn'](model)

        # Print config details if available
        if config is not None:
            print_config_details(config)
        else:
            print("Using default config with overrides:")
            for key, value in example['kwargs'].items():
                print(f"  {key}: {value}")
            print()

        # Compress model
        print("Compressing model...")
        compressed_model = FlatLLM.reassemble(
            model, tokenizer, config, **example['kwargs']
        )

        print(f"Compression complete!")
        print(f"  Parameters: {sum(p.numel() for p in compressed_model.parameters()):,}")
        print()

        # Test generation if requested
        if example['test_generation']:
            print("Testing generation after compression...")
            compressed_text = test_generation(compressed_model, tokenizer, test_prompt)
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

