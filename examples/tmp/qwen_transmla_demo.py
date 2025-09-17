#!/usr/bin/env python3
"""
Demonstration of TransMLA integration with Qwen2.5-0.5B model

This script shows how to:
1. Load Qwen2.5-0.5B model and tokenizer
2. Apply TransMLA conversion (deferred and immediate)
3. Test text generation before and after conversion
4. Save and load converted models

Requirements:
- transformers>=4.40.0
- torch>=2.0.0
- TransMLA dependencies (see external/transmla_core/)

Usage:
    python examples/qwen_transmla_demo.py [--mode deferred|immediate] [--save-path PATH]
"""

import argparse
import os
import torch
import time
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from fedcore.algorithm.reassembly.core_reassemblers import AttentionReassembler
from fedcore.algorithm.reassembly.transmla_reassembler import TransMLA, TransMLAConfig, get_transmla_status


def load_qwen_model(model_name="Qwen/Qwen2.5-0.5B", device="auto"):
    """Load Qwen2.5-0.5B model and tokenizer"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def test_text_generation(model, tokenizer, prompt="The capital of France is", max_length=50):
    """Test text generation with the model"""
    print(f"\nTesting generation with prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generated text: {generated_text}")
    print(f"Generation time: {generation_time:.2f}s")
    
    return generated_text, generation_time


def create_transmla_config():
    """Create optimized TransMLA configuration for Qwen2.5-0.5B"""
    return TransMLAConfig(
        freqfold="auto",
        collapse="auto",
        qk_mqa_dim=64,        # Match Qwen2.5-0.5B head_dim
        kv_lora_rank=128,     # Must be < 2*latent_dim - qk_mqa_dim = 2*128-64 = 192
        q_lora_rank=None,     # Use default
        balance_kv_ratio=1.0,
        use_qkv_norm=False,
        cal_dataset="wikitext2",
        cal_nsamples=32,      # Reduced for demo
        cal_batch_size=2,     # Memory efficient
        cal_max_seqlen=128,   # Shorter sequences
        ppl_eval_batch_size=1,
        deepseek_style=True,
        dtype="fp16" if torch.cuda.is_available() else "fp32",
        device="auto",
        seed=42
    )


def demo_deferred_conversion(model, tokenizer, config):
    """Demonstrate deferred TransMLA conversion"""
    print("\n" + "="*60)
    print("DEFERRED CONVERSION DEMO")
    print("="*60)
    
    # Create deferred conversion
    print("Creating deferred conversion...")
    deferred = TransMLA.reassemble(
        model=model,
        tokenizer=tokenizer,
        config=config,
        deferred=True
    )
    
    print(f"Deferred conversion created: {deferred}")
    print(f"Conversion type: {deferred.conversion_type}")
    print(f"Executed: {deferred.executed}")
    
    # Test original model first
    print("\nTesting original model:")
    test_text_generation(model, tokenizer)
    
    # Execute conversion
    print("\nExecuting deferred conversion...")
    print("Note: This may take several minutes for calibration and conversion...")
    
    try:
        converted_model = deferred.execute()
        print("Deferred conversion completed successfully!")
        
        # Test converted model
        print("\nTesting converted model:")
        test_text_generation(converted_model, tokenizer)
        
        return converted_model
        
    except Exception as e:
        print(f"Deferred conversion failed: {e}")
        return None


def demo_immediate_conversion(model, tokenizer, config):
    """Demonstrate immediate TransMLA conversion"""
    print("\n" + "="*60)
    print("IMMEDIATE CONVERSION DEMO")
    print("="*60)
    
    # Test original model
    print("Testing original model:")
    test_text_generation(model, tokenizer)
    
    # Perform immediate conversion
    print("\nPerforming immediate conversion...")
    print("Note: This may take several minutes for calibration and conversion...")
    
    try:
        converted_model = TransMLA.reassemble(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        print("Immediate conversion completed successfully!")
        
        # Test converted model
        print("\nTesting converted model:")
        test_text_generation(converted_model, tokenizer)
        
        return converted_model
        
    except Exception as e:
        print(f"Immediate conversion failed: {e}")
        return None


def demo_attention_reassembler(model, tokenizer, config, transmla_status):
    """Demonstrate AttentionReassembler usage"""
    print("\n" + "="*60)
    print("ATTENTION REASSEMBLER DEMO")
    print("="*60)
    
    # Test parental mode (should work without TransMLA)
    print("Testing AttentionReassembler in parental mode:")
    try:
        parental_result = AttentionReassembler.reassemble(model, mode='parental')
        print("Parental mode conversion successful")
    except Exception as e:
        print(f"Parental mode failed: {e}")
    
    # Test TransMLA mode
    if transmla_status['available']:
        print("\nTesting AttentionReassembler in TransMLA mode:")
        try:
            transmla_result = AttentionReassembler.reassemble(
                model=model,
                mode='trans-mla',
                tokenizer=tokenizer,
                config=config
            )
            print("TransMLA mode conversion successful")
            return transmla_result
        except Exception as e:
            print(f"TransMLA mode failed: {e}")
    else:
        print("TransMLA not available, skipping TransMLA mode test")
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B TransMLA Demo")
    parser.add_argument(
        "--mode", 
        choices=["deferred", "immediate", "reassembler", "all"],
        default="all",
        help="Conversion mode to demonstrate"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save converted model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Qwen model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    args = parser.parse_args()
    
    print("Qwen2.5-0.5B TransMLA Integration Demo")
    print("="*60)
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    transmla_status = get_transmla_status()
    print(f"TransMLA available: {transmla_status['available']}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not available. Install with: pip install transformers")
        return
    
    if not transmla_status['available']:
        print("TransMLA not available. Some features will be limited.")
        if transmla_status['error']:
            print(f"Error: {transmla_status['error']}")
    
    try:
        # Load model
        model, tokenizer = load_qwen_model(args.model_name, args.device)
        
        # Create TransMLA config
        config = create_transmla_config()
        print(f"\nTransMLA Configuration:")
        print(f"  qk_mqa_dim: {config.qk_mqa_dim}")
        print(f"  kv_lora_rank: {config.kv_lora_rank}")
        print(f"  cal_nsamples: {config.cal_nsamples}")
        print(f"  cal_batch_size: {config.cal_batch_size}")
        
        # Run demos based on mode
        if args.mode in ["deferred", "all"]:
            demo_deferred_conversion(model, tokenizer, config)
        
        if args.mode in ["immediate", "all"]:
            demo_immediate_conversion(model, tokenizer, config,)
        
        if args.mode in ["reassembler", "all"]:
            demo_attention_reassembler(model, tokenizer, config, transmla_status)
        
        print("\nDemo completed!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
