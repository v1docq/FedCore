#!/usr/bin/env python3
"""
Auto-configuration test for TransMLAConfig with Qwen2.5-0.5B

This script focuses on verifying that TransMLAConfig.auto_from_model() correctly
selects parameters across different priorities and compression levels.

What it does:
- Loads a small public model (Qwen/Qwen2.5-0.5B) and tokenizer
- Generates auto-configs for several (priority, compression, hardware) presets
- Prints/validates key parameters (qk_mqa_dim, kv_lora_rank, freqfold/collapse, calib params)
- Optionally runs a lightweight conversion smoke test if --run-conversion is provided

Usage:
    python examples/tmp/qwen_transmla_auto_config_demo.py \
        [--priority balanced] [--compression medium] [--hardware-budget auto] \
        [--list-modes] [--run-conversion]

Requirements:
- transformers>=4.40.0
- torch>=2.0.0
- TransMLA dependencies (see external/transmlacore/)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from fedcore.algorithm.low_rank.reassembly.transmla_reassembler import (
    TransMLA, TransMLAConfig, get_transmla_status,
)


def load_qwen_model(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "auto"):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")

    print(f"Loading model: {model_name}")
    print("WARNING: Models and datasets will be cached into ~/.cache/huggingface/")
    print("          For clean use: python cleanup_disk_space.py\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print(f"Model loaded: {model.config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {next(model.parameters()).device}")
    return model, tokenizer


def validate_auto_config(model, cfg: TransMLAConfig) -> None:
    """
    Validate key constraints implied by TransMLA and our auto calculation.
    
    NOTE: This function is now a wrapper around the built-in validate() method
    in TransMLAConfig. The validation logic has been moved into the reassembler
    to ensure all configurations are validated before conversion.
    """
    cfg.validate(model)


def print_config(cfg: TransMLAConfig, title: str):
    print(f"\n{title}")
    print("=" * len(title))
    print(f"  qk_mqa_dim:        {cfg.qk_mqa_dim}")
    print(f"  kv_lora_rank:      {cfg.kv_lora_rank}")
    print(f"  freqfold:          {cfg.freqfold}")
    print(f"  collapse:          {cfg.collapse}")
    print(f"  cal_nsamples:      {cfg.cal_nsamples}")
    print(f"  cal_batch_size:    {cfg.cal_batch_size}")
    print(f"  cal_max_seqlen:    {cfg.cal_max_seqlen}")
    print(f"  ppl_eval_batch:    {cfg.ppl_eval_batch_size}")
    print(f"  dtype:             {cfg.dtype}")
    print(f"  deepseek_style:    {cfg.deepseek_style}")


def get_model_size_mb(model):
    """Calculate model size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = total_bytes / (1024 * 1024)
    return total_params, size_mb


def smoke_test_conversion(model, tokenizer, cfg: TransMLAConfig):
    """Optionally execute a quick conversion to ensure the config works end-to-end.
    This will take time due to calibration; intended as a smoke test only.
    """
    status = get_transmla_status()
    if not status.get('available', False):
        print(f"[Skip] TransMLA core not available: {status.get('error')}")
        return None

    print("\nRunning a lightweight conversion smoke test...")
    print("Note: This can take several minutes depending on hardware and dataset downloads.")

    # Show original model size
    orig_params, orig_size_mb = get_model_size_mb(model)
    print(f"\n{'='*60}")
    print(f"ORIGINAL MODEL:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size_mb:.2f} MB")
    print(f"{'='*60}")

    try:
        # Use the TransMLA high-level reassembler API (deferred execution), then execute.
        print("\nRunning TransMLA conversion...")
        converted = TransMLA.reassemble(model=model, tokenizer=tokenizer, config=cfg)
        print("Conversion completed successfully.")
        
        # Show converted model size
        conv_params, conv_size_mb = get_model_size_mb(converted)
        compression_ratio = (orig_size_mb / conv_size_mb) if conv_size_mb > 0 else 0
        param_reduction = ((orig_params - conv_params) / orig_params * 100) if orig_params > 0 else 0
        size_reduction = ((orig_size_mb - conv_size_mb) / orig_size_mb * 100) if orig_size_mb > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"REASSEMBLED MODEL:")
        print(f"  Parameters: {conv_params:,}")
        print(f"  Size: {conv_size_mb:.2f} MB")
        print(f"\nCOMPRESSION STATISTICS:")
        print(f"  Parameter reduction: {param_reduction:.2f}%")
        print(f"  Size reduction: {size_reduction:.2f}%")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"{'='*60}")
        
        return converted
    except Exception as e:
        print(f"Smoke test failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="TransMLAConfig auto-configuration test")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HF model id to use for the demo")
    parser.add_argument('--device', type=str, default='auto', help="Device mapping for HF load")

    parser.add_argument('--priority', type=str, default='balanced',
                        choices=['quality', 'speed', 'memory', 'balanced'],
                        help="Priority for the primary auto-config run")
    parser.add_argument('--compression', type=str, default='medium',
                        choices=['light', 'medium', 'aggressive'],
                        help="Compression level for the primary auto-config run")
    parser.add_argument('--hardware-budget', type=str, default='auto',
                        choices=['auto', 'low', 'high'],
                        help="Hardware budget assumption for the primary run")

    parser.add_argument('--list-modes', action='store_true',
                        help="Also show several preset modes for comparison")
    parser.add_argument('--run-conversion', action='store_true',
                        help="Run a lightweight conversion smoke test with the primary auto-config")

    args = parser.parse_args()

    model, tokenizer = load_qwen_model(args.model, device=args.device)

    # Primary auto-config
    print("\nGenerating primary auto-configuration...")
    cfg = TransMLAConfig.auto_from_model(
        model,
        priority=args.priority,
        compression=args.compression,
        hardware_budget=args.hardware_budget,
    )

    validate_auto_config(model, cfg)
    print_config(cfg, f"Auto-generated Config (priority={args.priority}, compression={args.compression}, hw={args.hardware_budget})")

    if args.list_modes:
        presets = [
            ("quality", "light", "auto"),
            ("speed", "medium", "auto"),
            ("memory", "aggressive", "low"),
            ("balanced", "medium", "high"),
        ]
        for prio, comp, hw in presets:
            try:
                cfg2 = TransMLAConfig.auto_from_model(model, priority=prio, compression=comp, hardware_budget=hw)
                validate_auto_config(model, cfg2)
                print_config(cfg2, f"Preset Config (priority={prio}, compression={comp}, hw={hw})")
            except Exception as e:
                print(f"Failed preset {prio}/{comp}/{hw}: {e}")

    if args.run_conversion:
        smoke_test_conversion(model, tokenizer, cfg)


if __name__ == '__main__':
    main()
