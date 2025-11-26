"""
Template for running FedCore experiments with MLflow tracking.

This script demonstrates how to:
1. Connect to MLflow server
2. Create/use experiments
3. Log parameters, metrics, and models
4. Use FedCore reassemblers (TransMLA, FlatLLM)
"""

import os
import torch
import mlflow
from mlflow.tracking import MlflowClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedcore.algorithm.low_rank.reassembly import (
    TransMLA, TransMLAConfig,
    FlatLLM, FlatLLMConfig,
    get_transmla_status, get_flatllm_status
)


def setup_mlflow(experiment_name: str):
    """
    Setup MLflow connection and experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., "TransMLA-TinyLlama")
    
    Returns:
        experiment_id: ID of the experiment
    """
    # Connect to MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    # Get or create experiment
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Creating experiment: {experiment_name}")
        exp_id = client.create_experiment(
            experiment_name,
            artifact_location=f"s3://mlflow/experiments/{experiment_name}"
        )
    else:
        exp_id = exp.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {exp_id})")
    
    mlflow.set_experiment(experiment_id=exp_id)
    mlflow.enable_system_metrics_logging()  # Log GPU/CPU/RAM usage
    
    return exp_id


def log_model_info(model, prefix="original"):
    """Log model information to MLflow."""
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mlflow.log_param(f"{prefix}_num_parameters", num_params)
    mlflow.log_param(f"{prefix}_trainable_parameters", trainable_params)
    mlflow.log_param(f"{prefix}_model_type", model.config.model_type)
    mlflow.log_param(f"{prefix}_hidden_size", model.config.hidden_size)
    mlflow.log_param(f"{prefix}_num_layers", model.config.num_hidden_layers)
    
    print(f"[{prefix}] Parameters: {num_params:,}")
    return num_params


def test_generation(model, tokenizer, prompt="Hello, I am"):
    """Test model generation and log results."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=50, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Deterministic for reproducibility
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def run_transmla_experiment(
    model_name: str,
    priority: str = "balanced",
    compression: str = "medium",
    run_name: str = None
):
    """
    Run TransMLA compression experiment.
    
    Args:
        model_name: HuggingFace model name
        priority: "quality", "speed", "memory", or "balanced"
        compression: "light", "medium", or "aggressive"
        run_name: Optional custom run name
    """
    experiment_name = f"TransMLA-{model_name.split('/')[-1]}"
    setup_mlflow(experiment_name)
    
    if run_name is None:
        run_name = f"{priority}-{compression}"
    
    with mlflow.start_run(run_name=run_name):
        print("=" * 80)
        print(f"TransMLA Experiment: {run_name}")
        print("=" * 80)
        
        # Log experiment config
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("reassembler", "TransMLA")
        mlflow.log_param("priority", priority)
        mlflow.log_param("compression", compression)
        mlflow.log_param("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Check TransMLA status
        status = get_transmla_status()
        mlflow.log_param("transmla_available", status['available'])
        if not status['available']:
            print(f"TransMLA not available: {status['error']}")
            mlflow.log_param("status", "failed")
            return
        
        # Load model
        print(f"\nLoading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Log original model info
        original_params = log_model_info(model, "original")
        
        # Test generation before compression
        print("\nTesting generation before compression...")
        original_text = test_generation(model, tokenizer)
        mlflow.log_text(original_text, "generation_original.txt")
        
        # Create auto config
        print(f"\nCreating TransMLA config (priority={priority}, compression={compression})...")
        config = TransMLAConfig.auto_from_model(
            model,
            priority=priority,
            compression=compression,
            hardware_budget="auto"
        )
        
        # Log config
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            mlflow.log_param(f"config_{key}", value)
        
        # Compress model
        print("\nCompressing model with TransMLA...")
        import time
        start_time = time.time()
        
        compressed_model = TransMLA.reassemble(model, tokenizer, config)
        
        compression_time = time.time() - start_time
        mlflow.log_metric("compression_time_sec", compression_time)
        
        # Log compressed model info
        compressed_params = log_model_info(compressed_model, "compressed")
        
        # Calculate compression stats
        compression_ratio = (original_params - compressed_params) / original_params
        mlflow.log_metric("compression_ratio", compression_ratio)
        mlflow.log_metric("param_reduction_percent", compression_ratio * 100)
        
        print(f"\nCompression ratio: {compression_ratio:.2%}")
        print(f"Time taken: {compression_time:.2f}s")
        
        # Test generation after compression
        print("\nTesting generation after compression...")
        compressed_text = test_generation(compressed_model, tokenizer)
        mlflow.log_text(compressed_text, "generation_compressed.txt")
        
        # Log model to MLflow (optional - can be large!)
        # mlflow.transformers.log_model(compressed_model, "model")
        
        mlflow.log_param("status", "success")
        print("\n✅ Experiment completed successfully!")


def run_flatllm_experiment(
    model_name: str,
    target_sparsity: float = 0.7,
    cal_dataset: str = "wikitext2",
    run_name: str = None
):
    """
    Run FLAT-LLM compression experiment.
    
    Args:
        model_name: HuggingFace model name
        target_sparsity: Target sparsity (0.7 = keep 70%)
        cal_dataset: Calibration dataset ("wikitext2", "c4", "ptb")
        run_name: Optional custom run name
    """
    experiment_name = f"FlatLLM-{model_name.split('/')[-1]}"
    setup_mlflow(experiment_name)
    
    if run_name is None:
        run_name = f"sparsity-{target_sparsity}-{cal_dataset}"
    
    with mlflow.start_run(run_name=run_name):
        print("=" * 80)
        print(f"FLAT-LLM Experiment: {run_name}")
        print("=" * 80)
        
        # Log experiment config
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("reassembler", "FLAT-LLM")
        mlflow.log_param("target_sparsity", target_sparsity)
        mlflow.log_param("cal_dataset", cal_dataset)
        
        # Check FLAT-LLM status
        status = get_flatllm_status()
        mlflow.log_param("flatllm_available", status['available'])
        if not status['available']:
            print(f"FLAT-LLM not available: {status['error']}")
            mlflow.log_param("status", "failed")
            return
        
        # Load model
        print(f"\nLoading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Log original model info
        original_params = log_model_info(model, "original")
        
        # Test generation before compression
        print("\nTesting generation before compression...")
        original_text = test_generation(model, tokenizer)
        mlflow.log_text(original_text, "generation_original.txt")
        
        # Create config
        print(f"\nCreating FLAT-LLM config...")
        config = FlatLLMConfig(
            target_sparsity=target_sparsity,
            tolerance=0.96,
            cal_dataset=cal_dataset,
            cal_nsamples=16,
            layer_selection="auto",
            compression_ratio=0.3,
            verbose=True
        )
        
        # Log config
        mlflow.log_param("config_tolerance", config.tolerance)
        mlflow.log_param("config_cal_nsamples", config.cal_nsamples)
        mlflow.log_param("config_layer_selection", str(config.layer_selection))
        
        # Compress model
        print("\nCompressing model with FLAT-LLM...")
        import time
        start_time = time.time()
        
        compressed_model = FlatLLM.reassemble(model, tokenizer, config)
        
        compression_time = time.time() - start_time
        mlflow.log_metric("compression_time_sec", compression_time)
        
        # Log compressed model info
        compressed_params = log_model_info(compressed_model, "compressed")
        
        # Calculate compression stats
        compression_ratio = (original_params - compressed_params) / original_params
        mlflow.log_metric("compression_ratio", compression_ratio)
        mlflow.log_metric("param_reduction_percent", compression_ratio * 100)
        
        print(f"\nCompression ratio: {compression_ratio:.2%}")
        print(f"Time taken: {compression_time:.2f}s")
        
        # Test generation after compression
        print("\nTesting generation after compression...")
        compressed_text = test_generation(compressed_model, tokenizer)
        mlflow.log_text(compressed_text, "generation_compressed.txt")
        
        mlflow.log_param("status", "success")
        print("\n✅ Experiment completed successfully!")


if __name__ == "__main__":
    # Example: Run TransMLA experiment
    print("Starting TransMLA experiment...")
    run_transmla_experiment(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        priority="balanced",
        compression="medium",
        run_name="test-run"
    )
    
    # Example: Run FLAT-LLM experiment
    # print("\nStarting FLAT-LLM experiment...")
    # run_flatllm_experiment(
    #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     target_sparsity=0.7,
    #     cal_dataset="wikitext2",
    #     run_name="test-run"
    # )


