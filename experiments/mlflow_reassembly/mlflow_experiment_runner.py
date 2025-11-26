"""
ExperimentRunner for running experiments with LowRank + Reassembly + Training.

Experiment pipeline:
1. Load model
2. Apply LowRank SVD decomposition
3. Apply Reassembly (FlatLLM or TransMLA)
4. Train the model
5. Apply Rank Pruning (one_time at the end)
6. Log all metrics to MLflow
"""

# Configure matplotlib backend before imports
import os
os.environ['MPLBACKEND'] = 'Agg'  # Non-GUI backend
import time
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_experiments_config import ExperimentConfig
from gpu_monitor import GPUMonitor

# FedCore imports
try:
    # Import via __init__.py (proper way)
    import fedcore.algorithm.low_rank.reassembly as reassembly_module
    
    TransMLA = reassembly_module.TransMLA
    TransMLAConfig = reassembly_module.TransMLAConfig
    get_transmla_status = reassembly_module.get_transmla_status
    
    FlatLLM = reassembly_module.FlatLLM
    FlatLLMConfig = reassembly_module.FlatLLMConfig
    get_flatllm_status = reassembly_module.get_flatllm_status
    
    # Import decompose_module via svd_tools (work around problematic low_rank_opt.py)
    try:
        from fedcore.algorithm.low_rank.svd_tools import decompose_module
    except ImportError:
        from fedcore.algorithm.low_rank.low_rank_opt import decompose_module
    
    from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    
    FEDCORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FedCore components import failed: {e}")
    print("Some features may not work. Install missing dependencies:")
    print("  pip install fedot==0.7.5 golem-core")
    FEDCORE_AVAILABLE = False
    
    # Fallback: create stubs
    class IDecomposed:
        pass
    
    def decompose_module(*args, **kwargs):
        print("Warning: decompose_module not available (FedCore import failed)")
        pass
    
    def rank_threshold_pruning(*args, **kwargs):
        print("Warning: rank_threshold_pruning not available (FedCore import failed)")
        pass
    
    class TransMLA:
        @classmethod
        def reassemble(cls, *args, **kwargs):
            raise NotImplementedError("TransMLA not available")
    
    class FlatLLM:
        @classmethod
        def reassemble(cls, *args, **kwargs):
            raise NotImplementedError("FlatLLM not available")
    
    def get_transmla_status():
        return {'available': False, 'error': 'Import failed'}
    
    def get_flatllm_status():
        return {'available': False, 'error': 'Import failed'}
    
    TransMLAConfig = None
    FlatLLMConfig = None


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_perplexity: float
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    epoch_time: float = 0.0
    learning_rate: float = 0.0


class ExperimentRunner:
    """
    Class to run a single experiment with the full pipeline.

    Pipeline:
        1. Load model and data
        2. Apply LowRank SVD decomposition
        3. Apply Reassembly (FlatLLM/TransMLA)
        4. Configure optimizer
        5. Train (with metrics logging)
        6. Apply Rank Pruning at the end
        7. Save results
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # CRITICAL: Clear GPU memory WHEN CREATING the runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Компоненты, которые будут созданы
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None
        
        # Monitoring
        self.gpu_monitor = GPUMonitor(device_id=0, interval=5.0) if torch.cuda.is_available() else None
        
        # Metrics
        self.epoch_metrics: List[EpochMetrics] = []
        
        # MLflow
        self.mlflow_run_id: Optional[str] = None
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_mlflow(self):
        """Sets up MLflow tracking."""
        print("Setting up MLflow...")
        
        # Configure MinIO credentials
        os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
        
        # Connect to MLflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.enable_system_metrics_logging()
        
        # Create or get experiment
        client = MlflowClient()
        exp = client.get_experiment_by_name(self.config.mlflow_experiment_name)
        
        if exp is None:
            print(f"  Creating experiment: {self.config.mlflow_experiment_name}")
            exp_id = client.create_experiment(
                self.config.mlflow_experiment_name,
                artifact_location=f"s3://mlflow/experiments/{self.config.mlflow_experiment_name}"
            )
        else:
            exp_id = exp.experiment_id
            print(f"  Using experiment: {self.config.mlflow_experiment_name} (ID: {exp_id})")
        
        # Start run
        mlflow.set_experiment(experiment_id=exp_id)
        mlflow.start_run(run_name=self.config.experiment_name)
        self.mlflow_run_id = mlflow.active_run().info.run_id
        
        # Log configuration
        config_dict = self.config.to_dict()
        self._log_nested_params(config_dict, prefix="")
        
        print(f"  Run ID: {self.mlflow_run_id}")
        print()
    
    def _log_nested_params(self, params_dict: Dict, prefix: str = ""):
        """Recursively logs nested parameters to MLflow."""
        for key, value in params_dict.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                self._log_nested_params(value, prefix=f"{full_key}.")
            elif isinstance(value, (list, tuple)):
                mlflow.log_param(full_key, str(value))
            else:
                mlflow.log_param(full_key, value)
    
    def load_model(self):
        """Loads the model and tokenizer."""
        print(f"Loading model: {self.config.model.model_id}")
        
        if self.config.model.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.config.model.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        mlflow.log_metric("original_num_parameters", num_params)
        mlflow.log_metric("original_trainable_parameters", trainable_params)
        
        print(f"  Model loaded: {self.config.model.name}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print()
    
    def apply_lowrank_decomposition(self):
        """
        Applies LowRank SVD decomposition to the model.

        This step prepares the model for rank pruning after training.
        """
        if not FEDCORE_AVAILABLE:
            print("Skipping LowRank decomposition (FedCore not fully available)")
            mlflow.log_param("lowrank_applied", False)
            print()
            return
        
        print("Applying LowRank SVD decomposition...")
        
        # NaN FIX: convert to fp32 for stability
        original_dtype = next(self.model.parameters()).dtype
        if original_dtype == torch.float16:
            print("  Converting model from fp16 to fp32 for training stability...")
            self.model = self.model.float()
            mlflow.log_param("converted_to_fp32", True)
        
        try:
            # Apply decomposition
            decompose_module(
                self.model,
                decomposing_mode=self.config.lowrank.decomposing_mode,
                decomposer=self.config.lowrank.decomposer,
                compose_mode=None  # Do not compose immediately
            )
            
            # Count decomposed layers
            decomposed_count = sum(1 for m in self.model.modules() if isinstance(m, IDecomposed))
            
            mlflow.log_metric("decomposed_layers_count", decomposed_count)
            mlflow.log_param("lowrank_applied", True)
            
            print(f"  Decomposed {decomposed_count} layers")
            print(f"  Strategy: {self.config.lowrank.strategy}")
            print(f"  Threshold: {self.config.lowrank.threshold}")
            print(f"  Pruning mode: {'one_time' if self.config.lowrank.rank_prune_each == -1 else f'every {self.config.lowrank.rank_prune_each} epochs'}")
            print()
            
            # CRITICAL: Memory cleanup after LowRank
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("  Memory cleaned after LowRank")
        except Exception as e:
            print(f"  Warning: LowRank decomposition failed: {e}")
            print(f"  Continuing without LowRank...")
            mlflow.log_param("lowrank_applied", False)
            mlflow.log_param("lowrank_error", str(e))
            print()
    
    def _compose_weights_for_inference(self):
        """
        NO LONGER USED!

        Previously, composing weights after LowRank was required before applying
        FlatLLM/TransMLA to avoid ParameterPlaceHolder errors.

        The order is now changed:
        1. FlatLLM/TransMLA (operate on regular weights)
        2. LowRank (decomposes already compressed weights)

        Composition is no longer required!
        """
        pass  # Не используется
    
    def apply_compression(self):
        """Applies FlatLLM or TransMLA compression."""
        method = self.config.compression.method.lower()
        
        print(f"Applying {method.upper()} compression...")
        
        if method == "flatllm":
            self._apply_flatllm()
        elif method == "transmla":
            self._apply_transmla()
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        # Log parameters after compression
        compressed_params = sum(p.numel() for p in self.model.parameters())
        mlflow.log_metric("after_compression_num_parameters", compressed_params)
        
        print(f"  Parameters after compression: {compressed_params:,}")
        print()
    
    def _apply_flatllm(self):
        """Applies FlatLLM compression."""
        status = get_flatllm_status()
        if not status['available']:
            print(f"  FlatLLM not available: {status['error']}")
            print("  Skipping compression")
            return
        
        # Create configuration
        config_params = self.config.compression.config_params.copy()
        config_params['device'] = self.device
        config = FlatLLMConfig(**config_params)
        
        # Apply compression
        self.model = FlatLLM.reassemble(self.model, self.tokenizer, config)
        
        # CRITICAL: Multiple memory cleanups after FlatLLM
        # AbsorptionCompressor accumulates huge activations in memory
        print("  Cleaning memory after FlatLLM (multiple passes)...")
        for i in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
        print(f"  Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    def _apply_transmla(self):
        """Applies TransMLA compression."""
        status = get_transmla_status()
        if not status['available']:
            print(f"  TransMLA not available: {status['error']}")
            print("  Skipping compression")
            return
        
        # Create configuration
        config_params = self.config.compression.config_params.copy()
        
        # TransMLA uses auto_from_model
        if 'priority' in config_params and 'compression' in config_params:
            config = TransMLAConfig.auto_from_model(
                self.model,
                priority=config_params.get('priority', 'balanced'),
                compression=config_params.get('compression', 'medium'),
                hardware_budget=config_params.get('hardware_budget', 'auto')
            )
        else:
            config = TransMLAConfig(**config_params)
        
        # Apply compression
        self.model = TransMLA.reassemble(self.model, self.tokenizer, config)
        
        # CRITICAL: Memory cleanup after TransMLA
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("  Memory cleaned after TransMLA")
    
    def setup_optimizer(self):
        """Configures the optimizer and scheduler."""
        print(f"Setting up optimizer: {self.config.optimizer.name}")
        
        # Create optimizer
        if self.config.optimizer.is_model_based:
            # ULTG: accepts model instead of parameters
            self.optimizer = self.config.optimizer.optimizer_class(
                model=self.model,
                svd_type=self.config.optimizer.ultg_svd_type,
                rank=self.config.optimizer.ultg_rank,
                learning_rate=self.config.optimizer.learning_rate,
                training_samples=self.config.optimizer.ultg_training_samples,
                batch_size=self.config.optimizer.ultg_batch_size,
                scale=self.config.optimizer.ultg_scale,
                second_scale=self.config.optimizer.ultg_second_scale,
                second_rank=self.config.optimizer.ultg_second_rank
            )
        else:
            # Adam: accept parameters
            # NaN FIX: split params - no weight decay for LayerNorm/bias
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'layernorm' in name.lower() or 'bias' in name.lower():
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
            
            param_groups = [
                {'params': decay_params, 'weight_decay': self.config.optimizer.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            
            print(f"  Parameters with decay: {len(decay_params)}")
            print(f"  Parameters without decay: {len(no_decay_params)}")
            
            self.optimizer = self.config.optimizer.optimizer_class(
                param_groups,
                lr=self.config.optimizer.learning_rate,
                betas=self.config.optimizer.betas
            )
        
        # Approximate number of steps (will be refined after data loading)
        num_training_steps = self.config.training.epochs * 1000
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"  Optimizer: {self.config.optimizer.name}")
        print(f"  Learning rate: {self.config.optimizer.learning_rate}")
        print(f"  Weight decay: {self.config.optimizer.weight_decay}")
        print()
    
    def prepare_data(self, dataset_name: str = "openwebtext", num_samples: int = 500):
        """
        Prepares training and evaluation dataloaders.

        Args:
            dataset_name: Dataset name. Options:
                - "openwebtext" - OpenWebText (default, ~8M examples)
                - "simple_wikipedia" - Simple Wikipedia (~226k examples, ~65M tokens)
                - "wikitext" - WikiText-2-raw-v1
                - any other name - will be loaded via load_dataset()
            num_samples: Number of samples for training (default 500)
        """
        print(f"Preparing dataset: {dataset_name}")
        
        # Load dataset
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        elif dataset_name == "openwebtext":
            dataset = load_dataset("Skylion007/openwebtext")
        elif dataset_name == "simple_wikipedia":
            # simple_wikipedia loads as DatasetDict
            dataset = load_dataset("pszemraj/simple_wikipedia")
        else:
            dataset = load_dataset(dataset_name)
        
        # Tokenization function
        def tokenize_function(examples):
            # Extract text (auto-detect key)
            if 'text' in examples:
                texts = examples['text']
            elif 'content' in examples:
                texts = examples['content']
            else:
                # Take the first string field
                for key, value in examples.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                        texts = value
                        break
            
            # Filter out empty texts
            texts = [t for t in texts if len(t.strip()) > 10]
            
            if not texts:
                return {'input_ids': [], 'attention_mask': [], 'labels': []}
            
            # Tokenization
            result = self.tokenizer(
                texts,
                max_length=self.config.model.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            # Labels = input_ids for language modeling
            result['labels'] = result['input_ids'].copy()
            
            return result
        
        # Determine whether it's a Dataset or DatasetDict
        if hasattr(dataset, 'keys') and callable(dataset.keys):  # DatasetDict
            dataset_train = dataset['train']
            if 'validation' in dataset:
                dataset_eval = dataset['validation']
            elif 'test' in dataset:
                dataset_eval = dataset['test']
            else:
                dataset_eval = dataset['train']  # Fallback
        else:  # Dataset
            dataset_train = dataset
            # For Dataset create eval from the same set (last 10%)
            split_idx = int(len(dataset) * 0.9)
            dataset_eval = dataset.select(range(split_idx, len(dataset)))
        
        # Tokenize datasets
        tokenized_train = dataset_train.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_train.column_names
        )
        
        tokenized_eval = dataset_eval.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_eval.column_names
        )
        
        # Take a subset for a quick test
        if num_samples:
            tokenized_train = tokenized_train.select(range(min(num_samples, len(tokenized_train))))
            tokenized_eval = tokenized_eval.select(range(min(num_samples // 2, len(tokenized_eval))))
        
        # Create DataLoaders
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        self.train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        
        self.eval_dataloader = DataLoader(
            tokenized_eval,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        print(f"  Train samples: {len(tokenized_train)}")
        print(f"  Eval samples: {len(tokenized_eval)}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print()
    
    def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        Trains the model for one epoch.

        Returns:
            EpochMetrics with epoch results
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        for step, batch in enumerate(self.train_dataloader):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # NaN check
            if torch.isnan(loss):
                print(f"⚠️  NaN loss detected at step {step}, skipping...")
                continue
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update weights
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Progress logging
            if (step + 1) % self.config.training.logging_steps == 0:
                avg_loss = total_loss / num_batches
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}, Step {step + 1}/{len(self.train_dataloader)}: "
                      f"loss={avg_loss:.4f}, lr={current_lr:.2e}")
                
                # Log to MLflow
                mlflow.log_metric("train_loss_step", avg_loss, step=epoch * len(self.train_dataloader) + step)
                mlflow.log_metric("learning_rate", current_lr, step=epoch * len(self.train_dataloader) + step)
        
        # Средние метрики за эпоху
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        current_lr = self.scheduler.get_last_lr()[0]
        
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            train_perplexity=train_perplexity,
            epoch_time=epoch_time,
            learning_rate=current_lr
        )
        
        return metrics
    
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluates the model on the eval dataset.

        Returns:
            Dict with eval metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_eval_loss = total_loss / num_batches if num_batches > 0 else 0.0
        eval_perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()
        
        return {
            'eval_loss': avg_eval_loss,
            'eval_perplexity': eval_perplexity
        }
    
    def apply_rank_pruning(self):
        """
        Applies rank pruning to decomposed layers (one_time at the end).

        Uses strategy and threshold from LowRankConfig.
        """
        if not FEDCORE_AVAILABLE:
            print("Skipping rank pruning (FedCore not fully available)")
            mlflow.log_param("pruning_applied", False)
            print()
            return
        
        print("Applying rank pruning (one_time)...")
        
        try:
            pruned_count = 0
            params_before = sum(p.numel() for p in self.model.parameters())
            
            for name, module in self.model.named_modules():
                if isinstance(module, IDecomposed):
                    rank_threshold_pruning(
                        decomposed_module=module,
                        threshold=self.config.lowrank.threshold,
                        strategy=self.config.lowrank.strategy,
                        module_name=name
                    )
                    # After pruning set compose_mode for inference
                    # (as in FedCore's OnetimeRankPruner)
                    module.compose_weight_for_inference()
                    pruned_count += 1
            
            params_after = sum(p.numel() for p in self.model.parameters())
            reduction = (params_before - params_after) / params_before * 100
            
            mlflow.log_metric("pruned_layers_count", pruned_count)
            mlflow.log_metric("after_pruning_num_parameters", params_after)
            mlflow.log_metric("pruning_reduction_percent", reduction)
            mlflow.log_param("pruning_applied", True)
            
            print(f"  Pruned {pruned_count} layers")
            print(f"  Parameters: {params_before:,} → {params_after:,} ({reduction:.2f}% reduction)")
            print()
        except Exception as e:
            print(f"  Warning: Rank pruning failed: {e}")
            print(f"  Continuing without pruning...")
            mlflow.log_param("pruning_applied", False)
            mlflow.log_param("pruning_error", str(e))
            print()
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.training.epochs} epochs...")
        print()
        
        # Start GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.start()
        
        for epoch in range(1, self.config.training.epochs + 1):
            print(f"{'='*70}")
            print(f"Epoch {epoch}/{self.config.training.epochs}")
            print(f"{'='*70}")
            
            # Training
            epoch_metrics = self.train_epoch(epoch)
            
            # Evaluation
            eval_metrics = self.evaluate(epoch)
            epoch_metrics.eval_loss = eval_metrics['eval_loss']
            epoch_metrics.eval_perplexity = eval_metrics['eval_perplexity']
            
            # Save metrics
            self.epoch_metrics.append(epoch_metrics)
            
            # Log to MLflow
            mlflow.log_metric("epoch_train_loss", epoch_metrics.train_loss, step=epoch)
            mlflow.log_metric("epoch_train_perplexity", epoch_metrics.train_perplexity, step=epoch)
            mlflow.log_metric("epoch_eval_loss", epoch_metrics.eval_loss, step=epoch)
            mlflow.log_metric("epoch_eval_perplexity", epoch_metrics.eval_perplexity, step=epoch)
            mlflow.log_metric("epoch_time_seconds", epoch_metrics.epoch_time, step=epoch)
            mlflow.log_metric("learning_rate_epoch", epoch_metrics.learning_rate, step=epoch)
            
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {epoch_metrics.train_loss:.4f}, Perplexity: {epoch_metrics.train_perplexity:.2f}")
            print(f"  Eval Loss: {epoch_metrics.eval_loss:.4f}, Perplexity: {epoch_metrics.eval_perplexity:.2f}")
            print(f"  Epoch Time: {epoch_metrics.epoch_time:.1f}s")
            print(f"  Learning Rate: {epoch_metrics.learning_rate:.2e}")
            print()
        
        # Stop GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop()
            
            # Log GPU statistics
            gpu_stats = self.gpu_monitor.get_summary_stats()
            for key, value in gpu_stats.items():
                mlflow.log_metric(key, value)
            
            print("GPU Statistics:")
            for key, value in gpu_stats.items():
                print(f"  {key}: {value:.2f}")
            print()
    
    def save_results(self):
        """Saves experiment results."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        metrics_data = {
            'experiment_name': self.config.experiment_name,
            'epochs': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'train_perplexity': m.train_perplexity,
                    'eval_loss': m.eval_loss,
                    'eval_perplexity': m.eval_perplexity,
                    'epoch_time': m.epoch_time,
                    'learning_rate': m.learning_rate
                }
                for m in self.epoch_metrics
            ]
        }
        
        if self.gpu_monitor:
            metrics_data['gpu_stats'] = self.gpu_monitor.get_summary_stats()
        
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Log file as artifact
        mlflow.log_artifact(str(metrics_file))
        
        print(f"Results saved to: {metrics_file}")
    
    def run(self):
        """Runs the full experiment pipeline."""
        print("=" * 80)
        print(f"EXPERIMENT: {self.config.experiment_name}")
        print("=" * 80)
        print()
        
        try:
            # Setup
            self.setup_mlflow()
            self.load_model()
            
            # ORDER CHANGED:
            # 1. Apply reassembly first (FlatLLM/TransMLA)
            #    - Operate on regular weights (weight parameter exists)
            #    - Physically compress the model
            self.apply_compression()
            
            # 2. Then apply LowRank decomposition
            #    - Decomposes ALREADY COMPRESSED weights
            #    - weight → U @ S @ Vh
            #    - No longer necessary to compose back!
            self.apply_lowrank_decomposition()
            
            # Setup training
            self.setup_optimizer()
            self.prepare_data(num_samples=500)  # OpenWebText: 5000 samples
            
            # CRITICAL: Aggressive memory cleanup BEFORE training
            print("🧹 Aggressive memory cleanup before training...")
            for i in range(5):  # 5 cleanup passes
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print()
            
            # Training
            self.train()
            
            # Rank pruning at the end (one_time mode)
            self.apply_rank_pruning()
            
            # Сохранение
            self.save_results()
            
            # Финальные метрики
            final_params = sum(p.numel() for p in self.model.parameters())
            mlflow.log_metric("final_num_parameters", final_params)
            
            mlflow.log_param("status", "success")
            
            print("=" * 80)
            print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print()
            
        except Exception as e:
            print(f"\n❌ EXPERIMENT FAILED: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise
        
        finally:
            # End MLflow run
            if mlflow.active_run():
                mlflow.end_run()


if __name__ == "__main__":
    # CRITICAL: Fix PyTorch memory fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Single experiment test
    from mlflow_experiments_config import generate_experiment_configs
    
    configs = generate_experiment_configs()
    
    # Run the first experiment (Adam + FlatLLM + sLM)
    print("Running first experiment as test...")
    runner = ExperimentRunner(configs[0])
    runner.run()

