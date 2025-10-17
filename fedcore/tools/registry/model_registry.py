"""Refactored Model Registry with cleaner architecture."""

import os
import gc
import logging
from typing import Optional
from threading import Lock

import torch

from .checkpoint_manager import CheckpointManager
from .registry_storage import RegistryStorage
from .metrics_tracker import MetricsTracker


class ModelRegistry:
    """
    Thread-safe Singleton model registry for FedCore pipeline.
    
    Manages model checkpoints, metrics, and versioning across the PEFT pipeline.
    Uses composition pattern with specialized managers for different responsibilities.
    
    Components:
        - CheckpointManager: Handles checkpoint serialization/deserialization
        - RegistryStorage: Manages DataFrame persistence
        - MetricsTracker: Tracks metrics and versions
    """
    
    _instance = None
    _initialized = False
    _lock = Lock()

    def __new__(cls, auto_cleanup: bool = True):
        """Ensure only one instance exists (Thread-safe Singleton).
        
        Uses double-checked locking pattern:
        - First check without lock for performance
        - Acquire lock only if instance doesn't exist
        - Second check after acquiring lock to prevent race condition
        
        Args:
            auto_cleanup: Ignored in __new__, handled in __init__
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, auto_cleanup: bool = True):
        """Initialize registry components (only once, thread-safe).
        
        Uses the same lock as __new__ to ensure components are
        initialized exactly once even in multi-threaded environment.
        """
        if not ModelRegistry._initialized:
            with ModelRegistry._lock:
                if not ModelRegistry._initialized:
                    base_dir = os.environ.get("FEDCORE_MODEL_REGISTRY_PATH", "llm_output")
                    
                    self.checkpoint_manager = CheckpointManager(base_dir, auto_cleanup=auto_cleanup)
                    self.storage = RegistryStorage(base_dir)
                    self.metrics_tracker = MetricsTracker()
                    self.auto_cleanup = auto_cleanup
                    self.logger = logging.getLogger(self.__class__.__name__)
                    self.logger.setLevel(logging.INFO)
                    
                    ModelRegistry._initialized = True
                    self.logger.info(f"ModelRegistry initialized: auto_cleanup={auto_cleanup}, base_dir={base_dir}")
    
    
    def register_model(self, fedcore_id: str, model=None, model_path: str = None,
                      pipeline_params: dict = None, metrics: Optional[dict] = None,
                      note: str = "initial", params_format: str = 'yaml', 
                      delete_model_after_save: bool = True) -> str:
        """Register a new model in the registry.
        
        Args:
            fedcore_id: FedCore instance identifier
            model: Model object to register
            model_path: Path to model file
            pipeline_params: Pipeline parameters (currently unused, can be extended)
            metrics: Metrics dictionary
            note: Note about this registration (unused)
            params_format: Format for params (unused)
            delete_model_after_save: If True, delete model from memory after saving
            
        Returns:
            model_id: Generated model identifier
        """
        self.logger.info(f"register_model called: fedcore_id={fedcore_id}, model_type={type(model).__name__ if model else 'None'}, metrics={metrics}")
        
        model_id = self.metrics_tracker.generate_model_id(model, model_path)
        version = self.metrics_tracker.generate_version()
        safe_timestamp = self.metrics_tracker.sanitize_timestamp(version)
        
        if self.auto_cleanup and torch.cuda.is_available():
            mem_before = self.checkpoint_manager.get_gpu_memory_stats()
            self.logger.info(f"GPU memory before saving: {mem_before.get('allocated_gb', 0):.4f} GB")
        
        checkpoint_bytes = self.checkpoint_manager.serialize_to_bytes(model, model_path)
        
        if model_path and os.path.isfile(model_path):
            checkpoint_path = model_path
        else:
            checkpoint_path = self.checkpoint_manager.generate_checkpoint_path(
                fedcore_id, model_id, safe_timestamp
            )
            self.checkpoint_manager.save_to_file(checkpoint_bytes, checkpoint_path)
        
        record = {
            "record_id": str(self.metrics_tracker.generate_model_id()),
            "fedcore_id": fedcore_id,
            "model_id": model_id,
            "version": version,
            "created_at": version,
            "model_path": model_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics if metrics is not None else {},
            "pipeline_params": None,  # Can be extended later
        }
        
        self.storage.append_record(fedcore_id, record)
        
        if delete_model_after_save and model is not None:
            self.logger.info(f"Deleting model {model_id} from memory")
            self._delete_model_from_memory(model)
            self.logger.info("Model deleted from memory")
        
        if self.auto_cleanup and torch.cuda.is_available():
            mem_after = self.checkpoint_manager.get_gpu_memory_stats()
            self.logger.info(f"GPU memory after cleanup: {mem_after.get('allocated_gb', 0):.4f} GB")
        
        return model_id
    
    def register_changes(self, fedcore_id: str, model_id: str, model=None,
                        pipeline_params: dict = None, metrics: Optional[dict] = None,
                        note: str = "update", params_format: str = 'yaml',
                        delete_model_after_save: bool = True):
        """Register changes to an existing model.
        
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Existing model identifier
            model: Updated model object
            pipeline_params: Updated pipeline parameters
            metrics: Updated metrics
            note: Note about the update
            params_format: Format for params
            delete_model_after_save: If True, delete model from memory after saving
        """
        self.logger.info(f"register_changes called: fedcore_id={fedcore_id}, model_id={model_id}, metrics={metrics}")
        
        existing = self.storage.get_latest_record(fedcore_id, model_id)
        if existing is None:
            self.logger.warning("No existing record found, calling register_model instead")
            self.register_model(fedcore_id, model, None, pipeline_params, metrics, note, 
                              params_format, delete_model_after_save)
            return
        
        if self.auto_cleanup and torch.cuda.is_available():
            mem_before = self.checkpoint_manager.get_gpu_memory_stats()
            self.logger.info(f"GPU memory before saving changes: {mem_before.get('allocated_gb', 0):.4f} GB")
        
        version = self.metrics_tracker.generate_version()
        safe_timestamp = self.metrics_tracker.sanitize_timestamp(version)
        
        checkpoint_bytes = self.checkpoint_manager.serialize_to_bytes(model, None)
        checkpoint_path = self.checkpoint_manager.generate_checkpoint_path(
            fedcore_id, model_id, safe_timestamp
        )
        self.checkpoint_manager.save_to_file(checkpoint_bytes, checkpoint_path)
        
        record = {
            "record_id": str(self.metrics_tracker.generate_model_id()),
            "fedcore_id": fedcore_id,
            "model_id": model_id,
            "version": version,
            "created_at": version,
            "model_path": None,
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics if metrics is not None else {},
            "pipeline_params": None,
        }
        
        self.storage.append_record(fedcore_id, record)
        
        if delete_model_after_save and model is not None:
            self.logger.info(f"Deleting model {model_id} from memory")
            self._delete_model_from_memory(model)
            self.logger.info("Model deleted from memory")
        
        if self.auto_cleanup and torch.cuda.is_available():
            mem_after = self.checkpoint_manager.get_gpu_memory_stats()
            self.logger.info(f"GPU memory after cleanup: {mem_after.get('allocated_gb', 0):.4f} GB")
    
    def update_metrics(self, fedcore_id: str, model_id: str, metrics: dict):
        """Update metrics for the latest version of a model.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            metrics: Metrics dictionary to merge
        """
        self.storage.update_record_metrics(fedcore_id, model_id, metrics)
    
    def get_latest_record(self, fedcore_id: str, model_id: str) -> Optional[dict]:
        """Get the latest record for a specific model.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            
        Returns:
            Latest record dictionary or None
        """
        return self.storage.get_latest_record(fedcore_id, model_id)
    
    def get_model_history(self, fedcore_id: str, model_id: str):
        """Get complete history of a model.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            
        Returns:
            DataFrame with model history
        """
        return self.storage.get_records(fedcore_id, model_id)
    
    def get_best_checkpoint(self, fedcore_id: str, metric_name: str, mode: str = "max") -> Optional[dict]:
        """Get the best checkpoint based on a specific metric.
        Args:
            fedcore_id: FedCore instance identifier
            metric_name: Metric to optimize
            mode: 'max' or 'min'
            
        Returns:
            Best checkpoint record or None
        """
        df = self.storage.load(fedcore_id)
        return self.metrics_tracker.find_best_checkpoint(df, metric_name, mode)
    
    def load_model_from_latest_checkpoint(self, fedcore_id: str, model_id: str,
                                         device: torch.device = None) -> Optional[torch.nn.Module]:
        """Load model from the latest checkpoint.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            device: Device to load model on
            
        Returns:
            Loaded model or None
        """
        latest = self.storage.get_latest_record(fedcore_id, model_id)
        
        if latest is None or not latest.get('checkpoint_path'):
            return None
        
        return self.checkpoint_manager.load_from_file(latest['checkpoint_path'], device)
    
    def list_models(self, fedcore_id: str) -> list:
        """List all unique model IDs in the registry.
        Args:
            fedcore_id: FedCore instance identifier
            
        Returns:
            List of model IDs
        """
        return self.storage.list_model_ids(fedcore_id)
    
    def get_model_with_fallback(self, fedcore_id: str, model_id: str,
                               fallback_model=None, device: torch.device = None):
        """Load model from registry with fallback to provided model.
        
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            fallback_model: Model to use if loading from registry fails
            device: Device to load model on
            
        Returns:
            Loaded model from registry or fallback_model
        """
        try:
            loaded_model = self.load_model_from_latest_checkpoint(fedcore_id, model_id, device)
            if loaded_model is not None:
                return loaded_model
        except Exception as e:
            print(f"Failed to load model from registry: {e}")
        
        return fallback_model
    
    def _delete_model_from_memory(self, model) -> None:
        """Delete model from memory and clean up GPU memory.
        
        Args:
            model: Model object to delete
        """
        if model is None:
            return
        
        if hasattr(model, 'cpu'):
            try:
                model.cpu()
            except Exception as e:
                self.logger.debug(f"Could not move model to CPU: {e}")
        
        if isinstance(model, torch.nn.Module):
            try:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None
                    param.data = None
                
                for buffer in model.buffers():
                    buffer.data = None
                    
            except Exception as e:
                self.logger.debug(f"Could not clear model parameters: {e}")
        
        del model
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return self.checkpoint_manager.get_gpu_memory_stats()
    
    def force_cleanup(self) -> None:
        """Force cleanup of GPU memory.
        
        Useful for manual cleanup when needed.
        """
        self.checkpoint_manager._cleanup_gpu_memory()
        self.logger.info("Forced GPU memory cleanup executed")

