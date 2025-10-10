"""Refactored Model Registry with cleaner architecture."""

import os
from typing import Optional

import torch

from .checkpoint_manager import CheckpointManager
from .registry_storage import RegistryStorage
from .metrics_tracker import MetricsTracker


class ModelRegistry:
    """
    Singleton model registry for FedCore pipeline.
    
    Manages model checkpoints, metrics, and versioning across the PEFT pipeline.
    Uses composition pattern with specialized managers for different responsibilities.
    
    Components:
        - CheckpointManager: Handles checkpoint serialization/deserialization
        - RegistryStorage: Manages DataFrame persistence
        - MetricsTracker: Tracks metrics and versions
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton)."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry components (only once)."""
        if not ModelRegistry._initialized:
            base_dir = os.environ.get("FEDCORE_MODEL_REGISTRY_PATH", "llm_output")
            
            self.checkpoint_manager = CheckpointManager(base_dir)
            self.storage = RegistryStorage(base_dir)
            self.metrics_tracker = MetricsTracker()
            
            ModelRegistry._initialized = True
    
    @classmethod
    def get_instance(cls, model=None, model_path: Optional[str] = None) -> "ModelRegistry":
        """Get singleton instance (compatibility method).
        
        Args:
            model: Ignored (for compatibility)
            model_path: Ignored (for compatibility)
            
        Returns:
            ModelRegistry singleton instance
        """
        return cls()
    
    @classmethod
    def register_model(cls, fedcore_id: str, model=None, model_path: str = None,
                      pipeline_params: dict = None, metrics: Optional[dict] = None,
                      note: str = "initial", params_format: str = 'yaml') -> str:
        """Register a new model in the registry.
        
        Args:
            fedcore_id: FedCore instance identifier
            model: Model object to register
            model_path: Path to model file
            pipeline_params: Pipeline parameters (currently unused, can be extended)
            metrics: Metrics dictionary
            note: Note about this registration (unused)
            params_format: Format for params (unused)
            
        Returns:
            model_id: Generated model identifier
        """
        registry = cls()
        
        model_id = registry.metrics_tracker.generate_model_id(model, model_path)
        version = registry.metrics_tracker.generate_version()
        safe_timestamp = registry.metrics_tracker.sanitize_timestamp(version)
        checkpoint_bytes = registry.checkpoint_manager.serialize_to_bytes(model, model_path)
        
        if model_path and os.path.isfile(model_path):
            checkpoint_path = model_path
        else:
            checkpoint_path = registry.checkpoint_manager.generate_checkpoint_path(
                fedcore_id, model_id, safe_timestamp
            )
            registry.checkpoint_manager.save_to_file(checkpoint_bytes, checkpoint_path)
        
        record = {
            "record_id": str(registry.metrics_tracker.generate_model_id()),
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
        
        registry.storage.append_record(fedcore_id, record)
        
        return model_id
    
    @classmethod
    def register_changes(cls, fedcore_id: str, model_id: str, model=None,
                        pipeline_params: dict = None, metrics: Optional[dict] = None,
                        note: str = "update", params_format: str = 'yaml'):
        """Register changes to an existing model.
        
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Existing model identifier
            model: Updated model object
            pipeline_params: Updated pipeline parameters
            metrics: Updated metrics
            note: Note about the update
            params_format: Format for params
        """
        registry = cls()


        
        existing = registry.storage.get_latest_record(fedcore_id, model_id)
        if existing is None:
            cls.register_model(fedcore_id, model, None, pipeline_params, metrics, note, params_format)
            return
        
        version = registry.metrics_tracker.generate_version()
        safe_timestamp = registry.metrics_tracker.sanitize_timestamp(version)
        
        checkpoint_bytes = registry.checkpoint_manager.serialize_to_bytes(model, None)
        checkpoint_path = registry.checkpoint_manager.generate_checkpoint_path(
            fedcore_id, model_id, safe_timestamp
        )
        registry.checkpoint_manager.save_to_file(checkpoint_bytes, checkpoint_path)
        
        record = {
            "record_id": str(registry.metrics_tracker.generate_model_id()),
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
        
        registry.storage.append_record(fedcore_id, record)
    
    @classmethod
    def update_metrics(cls, fedcore_id: str, model_id: str, metrics: dict):
        """Update metrics for the latest version of a model.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            metrics: Metrics dictionary to merge
        """
        registry = cls()
        registry.storage.update_record_metrics(fedcore_id, model_id, metrics)
    
    @classmethod
    def get_latest_record(cls, fedcore_id: str, model_id: str) -> Optional[dict]:
        """Get the latest record for a specific model.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            
        Returns:
            Latest record dictionary or None
        """
        registry = cls()
        return registry.storage.get_latest_record(fedcore_id, model_id)
    
    @classmethod
    def get_model_history(cls, fedcore_id: str, model_id: str):
        """Get complete history of a model.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            
        Returns:
            DataFrame with model history
        """
        registry = cls()
        return registry.storage.get_records(fedcore_id, model_id)
    
    @classmethod
    def get_best_checkpoint(cls, fedcore_id: str, metric_name: str, mode: str = "max") -> Optional[dict]:
        """Get the best checkpoint based on a specific metric.
        Args:
            fedcore_id: FedCore instance identifier
            metric_name: Metric to optimize
            mode: 'max' or 'min'
            
        Returns:
            Best checkpoint record or None
        """
        registry = cls()
        df = registry.storage.load(fedcore_id)
        return registry.metrics_tracker.find_best_checkpoint(df, metric_name, mode)
    
    @classmethod
    def load_model_from_latest_checkpoint(cls, fedcore_id: str, model_id: str,
                                         device: torch.device = None) -> Optional[torch.nn.Module]:
        """Load model from the latest checkpoint.
        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier
            device: Device to load model on
            
        Returns:
            Loaded model or None
        """
        registry = cls()
        latest = registry.storage.get_latest_record(fedcore_id, model_id)
        
        if latest is None or not latest.get('checkpoint_path'):
            return None
        
        return registry.checkpoint_manager.load_from_file(latest['checkpoint_path'], device)
    
    @classmethod
    def list_models(cls, fedcore_id: str) -> list:
        """List all unique model IDs in the registry.
        Args:
            fedcore_id: FedCore instance identifier
            
        Returns:
            List of model IDs
        """
        registry = cls()
        return registry.storage.list_model_ids(fedcore_id)

