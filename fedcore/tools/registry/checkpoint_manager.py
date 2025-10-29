"""Checkpoint management for model registry."""

import gc
import io
import os
import logging
from typing import Optional

import torch


class CheckpointManager:
    """Manages model checkpoint serialization and deserialization."""
    
    def __init__(self, base_dir: str, auto_cleanup: bool = True):
        """Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for storing checkpoints
            auto_cleanup: If True, automatically clean GPU memory after saving checkpoints
        """
        self.base_dir = base_dir
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
    
    def get_checkpoint_dir(self, fedcore_id: str) -> str:
        """Get checkpoint directory for specific fedcore instance."""
        return os.path.join(self.base_dir, "checkpoints", fedcore_id)
    
    def generate_checkpoint_path(self, fedcore_id: str, model_id: str, timestamp: str) -> str:
        """Generate unique checkpoint path.
        Args:
            fedcore_id: FedCore instance ID
            model_id: Model identifier
            timestamp: ISO timestamp (will be sanitized for filesystem)
        Returns:
            Full path to checkpoint file
        """
        checkpoint_dir = self.get_checkpoint_dir(fedcore_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        safe_model_id = model_id.replace('/', '_').replace('\\', '_')
        filename = f"{safe_model_id}_{timestamp}.pt"
        return os.path.join(checkpoint_dir, filename)
    
    def serialize_to_bytes(self, model=None, model_path: Optional[str] = None) -> Optional[bytes]:
        """Serialize checkpoint to bytes.
        Args:
            model: Model object to serialize
            model_path: Path to existing checkpoint file
            
        Returns:
            Serialized checkpoint as bytes
        """
        if model_path and os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                return f.read()
        
        if model is not None:
            buffer = io.BytesIO()
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), buffer)
            else:
                torch.save(model, buffer)
            return buffer.getvalue()
        
        return None
    
    def save_to_file(self, checkpoint_bytes: Optional[bytes], target_path: str, 
                     cleanup_after_save: bool = None) -> None:
        """Save checkpoint bytes to file.
        
        Args:
            checkpoint_bytes: Serialized checkpoint
            target_path: Target file path
            cleanup_after_save: If True, clean GPU memory after saving. 
                              If None, use self.auto_cleanup setting
        """
        if checkpoint_bytes is None:
            return
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(checkpoint_bytes)
        
        should_cleanup = cleanup_after_save if cleanup_after_save is not None else self.auto_cleanup
        if should_cleanup:
            self.logger.info("Cleaning GPU memory after saving checkpoint")
            self._cleanup_gpu_memory()
            self.logger.info("GPU memory cleanup completed")
    
    def load_from_file(self, checkpoint_path: str, 
                      device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
        """Load model from checkpoint file.
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded model or checkpoint dict
        """
        if not os.path.exists(checkpoint_path):
            self.logger.info(f"Checkpoint file not found: {checkpoint_path}")
            return None
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
                return ckpt['model']
            elif 'state_dict' in ckpt:
                self.logger.info(f"Warning: Only state_dict found. Need model architecture to load.")
                return ckpt
        return ckpt
    
    def deserialize_from_bytes(self, checkpoint_bytes: bytes, 
                               device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
        """Load model from checkpoint bytes.
        Args:
            checkpoint_bytes: Serialized checkpoint bytes
            device: Device to load model on
            
        Returns:
            Loaded model or checkpoint dict
        """
        if checkpoint_bytes is None:
            return None
            
        buffer = io.BytesIO(checkpoint_bytes)
        ckpt = torch.load(buffer, map_location=device)
        
        if isinstance(ckpt, dict):
            if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
                return ckpt['model']
            elif 'state_dict' in ckpt:
                self.logger.info(f"Warning: Only state_dict found. Need model architecture to load.")
                return ckpt
        return ckpt
    
    def get_gpu_memory_stats(self) -> dict:
        """Get current GPU memory statistics.
        
        Returns:
            Dictionary with GPU memory statistics:
            - allocated_gb: Currently allocated memory in GB
            - reserved_gb: Currently reserved memory in GB
            - allocated_mb: Currently allocated memory in MB
            - reserved_mb: Currently reserved memory in MB
        """
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'allocated_mb': 0.0,
                'reserved_mb': 0.0
            }
        
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        
        return {
            'allocated_gb': allocated_bytes / (1024 ** 3),
            'reserved_gb': reserved_bytes / (1024 ** 3),
            'allocated_mb': allocated_bytes / (1024 ** 2),
            'reserved_mb': reserved_bytes / (1024 ** 2)
        }
    
    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory by clearing cache and running garbage collection."""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                self.logger.debug(f"Could not run torch.cuda.ipc_collect(): {e}")