"""Checkpoint management for model registry."""

import io
import os
import logging
from typing import Optional

import torch


class CheckpointManager:
    """Manages model checkpoint serialization and deserialization."""
    
    def __init__(self, base_dir: str):
        """Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for storing checkpoints
        """
        self.base_dir = base_dir
    
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
    
    def save_to_file(self, checkpoint_bytes: Optional[bytes], target_path: str) -> None:
        """Save checkpoint bytes to file.
        
        Args:
            checkpoint_bytes: Serialized checkpoint
            target_path: Target file path
        """
        if checkpoint_bytes is None:
            return
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(checkpoint_bytes)
    
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
            logging.info(f"Checkpoint file not found: {checkpoint_path}")
            return None
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
                return ckpt['model']
            elif 'state_dict' in ckpt:
                logging.info(f"Warning: Only state_dict found. Need model architecture to load.")
                return ckpt
        return ckpt

