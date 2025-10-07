import io
import os
import uuid
import torch
import json
import yaml
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd



class ModelRegistry:
    """
    In-memory model registry with Singleton pattern.

    - Stores checkpoints directly inside the DataFrame as bytes (not in a DB).
    - Implements proper singleton pattern to ensure only one instance exists.
    - Persists the registry to disk via pickle by default (optional).

    DataFrame fields:
        record_id - unique id per registry record
        fedcore_id - identifier of the fedcore instance
        model_id - logical model identifier (id(model) or hash of model_path)
        version - timestamp-based version for chronological ordering
        created_at - ISO timestamp
        model_path - optional path used when provided
        checkpoint_path - path to saved checkpoint file
        checkpoint_bytes - serialized checkpoint bytes
        metrics - dictionary of metrics
        pipeline_params - serialized pipeline parameters
    """

    _instance = None
    _initialized = False
    
    _registries: Dict[str, pd.DataFrame] = {}
    _df: Optional[pd.DataFrame] = None
    _columns = [
        "record_id",  
        "fedcore_id",  
        "model_id",  
        "version",  
        "created_at",  
        "model_path",  
        "checkpoint_path",  
        "checkpoint_bytes",  
        "metrics",  
        "pipeline_params",  
    ]
    _default_base_dir = os.environ.get(
        "FEDCORE_MODEL_REGISTRY_PATH",
        "llm_output",
    )

    def __new__(cls):
        """Ensure only one instance of ModelRegistry exists."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry only once."""
        if not ModelRegistry._initialized:
            ModelRegistry._initialized = True

    @classmethod
    def _get_registry_path(cls, fedcore_id: str) -> str:
        return os.path.join(cls._default_base_dir, "registries", f"{fedcore_id}_registry.pkl")

    @classmethod
    def _get_checkpoint_base_dir(cls, fedcore_id: str) -> str:
        """Get checkpoint base directory for specific fedcore_id."""
        return os.path.join(cls._default_base_dir, "checkpoints", fedcore_id)
    
    @classmethod
    def _get_registry(cls, fedcore_id: str) -> pd.DataFrame:
        """Get or load registry for specific fedcore_id."""
        if fedcore_id not in cls._registries:
            cls.load_registry(fedcore_id)
        return cls._registries[fedcore_id]

    @classmethod
    def _set_registry(cls, fedcore_id: str, df: pd.DataFrame):
        """Set registry for specific fedcore_id."""
        cls._registries[fedcore_id] = df

    @classmethod
    def _get_pipeline_params_base_dir(cls, fedcore_id: str) -> str:
        """Get pipeline parameters base directory for specific fedcore_id."""
        return os.path.join(cls._default_base_dir, "pipeline_params", fedcore_id)
    
    @classmethod
    def _serialize_pipeline_params(cls, params: Dict[str, Any], format: str = 'yaml') -> str:
        """Serialize pipeline parameters to YAML or JSON string."""
        if format.lower() == 'yaml':
            return yaml.dump(params, default_flow_style=False, allow_unicode=True)
        elif format.lower() == 'json':
            return json.dumps(params, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")

    @classmethod
    def load_registry(cls, fedcore_id: str):
        """Load the registry DataFrame from disk into memory for specific fedcore_id."""
        target_path = cls._get_registry_path(fedcore_id)
        if os.path.isfile(target_path):
            df = pd.read_pickle(target_path)
            cls._set_registry(fedcore_id, df)
        else:
            cls._set_registry(fedcore_id, pd.DataFrame(columns=cls._columns))

    @classmethod
    def save_registry(cls, fedcore_id: str):
        """Persist the in-memory registry DataFrame to disk for specific fedcore_id."""
        target_path = cls._get_registry_path(fedcore_id)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        df = cls._get_registry(fedcore_id)
        df.to_pickle(target_path)

    @classmethod
    def _deserialize_pipeline_params(cls, params_str: str, format: str = 'yaml') -> Dict[str, Any]:
        """Deserialize pipeline parameters from YAML or JSON string."""
        if not params_str:
            return {}
            
        if format.lower() == 'yaml':
            return yaml.safe_load(params_str)
        elif format.lower() == 'json':
            return json.loads(params_str)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")

    @classmethod
    def _save_pipeline_params_to_file(cls, params: Dict[str, Any], fedcore_id: str, 
                                    model_id: str, timestamp: str, format: str = 'yaml') -> str:
        """Save pipeline parameters to file and return the file path."""
        base_dir = cls._get_pipeline_params_base_dir(fedcore_id)
        os.makedirs(base_dir, exist_ok=True)
        
        safe_model_id = model_id.replace('/', '_').replace('\\', '_')
        filename = f"{safe_model_id}_{timestamp}.{format}"
        filepath = os.path.join(base_dir, filename)
        
        if format == 'yaml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(params, f, default_flow_style=False, allow_unicode=True)
        elif format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
        
        return filepath

    @classmethod
    def _generate_model_id(cls, model=None, model_path: Optional[str] = None) -> str:
        """Generate model_id from model object or model path."""
        if model is not None:
            return f"model_{id(model)}"
        elif model_path is not None:
            import hashlib
            return f"path_{hashlib.md5(model_path.encode()).hexdigest()[:16]}"
        else:
            return str(uuid.uuid4())
        
    @classmethod
    def _serialize_checkpoint_to_bytes(cls, model=None, model_path: Optional[str] = None) -> Optional[bytes]:
        """Serialize checkpoint either from a file path or from the model object."""
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

    @classmethod
    def _default_checkpoint_path(cls, fedcore_id: str, model_id: str, timestamp: str) -> str:
        """Generate checkpoint path with timestamp for uniqueness."""
        base_dir = cls._get_checkpoint_base_dir(fedcore_id)
        os.makedirs(base_dir, exist_ok=True)
        safe_model_id = model_id.replace('/', '_').replace('\\', '_')
        return os.path.join(base_dir, f"{safe_model_id}_{timestamp}.pt")

    @classmethod
    def _materialize_checkpoint(cls, checkpoint_bytes: Optional[bytes], target_path: str) -> None:
        """Save checkpoint bytes to file."""
        if checkpoint_bytes is None:
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(checkpoint_bytes)

    @classmethod
    def _get_next_version(cls, fedcore_id: str, model_id: str) -> str:
        """Generate timestamp-based version for chronological ordering."""
        return datetime.utcnow().isoformat()

    @classmethod
    def register_model(cls, fedcore_id: str, model=None, model_path: str = None, 
                      pipeline_params: Dict[str, Any] = None,
                      metrics: Optional[dict] = None, note: str = "initial",
                      params_format: str = 'yaml') -> str:
        """Register a new model in the specified fedcore registry.

        Args:
            fedcore_id: Identifier of the fedcore instance
            model: Model object (optional)
            model_path: Path to model file (optional)
            pipeline_params: Dictionary of pipeline parameters to save
            metrics: Dictionary of metrics
            params_format: Format for saving pipeline params ('yaml' or 'json')

        Returns:
            model_id: Generated model identifier
        """
        model_id = cls._generate_model_id(model, model_path)
        
        df = cls._get_registry(fedcore_id)
        
        checkpoint_bytes = cls._serialize_checkpoint_to_bytes(model, model_path)
        
        version = cls._get_next_version(fedcore_id, model_id)
        safe_timestamp = version.replace(':', '-')
        checkpoint_path = model_path if (model_path and os.path.isfile(model_path)) else cls._default_checkpoint_path(fedcore_id, model_id, safe_timestamp)
        
        cls._materialize_checkpoint(checkpoint_bytes, checkpoint_path)
        
        pipeline_params_str = None
        pipeline_params_path = None
        if pipeline_params:
            pipeline_params_str = cls._serialize_pipeline_params(pipeline_params, params_format)
            pipeline_params_path = cls._save_pipeline_params_to_file(
                pipeline_params, fedcore_id, model_id, safe_timestamp, params_format
            )
        
        record = {
            "record_id": str(uuid.uuid4()),
            "fedcore_id": fedcore_id,
            "model_id": model_id,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "model_path": model_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics if metrics is not None else {},
            "pipeline_params": pipeline_params_str,
        }

        new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        cls._set_registry(fedcore_id, new_df)
        cls.save_registry(fedcore_id)
        
        return model_id

    @classmethod
    def get_latest_record(cls, fedcore_id: str, model_id: str) -> Optional[dict]:
        """Get the latest record for a specific model in fedcore registry."""
        df = cls._get_registry(fedcore_id)
        if df.empty:
            return None
        
        records = df[df["model_id"] == model_id]
        if records.empty:
            return None
            
        latest = records.sort_values("version", ascending=False).iloc[0]
        return latest.to_dict()
    
    @classmethod
    def get_model_history(cls, fedcore_id: str, model_id: str) -> pd.DataFrame:
        """Get complete history of a model in chronological order."""
        df = cls._get_registry(fedcore_id)
        if df.empty:
            return pd.DataFrame(columns=cls._columns)
            
        history = df[df["model_id"] == model_id].sort_values("version")
        return history

    @classmethod
    def get_best_checkpoint(cls, fedcore_id: str, metric_name: str, mode: str = "max") -> Optional[dict]:
        """Get the best checkpoint based on a specific metric."""
        df = cls._get_registry(fedcore_id)
        if df.empty:
            return None
            
        def extract_metric(row):
            m = row.get("metrics", {})
            return m.get(metric_name, None) if isinstance(m, dict) else None
            
        df_copy = df.copy()
        df_copy["_metric_val"] = df_copy.apply(lambda r: extract_metric(r), axis=1)
        df_copy = df_copy[df_copy["_metric_val"].notnull()]
        
        if df_copy.empty:
            return None
            
        ascending = (mode == "min")
        best = df_copy.sort_values(["_metric_val", "version"], ascending=[ascending, False]).iloc[0]
        result = best.drop(labels=["_metric_val"]).to_dict()
        return result

    @classmethod
    def register_changes(cls, fedcore_id: str, model_id: str, model=None,
                        pipeline_params: Dict[str, Any] = None,
                        metrics: Optional[dict] = None, note: str = "update",
                        params_format: str = 'yaml'):
        """Register changes to an existing model in the registry.

        Args:
            fedcore_id: Identifier of the fedcore instance
            model_id: Existing model identifier
            model: Updated model object
            pipeline_params: Updated pipeline parameters
            metrics: Updated metrics dictionary
            params_format: Format for saving pipeline params ('yaml' or 'json')
        """
        df = cls._get_registry(fedcore_id)
        
        existing_records = df[df["model_id"] == model_id]
        if existing_records.empty:
            cls.register_model(fedcore_id, model, None, pipeline_params, metrics, note, params_format)
            return

        checkpoint_bytes = cls._serialize_checkpoint_to_bytes(model, None)
        
        version = cls._get_next_version(fedcore_id, model_id)
        safe_timestamp = version.replace(':', '-')
        checkpoint_path = cls._default_checkpoint_path(fedcore_id, model_id, safe_timestamp)
        
        cls._materialize_checkpoint(checkpoint_bytes, checkpoint_path)
        
        pipeline_params_str = None
        if pipeline_params:
            pipeline_params_str = cls._serialize_pipeline_params(pipeline_params, params_format)
            cls._save_pipeline_params_to_file(
                pipeline_params, fedcore_id, model_id, safe_timestamp, params_format
            )
        
        record = {
            "record_id": str(uuid.uuid4()),
            "fedcore_id": fedcore_id,
            "model_id": model_id,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "model_path": None,  # For updates, we don't have an original path
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics if metrics is not None else {},
            "pipeline_params": pipeline_params_str,
        }

        new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        cls._set_registry(fedcore_id, new_df)
        cls.save_registry(fedcore_id)

    @classmethod
    def get_pipeline_params(cls, fedcore_id: str, model_id: str, version: str = None, 
                          format: str = 'yaml') -> Optional[Dict[str, Any]]:
        """Get pipeline parameters for a specific model version."""
        record = cls.get_latest_record(fedcore_id, model_id)
        if not record or not record.get('pipeline_params'):
            return None
            
        return cls._deserialize_pipeline_params(record['pipeline_params'], format)

    @classmethod
    def list_models(cls, fedcore_id: str) -> list:
        """List all unique model_ids in the registry."""
        df = cls._get_registry(fedcore_id)
        if df.empty:
            return []
        return df["model_id"].unique().tolist()
    
    @classmethod
    def load_model_from_latest_checkpoint(cls, fedcore_id: str, model_id: str, 
                                        device: torch.device = None) -> Optional[torch.nn.Module]:
        """Load model from the latest checkpoint in registry."""
        latest = cls.get_latest_record(fedcore_id, model_id)
        if not latest or not latest.get('checkpoint_path'):
            return None
            
        checkpoint_path = latest['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return None
            
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
            return ckpt['model']
        elif 'state_dict' in ckpt:
            print(f"Warning: Only state_dict found in {checkpoint_path}. Need model architecture to load.")
            return ckpt
        else:
            print(f"Warning: Unknown checkpoint format in {checkpoint_path}")
            return ckpt

    @classmethod
    def get_model_with_fallback(cls, fedcore_id: str, model_id: str, 
                              fallback_model: torch.nn.Module,
                              device: torch.device = None) -> torch.nn.Module:
        loaded_model = cls.load_model_from_latest_checkpoint(fedcore_id, model_id, device)
        if loaded_model is not None and isinstance(loaded_model, torch.nn.Module):
            return loaded_model
        else:
            return fallback_model