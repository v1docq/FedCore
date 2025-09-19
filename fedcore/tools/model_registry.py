import io
import os
import uuid
import torch
from datetime import datetime
from typing import Optional

import pandas as pd



class ModelRegistry:
    """
    In-memory model registry.

    - Stores checkpoints directly inside the DataFrame as bytes (not in a DB).
    - Provides simple singleton-style access.
    - Persists the registry to disk via pickle by default (optional).

    DataFrame fields:
        record_id - unique id per registry record
        model_id - logical model identifier (stable across versions)
        version - monotonically increasing per model_id
        created_at - ISO timestamp
        model_path - optional path used when provided
        checkpoint_bytes - serialized checkpoint bytes
        note - optional short note
    """

    _instance: Optional["ModelRegistry"] = None
    _df: Optional[pd.DataFrame] = None
    _columns = [
        "record_id",  
        "model_id",  
        "version",  
        "created_at",  
        "model_path",  
        "checkpoint_path",  
        "checkpoint_bytes",  
        "metrics",  
        "note",  
    ]
    _default_registry_path = os.environ.get(
        "FEDCORE_MODEL_REGISTRY_PATH",
        os.path.join("llm_output", "model_registry.pkl"),
    )

    def __init__(self, model=None):
        self.model = model
        if ModelRegistry._df is None:
            ModelRegistry._df = pd.DataFrame(columns=ModelRegistry._columns)
            try:
                ModelRegistry.load_registry()
            except Exception:
                ModelRegistry._df = pd.DataFrame(columns=ModelRegistry._columns)
        self._current_model_id: Optional[str] = None

    def get_instance(self, model=None):
        """Return a process-wide instance, optionally updating attached model."""
        if ModelRegistry._instance is None:
            ModelRegistry._instance = ModelRegistry(model=model)
        else:
            if model is not None:
                ModelRegistry._instance.model = model
        return ModelRegistry._instance

    @staticmethod
    def save_registry():
        """Persist the in-memory registry DataFrame to disk (pickle).

        Uses FEDCORE_MODEL_REGISTRY_PATH or defaults to llm_output/model_registry.pkl
        """
        target_path = ModelRegistry._default_registry_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if ModelRegistry._df is None:
            ModelRegistry._df = pd.DataFrame(columns=ModelRegistry._columns)
        ModelRegistry._df.to_pickle(target_path)

    @staticmethod
    def load_registry():
        """Load the registry DataFrame from disk into memory if file exists."""
        target_path = ModelRegistry._default_registry_path
        if os.path.isfile(target_path):
            ModelRegistry._df = pd.read_pickle(target_path)
        else:
            ModelRegistry._df = pd.DataFrame(columns=ModelRegistry._columns)

    def _serialize_checkpoint_to_bytes(self, model_path: Optional[str]) -> Optional[bytes]:
        """Serialize checkpoint either from a file path or from the attached model."""
        if model_path and os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                return f.read()

        if self.model is not None:
            buffer = io.BytesIO()
            if hasattr(self.model, "state_dict"):
                torch.save(self.model.state_dict(), buffer)
            else:
                torch.save(self.model, buffer)
            return buffer.getvalue()

        return None

    def _default_checkpoint_path(self, model_id: str, version: int) -> str:
        base_dir = os.environ.get(
            "FEDCORE_CHECKPOINT_DIR",
            os.path.join("llm_output", "checkpoints"),
        )
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{model_id}_v{version}.pt")

    def _materialize_checkpoint(self, checkpoint_bytes: Optional[bytes], target_path: str) -> None:
        if checkpoint_bytes is None:
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(checkpoint_bytes)

    def _next_version_for(self, model_id: str) -> int:
        """Compute next version number for given model_id based on existing rows."""
        if ModelRegistry._df is None or ModelRegistry._df.empty:
            return 1
        rows = ModelRegistry._df[ModelRegistry._df["model_id"] == model_id]
        if rows.empty:
            return 1
        return int(rows["version"].max()) + 1

    def register_model(self, model_path: str = None, metrics: Optional[dict] = None, note: str = "initial"):
        """Register a new logical model and its initial checkpoint.

        Returns the newly created model_id.
        """
        model_id = str(uuid.uuid4())
        self._current_model_id = model_id

        checkpoint_bytes = self._serialize_checkpoint_to_bytes(model_path)
        version = 1
        checkpoint_path = model_path if (model_path and os.path.isfile(model_path)) else self._default_checkpoint_path(model_id, version)
        self._materialize_checkpoint(checkpoint_bytes, checkpoint_path)
        record = {
            "record_id": str(uuid.uuid4()),
            "model_id": model_id,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "model_path": model_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics if metrics is not None else {},
            "note": note,
        }

        ModelRegistry._df = pd.concat(
            [ModelRegistry._df, pd.DataFrame([record])], ignore_index=True
        )

        ModelRegistry.save_registry()

    @staticmethod
    def update_metrics(model_id: str, metrics: dict, version: Optional[int] = None) -> None:
        if ModelRegistry._df is None or ModelRegistry._df.empty:
            return
        df = ModelRegistry._df
        rows = df[df["model_id"] == model_id]
        if rows.empty:
            return
        if version is None:
            version = int(rows["version"].max())
        mask = (df["model_id"] == model_id) & (df["version"] == version)
        if not mask.any():
            return
        current_metrics = df.loc[mask, "metrics"].iloc[0]
        if not isinstance(current_metrics, dict):
            current_metrics = {}
        new_metrics = dict(current_metrics)
        new_metrics.update(metrics)
        ModelRegistry._df.loc[mask, "metrics"] = [new_metrics]
        ModelRegistry.save_registry()

    @staticmethod
    def get_latest_record(model_id: str) -> Optional[dict]:
        if ModelRegistry._df is None or ModelRegistry._df.empty:
            return None
        rows = ModelRegistry._df[ModelRegistry._df["model_id"] == model_id]
        if rows.empty:
            return None
        latest = rows.sort_values("version", ascending=False).iloc[0]
        return latest.to_dict()

    @staticmethod
    def get_best_checkpoint(metric_name: str, mode: str = "max") -> Optional[dict]:
        if ModelRegistry._df is None or ModelRegistry._df.empty:
            return None
        def extract_metric(row):
            m = row.get("metrics", {})
            return m.get(metric_name, None) if isinstance(m, dict) else None
        df = ModelRegistry._df.copy()
        df["_metric_val"] = df.apply(lambda r: extract_metric(r), axis=1)
        df = df[df["_metric_val"].notnull()]
        if df.empty:
            return None
        ascending = True if mode == "min" else False
        best = df.sort_values(["_metric_val", "version"], ascending=[ascending, False]).iloc[0]
        result = best.drop(labels=["_metric_val"]).to_dict()
        return result

    def register_changes(metrics: Optional[dict] = None, note: str = "update"):
        """Append a new checkpoint version for the last registered model.

        Uses the instance's currently attached model (or a previously provided path)
        to capture the latest weights. If no model has been registered yet, the call
        is a no-op.
        """
        instance = ModelRegistry._instance
        if instance is None or instance._current_model_id is None:
            return

        model_id = instance._current_model_id
        checkpoint_bytes = instance._serialize_checkpoint_to_bytes(model_path=None)
        version = instance._next_version_for(model_id)
        checkpoint_path = instance._default_checkpoint_path(model_id, version)
        instance._materialize_checkpoint(checkpoint_bytes, checkpoint_path)
        record = {
            "record_id": str(uuid.uuid4()),
            "model_id": model_id,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "model_path": None,
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics if metrics is not None else {},
            "note": note,
        }

        ModelRegistry._df = pd.concat(
            [ModelRegistry._df, pd.DataFrame([record])], ignore_index=True
        )

        ModelRegistry.save_registry()