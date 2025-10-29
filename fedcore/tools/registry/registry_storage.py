"""Registry storage layer using pandas DataFrame."""

import os
import logging
from typing import Dict, Optional

import pandas as pd


class RegistryStorage:
    """Manages persistent storage of model registry data."""

    COLUMNS = [
        "record",
        "fedcore",
        "model",
        "created_at",
        "model_path",
        "checkpoint_path",
        "stage",
        "mode",
        "metrics",
    ]

    def __init__(self, base_dir: str):
        """Initialize registry storage.
        Args:
            base_dir: Base directory for storing registry files
        """
        self.base_dir = base_dir
        self._registries: Dict[str, pd.DataFrame] = {}

    def get_registry_path(self, fedcore_id: str) -> str:
        """Get path to registry file for specific fedcore instance."""
        registry_dir = os.path.join(self.base_dir, "registries")
        os.makedirs(registry_dir, exist_ok=True)
        return os.path.join(registry_dir, f"{fedcore_id}_registry.csv")

    def load(self, fedcore_id: str) -> pd.DataFrame:
        """Load registry from disk.
        Args:
            fedcore_id: FedCore instance ID
            
        Returns:
            DataFrame with registry data
        """
        if fedcore_id in self._registries:
            return self._registries[fedcore_id]

        registry_path = self.get_registry_path(fedcore_id)

        if os.path.isfile(registry_path):
            df = pd.read_csv(registry_path)
        else:
            df = pd.DataFrame(columns=self.COLUMNS)

        self._registries[fedcore_id] = df
        return df

    def save(self, fedcore_id: str, df: pd.DataFrame) -> None:
        """Save registry to disk.
        Args:
            fedcore_id: FedCore instance ID
            df: DataFrame to save
        """
        self._registries[fedcore_id] = df

        registry_path = self.get_registry_path(fedcore_id)
        df.to_csv(registry_path)

    def append_record(self, fedcore_id: str, record: dict) -> None:
        """Append new record to registry.
        Args:
            fedcore_id: FedCore instance ID
            record: Record dictionary to append
        """
        df = self.load(fedcore_id)
        new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        self.save(fedcore_id, new_df)

    def get_records(self, fedcore_id: str, model_id: str) -> pd.DataFrame:
        """Get all records for specific model.
        Args:
            fedcore_id: FedCore instance ID
            model_id: Model identifier
            
        Returns:
            DataFrame with matching records
        """
        df = self.load(fedcore_id)
        if df.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        return df[df["model"] == model_id].sort_values("created_at")

    def get_latest_record(self, fedcore_id: str, model_id: str) -> Optional[dict]:
        """Get latest record for specific model.
        Args:
            fedcore_id: FedCore instance ID
            model_id: Model identifier
            
        Returns:
            Latest record as dictionary or None
        """
        records = self.get_records(fedcore_id, model_id)

        if records.empty:
            return None

        latest = records.iloc[-1]
        return latest.to_dict()

    def update_record_metrics(self, fedcore_id: str, model_id: str, metrics: dict) -> None:
        """Update metrics in the latest record.
        Args:
            fedcore_id: FedCore instance ID
            model_id: Model identifier
            metrics: Metrics dictionary to merge
        """
        df = self.load(fedcore_id)

        records = df[df["model"] == model_id]
        if records.empty:
            logging.info(f"Warning: No records found for model_id {model_id}")
            return

        latest_idx = records["created_at"].idxmax() if "created_at" in records else records.index[-1]

        if isinstance(metrics, dict):
            metrics_copy = metrics.copy()
            stage = metrics_copy.pop("stage", None)
            mode = metrics_copy.pop("operation", None)
            metrics_copy.pop("is_processed", None)
            
            if stage is not None:
                if isinstance(stage, str):
                    if "before" in stage.lower():
                        stage = "before"
                    elif "after" in stage.lower():
                        stage = "after"
                df.at[latest_idx, "stage"] = stage
            
            if mode is not None:
                df.at[latest_idx, "mode"] = mode
            
            existing_metrics = df.at[latest_idx, "metrics"]
            if not isinstance(existing_metrics, dict):
                existing_metrics = {}
            
            existing_metrics.update(metrics_copy)
            df.at[latest_idx, "metrics"] = existing_metrics
        else:
            df.at[latest_idx, "metrics"] = metrics

        self.save(fedcore_id, df)

    def list_model_ids(self, fedcore_id: str) -> list:
        """List all unique model IDs in registry.
        Args:
            fedcore_id: FedCore instance ID
            
        Returns:
            List of unique model IDs
        """
        df = self.load(fedcore_id)
        if df.empty:
            return []
        return df["model"].unique().tolist()