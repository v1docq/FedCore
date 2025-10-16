"""Metrics tracking and versioning for model registry."""

import hashlib
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd


class MetricsTracker:
    """Handles model versioning and metrics tracking."""
    @staticmethod
    def generate_model_id(model=None, model_path: Optional[str] = None) -> str:
        """Generate unique model identifier.
        Args:
            model: Model object
            model_path: Path to model file
            
        Returns:
            Unique model identifier
        """
        if model is not None:
            return f"model_{id(model)}"
        elif model_path is not None:
            hash_val = hashlib.md5(model_path.encode()).hexdigest()[:16]
            return f"path_{hash_val}"
        else:
            return str(uuid.uuid4())
    
    @staticmethod
    def generate_version() -> str:
        """Generate timestamp-based version for chronological ordering.
        
        Returns:
            ISO timestamp string
        """
        return datetime.utcnow().isoformat()
    
    @staticmethod
    def sanitize_timestamp(timestamp: str) -> str:
        """Sanitize timestamp for use in filenames.
        Args:
            timestamp: ISO timestamp
            
        Returns:
            Filesystem-safe timestamp
        """
        return timestamp.replace(':', '-')
    
    @staticmethod
    def find_best_checkpoint(df: pd.DataFrame, metric_name: str, mode: str = "max") -> Optional[dict]:
        """Find best checkpoint based on metric.
        Args:
            df: DataFrame with registry records
            metric_name: Metric to optimize
            mode: 'max' or 'min'
            
        Returns:
            Best record as dictionary or None
        """
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
        best = df_copy.sort_values(
            ["_metric_val", "version"], 
            ascending=[ascending, False]
        ).iloc[0]
        
        result = best.drop(labels=["_metric_val"]).to_dict()
        return result

