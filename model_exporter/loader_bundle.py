"""Portable dataloader bundles for the FedCore server wrapper.

I/O contract
------------
Save:   DataLoader → ``*.pt`` dict with tensors + structural metadata
Load:   path → metadata (and optional tensors)

No train/test/calibration roles — the server is a generic ops wrapper;
callers name files as they like and decide how to use the data later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset

BUNDLE_KIND = "fedcore_dataloader_bundle"
BUNDLE_VERSION = 2


@dataclass
class LoaderBundleMeta:
    """JSON-safe summary for the Web UI (no heavy tensors)."""

    kind: str = BUNDLE_KIND
    version: int = BUNDLE_VERSION
    name: str = ""
    num_samples: int = 0
    batch_size: int = 0
    num_batches: int = 0
    sample_shape: List[int] = field(default_factory=list)
    num_classes: Optional[int] = None
    dtype: str = ""
    file_name: str = ""
    has_tensors: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoaderBundle:
    """Build / load / inspect portable ``.pt`` dataloader bundles."""

    @classmethod
    def from_dataloader(
        cls,
        loader: DataLoader,
        *,
        name: str = "",
        num_classes: Optional[int] = None,
        materialize: bool = True,
    ) -> Dict[str, Any]:
        features, targets = cls._materialize(loader) if materialize else (None, None)
        if features is None:
            raise ValueError("Cannot materialize empty dataloader")

        sample_shape = list(features.shape[1:])
        batch_size = int(getattr(loader, "batch_size", 0) or 0)
        num_samples = int(features.shape[0])
        num_batches = (num_samples + max(batch_size, 1) - 1) // max(batch_size, 1)

        return {
            "kind": BUNDLE_KIND,
            "version": BUNDLE_VERSION,
            "name": name or "",
            "num_samples": num_samples,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "sample_shape": sample_shape,
            "num_classes": num_classes,
            "dtype": str(features.dtype).replace("torch.", ""),
            "features": features,
            "targets": targets,
        }

    @classmethod
    def save(
        cls,
        path: Union[str, Path],
        loader: DataLoader,
        *,
        name: str = "",
        num_classes: Optional[int] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = cls.from_dataloader(
            loader,
            name=name or path.stem,
            num_classes=num_classes,
        )
        torch.save(bundle, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or obj.get("kind") != BUNDLE_KIND:
            raise ValueError(
                f"Not a FedCore loader bundle ({path.name}). "
                f"Expected kind={BUNDLE_KIND}."
            )
        return obj

    @classmethod
    def inspect(cls, path: Union[str, Path]) -> LoaderBundleMeta:
        path = Path(path)
        bundle = cls.load(path)
        features = bundle.get("features")
        sample_shape = list(bundle.get("sample_shape") or [])
        if not sample_shape and isinstance(features, torch.Tensor):
            sample_shape = list(features.shape[1:])
        return LoaderBundleMeta(
            kind=str(bundle.get("kind", BUNDLE_KIND)),
            version=int(bundle.get("version", BUNDLE_VERSION)),
            name=str(bundle.get("name") or path.stem),
            num_samples=int(bundle.get("num_samples") or 0),
            batch_size=int(bundle.get("batch_size") or 0),
            num_batches=int(bundle.get("num_batches") or 0),
            sample_shape=sample_shape,
            num_classes=bundle.get("num_classes"),
            dtype=str(bundle.get("dtype", "")),
            file_name=path.name,
            has_tensors=isinstance(features, torch.Tensor),
        )

    @classmethod
    def to_dataloader(
        cls, bundle: Dict[str, Any], batch_size: Optional[int] = None
    ) -> DataLoader:
        features = bundle["features"]
        targets = bundle["targets"]
        bs = batch_size or int(bundle.get("batch_size") or 64)
        return DataLoader(
            TensorDataset(features, targets), batch_size=bs, shuffle=False
        )

    @staticmethod
    def _materialize(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("Loader batches must be (features, targets)")
            xs.append(x.detach().cpu())
            ys.append(
                y.detach().cpu() if torch.is_tensor(y) else torch.as_tensor(y)
            )
        if not xs:
            raise ValueError("Loader is empty")
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
