"""Explicit CIFAR roles; the official test loader requires a frozen selection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .cv_sparse_protocol import PilotProtocol, digest


@dataclass(frozen=True)
class DataRoles:
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    search_validation: tuple[int, ...]
    seed: int

    def payload(self):
        return {'train': list(self.train), 'calibration': list(self.calibration),
                'search_validation': list(self.search_validation), 'seed': self.seed}

    @property
    def sha256(self):
        return digest(self.payload())


def split_roles(labels, *, seed):
    """Keep each of the 5000 official training labels per class in one role."""
    values = np.asarray(labels)
    if (values.shape != (50000,) or values.dtype.kind not in 'iu'
            or type(seed) is not int or seed < 0
            or not np.array_equal(np.unique(values), np.arange(10))):
        raise ValueError('Exact CIFAR-10 training labels and an explicit seed required')
    generator = np.random.default_rng(seed)
    train, calibration, validation = [], [], []
    for label in range(10):
        indices = np.flatnonzero(values == label)
        if len(indices) != 5000:
            raise ValueError('Official CIFAR-10 train must have 5000 rows per class')
        indices = generator.permutation(indices)
        calibration.extend(indices[:100].tolist())
        validation.extend(indices[100:600].tolist())
        train.extend(indices[600:].tolist())
    return DataRoles(tuple(sorted(train)), tuple(sorted(calibration)), tuple(sorted(validation)), seed)


@dataclass(frozen=True)
class FrozenSelection:
    protocol_sha256: str
    roles: tuple[tuple[str, str], ...]
    complete_recovery_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self):
        def is_digest(value):
            return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
        if (not is_digest(self.protocol_sha256) or not self.roles
                or any(type(role) is not str or not role or not is_digest(key) for role, key in self.roles)
                or not self.complete_recovery_sha256
                or any(not is_digest(key) or not is_digest(value) for key, value in self.complete_recovery_sha256)
                or set(key for _role, key in self.roles) != set(key for key, _sha in self.complete_recovery_sha256)):
            raise ValueError('Seal all selected roles and completed recovery artifacts before opening test')


class CifarData:
    """Public data with independent role indices and no implicit test reads."""
    mean = (.4914, .4822, .4465)
    std = (.2023, .1994, .2010)

    def __init__(self, root, protocol=PilotProtocol(), *, download=False):
        from torchvision.datasets import CIFAR10
        self.root = Path(root)
        self.protocol = protocol
        # CIFAR10 verifies the official files. Test bytes may be integrity-hashed,
        # but test labels/tensors are not loaded or evaluated here.
        source = CIFAR10(str(self.root), train=True, download=download)
        self.images = np.ascontiguousarray(source.data)
        self.labels = np.asarray(source.targets, dtype=np.int64)
        self.roles = split_roles(self.labels, seed=protocol.split_seed)
        self.source_hashes = {}
        for path in sorted((self.root / 'cifar-10-batches-py').iterdir()):
            if path.is_file():
                hasher = hashlib.sha256()
                with path.open('rb') as stream:
                    for part in iter(lambda: stream.read(1024**2), b''):
                        hasher.update(part)
                self.source_hashes[path.name] = {'sha256': hasher.hexdigest(), 'bytes': path.stat().st_size}

    def rows(self, indices):
        import torch
        images = torch.from_numpy(self.images[np.asarray(indices)]).permute(0, 3, 1, 2).contiguous().float().div_(255)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (images - mean) / std

    def open_test(self, selection):
        from torchvision.datasets import CIFAR10
        if type(selection) is not FrozenSelection or selection.protocol_sha256 != self.protocol.sha256:
            raise ValueError('Test is sealed until this exact protocol has frozen its final selection')
        source = CIFAR10(str(self.root), train=False, download=False)
        if len(source.targets) != 10000:
            raise ValueError('Expected the complete official test split')
        return np.ascontiguousarray(source.data), np.asarray(source.targets, dtype=np.int64)
