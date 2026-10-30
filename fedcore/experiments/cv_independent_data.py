"""Independent CIFAR roles and explicit recovery tapes; imports ML only on use."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .cv_independent_protocol import GlobalSeal, IndependentProtocol, ModelSeeds, digest


@dataclass(frozen=True)
class DataRoles:
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    search_validation: tuple[int, ...]
    report_validation: tuple[int, ...]
    seed: int

    def __post_init__(self):
        groups = (self.train, self.calibration, self.search_validation, self.report_validation)
        if ([len(g) for g in groups] != [40000, 1000, 4000, 5000]
                or type(self.seed) is not int or self.seed != 140100
                or any(type(x) is not int for g in groups for x in g)
                or len(set(x for g in groups for x in g)) != 50000
                or set(x for g in groups for x in g) != set(range(50000))):
            raise ValueError('Four disjoint roles must partition all 50000 official training rows')

    def payload(self):
        return asdict(self)

    @property
    def sha256(self):
        return digest(self.payload())


def split_roles(labels, *, seed=140100):
    import numpy as np
    values = np.asarray(labels)
    if (values.shape != (50000,) or values.dtype.kind not in 'iu'
            or type(seed) is not int or seed != 140100
            or not np.array_equal(np.unique(values), np.arange(10))):
        raise ValueError('Exact CIFAR10 labels and frozen split seed required')
    rng = np.random.default_rng(seed)
    train, calibration, search, report = [], [], [], []
    for label in range(10):
        ids = np.flatnonzero(values == label)
        if len(ids) != 5000:
            raise ValueError('CIFAR10 must have 5000 training rows per class')
        ids = rng.permutation(ids).tolist()
        calibration.extend(ids[:100])
        search.extend(ids[100:500])
        report.extend(ids[500:1000])
        train.extend(ids[1000:])
    return DataRoles(*(tuple(sorted(g)) for g in (train, calibration, search, report)), seed)


@dataclass(frozen=True)
class RecoveryBatch:
    row_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    flips: tuple[bool, ...]

    def __post_init__(self):
        if (len(self.row_ids) != 128 or len(self.offsets) != 128 or len(self.flips) != 128
                or any(type(x) is not int or not 0 <= x < 50000 for x in self.row_ids)
                or any(len(pair) != 2 or any(type(x) is not int or not 0 <= x <= 8 for x in pair)
                       for pair in self.offsets)
                or any(type(x) is not bool for x in self.flips)):
            raise ValueError('A recovery batch has 128 explicit IDs, crop offsets and flips')


def recovery_tape(indices, seeds: ModelSeeds):
    import numpy as np
    if len(indices) != 40000 or len(set(indices)) != 40000:
        raise ValueError('The complete distinct train role is required')
    rng = np.random.default_rng(seeds.recovery)
    ordered = rng.permutation(np.asarray(indices, dtype=np.int64))[:200*128]
    batches = []
    for start in range(0, len(ordered), 128):
        offsets = tuple(tuple(int(x) for x in pair) for pair in rng.integers(0, 9, (128, 2)))
        flips = tuple(bool(x) for x in rng.random(128) < .5)
        batches.append(RecoveryBatch(tuple(int(x) for x in ordered[start:start+128]), offsets, flips))
    return tuple(batches)


def tape_payload(tape):
    return [asdict(batch) for batch in tape]


class IndependentCifarData:
    mean = (.4914, .4822, .4465)
    std = (.2023, .1994, .2010)

    def __init__(self, root, protocol=IndependentProtocol()):
        import numpy as np
        from torchvision.datasets import CIFAR10
        from .cv_sparse_runtime import sha256_file
        self.root, self.protocol = Path(root), protocol
        source = CIFAR10(str(self.root), train=True, download=False)
        self.images = np.ascontiguousarray(source.data)
        self.labels = np.asarray(source.targets, dtype=np.int64)
        self.roles = split_roles(self.labels, seed=protocol.split_seed)
        self.source_hashes = {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                              for p in sorted((self.root / 'cifar-10-batches-py').iterdir()) if p.is_file()}

    def rows(self, indices):
        import numpy as np
        import torch
        images = torch.from_numpy(self.images[np.asarray(indices)]).permute(0, 3, 1, 2).contiguous().float().div_(255)
        return (images - torch.tensor(self.mean).view(1, 3, 1, 1)) / torch.tensor(self.std).view(1, 3, 1, 1)

    @property
    def identity(self):
        return digest({'roles': self.roles.sha256, 'source_files': self.source_hashes,
                       'mean': self.mean, 'std': self.std})

    def _admit_seal(self, seal):
        if (type(seal) is not GlobalSeal or seal.protocol_sha256 != self.protocol.sha256
                or seal.data_sha256 != self.identity):
            raise ValueError('Vreport/test require the complete matching global seal')

    def open_report(self, seal):
        self._admit_seal(seal)
        return self.roles.report_validation

    def open_test(self, seal):
        self._admit_seal(seal)
        import numpy as np
        from torchvision.datasets import CIFAR10
        source = CIFAR10(str(self.root), train=False, download=False)
        if len(source.targets) != 10000:
            raise ValueError('Expected exactly 10000 official repeated-test rows')
        return np.ascontiguousarray(source.data), np.asarray(source.targets, dtype=np.int64)
