"""Evo-only CIFAR training roles; the official test is never constructed."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class DataRoles:
    fit: tuple[int,...]
    calibration: tuple[int,...]
    search: tuple[int,...]
    report: tuple[int,...]
    seed: int

    def __post_init__(self):
        groups = (self.fit,self.calibration,self.search,self.report)
        if (tuple(map(len,groups)) != (40000,2000,4000,4000)
                or any(type(g) is not tuple or g != tuple(sorted(g))
                       or any(type(i) is not int for i in g) for g in groups)
                or set(i for g in groups for i in g) != set(range(50000))
                or len(set(i for g in groups for i in g)) != sum(map(len,groups))
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError('Four disjoint sorted roles must partition the 50000 training rows')

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def split_roles(labels, *, seed=20261102):
    import numpy as np
    values = np.asarray(labels)
    if (values.shape != (50000,) or values.dtype.kind not in 'iu'
            or type(seed) is not int or seed < 0
            or not np.array_equal(np.unique(values),np.arange(10))):
        raise ValueError('Exact balanced CIFAR10 training labels required')
    rng, groups = np.random.default_rng(seed), [[],[],[],[]]
    for label in range(10):
        ids = np.flatnonzero(values == label)
        if len(ids) != 5000: raise ValueError('Require 5000 training examples per class')
        ids = rng.permutation(ids)
        for result, part in zip(groups,np.split(ids,(4000,4200,4600))):
            result.extend(int(i) for i in part)
    return DataRoles(*(tuple(sorted(g)) for g in groups),seed)


class CifarTrainingData:
    mean = (.4914,.4822,.4465)
    std = (.2023,.1994,.2010)

    def __init__(self, root, *, split_seed=20261102):
        import numpy as np
        from torchvision.datasets import CIFAR10
        data = CIFAR10(root,train=True,download=False)
        self.images = np.asarray(data.data)
        self.labels = np.asarray(data.targets,dtype=np.int64)
        if self.images.shape != (50000,32,32,3) or self.images.dtype != np.uint8:
            raise ValueError('Exact CIFAR10 training corpus required')
        self.roles = split_roles(self.labels,seed=split_seed)
        self._fit = frozenset(self.roles.fit)
        digest = hashlib.sha256()
        digest.update(memoryview(np.ascontiguousarray(self.images)).cast('B'))
        digest.update(self.labels.tobytes()); digest.update(self.roles.sha256.encode())
        self.fingerprint = digest.hexdigest()
        self.images.flags.writeable = False
        self.labels.flags.writeable = False

    def _tensor(self, images):
        import numpy as np
        import torch
        tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0,3,1,2).float()/255
        mean = torch.tensor(self.mean).reshape(1,3,1,1)
        std = torch.tensor(self.std).reshape(1,3,1,1)
        return ((tensor-mean)/std).contiguous()

    def training(self, indices, seed):
        import numpy as np
        import torch
        if (not indices or any(type(i) is not int or i not in self._fit for i in indices)
                or type(seed) is not int or seed < 0):
            raise ValueError('Augmented learning may consume fit rows only')
        rng = np.random.default_rng(seed)
        padded = np.pad(self.images[list(indices)],((0,0),(4,4),(4,4),(0,0)))
        offsets = rng.integers(0,9,(len(indices),2))
        images = np.stack([padded[i,y:y+32,x:x+32] for i,(y,x) in enumerate(offsets)])
        flips = rng.random(len(indices)) < .5
        images[flips] = images[flips,:,::-1,:]
        return self._tensor(images),torch.from_numpy(self.labels[list(indices)].copy())

    def calibration_tensor(self):
        return self._tensor(self.images[list(self.roles.calibration)])

    def search_batches(self, batch_size=128):
        import torch
        if type(batch_size) is not int or batch_size < 1: raise ValueError('Positive evaluation batch size required')
        for start in range(0,len(self.roles.search),batch_size):
            ids = list(self.roles.search[start:start+batch_size])
            yield self._tensor(self.images[ids]),torch.from_numpy(self.labels[ids].copy())
