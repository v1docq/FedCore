"""Preparation-only multi-M0 comparison. No training, data IO or execution CLI."""
from dataclasses import dataclass
import hashlib
import json
import random

@dataclass(frozen=True)
class ComparisonJob:
    model_seed: int
    search_seed: int
    space: str
    optimizer: str

@dataclass(frozen=True)
class ComparisonPlan:
    model_seeds: tuple = (20261101, 20261102, 20261103)
    search_seeds: tuple = (101, 211, 307, 401, 503)
    evaluations: int = 24
    population: int = 8
    search_seconds: int = 600
    pick_seconds: int = 300

    def __post_init__(self):
        for seeds in (self.model_seeds, self.search_seeds):
            if type(seeds) is not tuple or not seeds or len(set(seeds)) != len(seeds):
                raise ValueError('Distinct immutable seed lists required')
            if any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
                raise ValueError('Seeds must be nonnegative 63-bit integers')
        for value in (self.evaluations, self.population, self.search_seconds, self.pick_seconds):
            if type(value) is not int or value <= 0:
                raise ValueError('Positive integer budgets required')
        if self.population > self.evaluations:
            raise ValueError('Population exceeds charged budget')

    def jobs(self):
        result = []
        for model_seed in self.model_seeds:
            for search_seed in self.search_seeds:
                block = [ComparisonJob(model_seed, search_seed, space, optimizer)
                         for space in ('svd_only', 'heterogeneous') for optimizer in ('random', 'evo')]
                # Stable independent block order; never consume proposal RNG here.
                key = f'order:{model_seed}:{search_seed}'.encode()
                random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big')).shuffle(block)
                result.extend(block)
        return tuple(result)

    def preparation_manifest(self):
        from dataclasses import asdict
        return {'schema': 'fashion_multim0_preparation_v1', 'execution_authorized': False,
                'cluster_allowed': False, 'configuration': asdict(self),
                'jobs': [asdict(job) for job in self.jobs()],
                'maximum_search_and_pick_seconds': len(self.jobs()) * (self.search_seconds + self.pick_seconds),
                'data_status': 'exploratory only; independent confirmation dataset not assigned',
                'required_before_execution': ['fresh resource authorization', 'source and data hashes',
                    'locked partitions and analysis', 'guarded runner integration and boundary tests']}


def search_domain(plan, space):
    """Supply finite alternatives and prefix policy to the existing grammar.

    SVD-only intentionally excludes factor quantization, PTQ prefixes and QAT.
    No global monkeypatching of the historical Fashion worker is used.
    """
    from .fashion_worker import alternatives, recipe
    from .fashion_protocol import PATHS
    if space == 'heterogeneous':
        return alternatives(plan), (2, 4)
    if space != 'svd_only':
        raise ValueError('Unknown search space')
    choices = {path: tuple([recipe('dense')] + [recipe('svd', rank)
               for rank in plan.payload['search']['ranks'][index]])
               for index, path in enumerate(PATHS)}
    return choices, ()


def write_preparation(path, plan=ComparisonPlan()):
    """Write a plan only; deliberately no run or resume command."""
    from pathlib import Path
    target = Path(path)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan.preparation_manifest(), stream, indent=2)

