"""Frozen, bounded first CV/Sparse pilot; no model, data or device effects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


STAGES = (
    ('manifest', 15), ('baseline', 80), ('admission', 15),
    ('controls', 25), ('search', 60), ('recovery', 30),
    ('final', 10), ('close', 5),
)


@dataclass(frozen=True)
class PilotProtocol:
    schema: str = 'fedcore_sparse_cv_a100_pilot_v1'
    model_seed: int = 20261021
    split_seed: int = 990031
    search_seed: int = 31
    recovery_seed: int = 990032
    baseline_epochs: int = 80
    baseline_batch: int = 128
    baseline_lr: float = .1
    momentum: float = .9
    weight_decay: float = .0005
    baseline_min_top1: float = .90
    calibration_rows: int = 1000
    search_rows: int = 5000
    recovery_updates: int = 200
    recovery_batch: int = 128
    recovery_lr: float = .01
    search_evaluations: int = 12
    search_proposals: int = 48
    search_population: int = 4
    search_generations: int = 3
    search_seconds: int = 1800
    storage_fraction: float = .80
    final_quality_drop: float = .02
    max_slot_seconds: int = 14400
    max_vram_bytes: int = 36 * 1024**3
    cpu_threads: int = 4
    aa_pairs: int = 20
    timing_calls: int = 20
    timing_warmup: int = 20
    gpu_dtype: str = 'float32'
    tf32: bool = False
    bn_calibration: bool = True

    def __post_init__(self):
        # This is a named pilot, not a tuning API with silently widened budgets.
        expected = {name: field.default for name, field in self.__dataclass_fields__.items()}
        if any(type(getattr(self, name)) is not type(value) or getattr(self, name) != value
               for name, value in expected.items()):
            raise ValueError('This named pilot has frozen settings; use a separately reviewed protocol for changes')

    def payload(self):
        return {
            **asdict(self),
            'stage_minutes': dict(STAGES),
            'dataset': 'CIFAR-10 official Python archive',
            'roles': {'train': 44000, 'calibration': 1000, 'search_validation': 5000, 'sealed_test': 10000},
            'architecture': 'CIFAR ResNet18: 3x3 stride-1 stem, no maxpool, standard 8 BasicBlocks',
            'choice_menu': ['dense', 'P:0.75', 'P:0.5', 'SVD:0.5', 'SVD:0.25', 'P:0.75->SVD:0.5', 'P:0.75->SVD:0.25'],
            'prefix': ['dense', 'P:0.75', 'SVD:0.5', 'P:0.75->SVD:0.5'],
            'short_fidelity': '0 SGD updates; same image-only BatchNorm calibration for every graph',
            'fitness': ['search_validation_cross_entropy', 'physical_tensor_bytes / dense_tensor_bytes'],
            'short_quality_accepted': 'finite logits/loss and valid independently restored artifact; final quality tolerance is not an early rejection rule',
            'finalist_order': ['minimum short CE with timely completion and bytes <= ceiling', 'bytes', 'canonical genotype id'],
            'random_and_evolution': 'independent empty caches; identical evaluated prefix; distinct uniformly sampled random proposals versus actual FedcoreEvoOptimizer/SPEA2',
            'search_order': ['random', 'evolution'],
            'test_gate': 'test is opened only after selected IDs and all completed recovery artifacts are frozen',
            'final_quality_reference': 'dense-recovery after the same 200 SGD updates and BN calibration',
            'latency': 'report only; not a third search objective or a claim of CUDA sparse kernels',
        }

    @property
    def sha256(self):
        return digest(self.payload())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stage_deadline(started, global_deadline, stage):
    if stage not in dict(STAGES):
        raise ValueError('Unknown pilot stage')
    if not all(type(v) in (float, int) and math.isfinite(v) for v in (started, global_deadline)):
        raise ValueError('Finite monotonic deadlines required')
    return min(global_deadline, started + dict(STAGES)[stage] * 60)


def eligible_finalists(records, *, byte_ceiling):
    """A diagnostic archive must never promote late or over-budget outcomes."""
    if type(byte_ceiling) is not int or byte_ceiling <= 0:
        raise ValueError('Positive physical byte ceiling required')
    accepted = []
    for record in records:
        if (record.get('status') != 'succeeded' or record.get('completed_within_deadline') is not True
                or record.get('quality_accepted') is not True):
            continue
        cost, ce = record.get('tensor_bytes'), record.get('cross_entropy')
        if (type(cost) is not int or not 0 < cost <= byte_ceiling
                or type(ce) not in (float, int) or not math.isfinite(ce)):
            continue
        if type(record.get('solution_id')) is not str or not record['solution_id']:
            continue
        accepted.append(record)
    return sorted(accepted, key=lambda row: (row['cross_entropy'], row['tensor_bytes'], row['solution_id']))


def aa_admission(ratios, orders):
    """Fixed engineering control, never extended after seeing a failure."""
    import statistics
    if len(ratios) != 20 or len(orders) != 20 or orders != ['AB', 'BA'] * 10:
        return {'accepted': False, 'reason': 'incomplete_or_changed_pair_protocol'}
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in ratios):
        return {'accepted': False, 'reason': 'nonfinite_measurement'}
    median = statistics.median(ratios)
    ab = statistics.median(ratios[::2])
    ba = statistics.median(ratios[1::2])
    inside = sum(.85 <= value <= 1.15 for value in ratios)
    return {'accepted': .95 <= median <= 1.05 and inside >= 18 and .9 <= ab / ba <= 1.1,
            'median_ratio': median, 'inside_15_percent': inside, 'ab_ba_median_ratio': ab / ba}
