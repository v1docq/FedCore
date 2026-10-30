"""Frozen independent-M0 comparison: pure plans, charging and sealing contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


STAGES = (('manifest', 20), ('baseline', 150), ('controls', 10),
          ('search', 216), ('final', 20), ('timing', 50), ('close', 14))


@dataclass(frozen=True)
class ModelSeeds:
    name: str
    initialization: int
    train_order: int
    augmentation: int
    recovery: int


MODELS = tuple(ModelSeeds(f'M{i}', 140100+i, 140110+i, 140120+i, 140300+i)
               for i in (1, 2, 3))
SEARCH_SEEDS = (140201, 140202, 140203)
WINDOW_SEEDS = (140401, 140402, 140403)


class Strategy(str, Enum):
    RANDOM = 'random'
    EVOLUTION = 'evolution'


@dataclass(frozen=True)
class SearchJob:
    model: str
    seed: int
    strategy: Strategy

    @property
    def name(self):
        return f'{self.model}_{self.seed}_{self.strategy.value}'


def search_jobs():
    return tuple(SearchJob(model.name, seed, strategy)
                 for m, model in enumerate(MODELS)
                 for s, seed in enumerate(SEARCH_SEEDS)
                 for strategy in ((Strategy.RANDOM, Strategy.EVOLUTION) if (m+s) % 2 == 0
                                  else (Strategy.EVOLUTION, Strategy.RANDOM)))


@dataclass(frozen=True)
class IndependentProtocol:
    schema: str = 'fedcore_sparse_cv_a100_independent_m0_v1'
    split_seed: int = 140100
    baseline_epochs: int = 80
    baseline_batch: int = 128
    baseline_lr: float = .1
    baseline_min_top1: float = .90
    momentum: float = .9
    weight_decay: float = .0005
    recovery_updates: int = 200
    recovery_batch: int = 128
    recovery_lr: float = .001
    search_evaluations: int = 32
    search_proposals: int = 256
    search_population: int = 4
    search_generations: int = 32
    search_seconds: int = 720
    baseline_seconds: int = 3000
    timing_window_seconds: int = 960
    storage_fraction: float = .80
    final_quality_drop: float = .02
    max_slot_seconds: int = 28800
    worker_seconds: int = 28790
    max_vram_bytes: int = 16 * 1024**3
    max_rss_bytes: int = 32 * 1024**3
    minimum_disk_bytes: int = 100 * 1024**3
    cpu_threads: int = 4
    aa_pairs: int = 20
    timing_calls: int = 20
    timing_warmup: int = 20
    gpu_dtype: str = 'float32'
    tf32: bool = False
    bn_calibration: bool = False

    def __post_init__(self):
        if any(type(getattr(self, name)) is not type(field.default)
               or getattr(self, name) != field.default
               for name, field in self.__dataclass_fields__.items()):
            raise ValueError('Named independent-M0 protocol settings are frozen')

    def payload(self):
        return {**asdict(self), 'stage_minutes': dict(STAGES),
                'model_seeds': [asdict(model) for model in MODELS],
                'search_jobs': [dict(model=j.model, seed=j.seed, strategy=j.strategy.value,
                                     name=j.name) for j in search_jobs()],
                'window_seeds': list(WINDOW_SEEDS),
                'roles': {'train': 40000, 'calibration': 1000, 'Vsearch': 4000,
                          'Vreport': 5000, 'official_repeated_test': 10000},
                'epoch_seed_derivation': 'SHA256 JSON [domain, base_seed, epoch, batch]; first 8 hex digits',
                'recovery_tape': 'NumPy PCG64; one train permutation, explicit per-image crop offsets and flips',
                'recovery': 'fresh SGD; 200 updates; inherited BN; no extra BN reset/recalibration',
                'grammar_recovery': 'NoRecovery; caller_owned_recovery is mandatory',
                'fitness': ['Vsearch_cross_entropy', 'whole_model_physical_bytes / own_M0_bytes'],
                'prefix_menu_indices': [0, 1, 3, 5],
                'search_cache': 'empty independent process for every search; no factors/weights/fitness reuse',
                'finalist': 'timely successful 200-update minimum CE; then bytes; then canonical ID; bytes <= floor(.8*M0)',
                'global_seal': 'all 18 outcomes, finalist IDs, all final checkpoint hashes before Vreport/test',
                'Vreport_equivalence': {'atol': 1e-5, 'rtol': 1e-5, 'exact_ids_labels_classes': True},
                'test_role': 'descriptive repeated evaluation; historically used, not untouched',
                'timing': '3 fresh processes; initial dense AA, own candidate AA, dense pre AA, AB, dense post AA',
                'counts': count_limits()}

    @property
    def sha256(self):
        return digest(self.payload())


def count_limits():
    return {'baseline_updates': 3 * 80 * math.ceil(40000 / 128),
            'search_attempts': 18 * 32, 'search_proposals': 18 * 256,
            'search_updates': 18 * 32 * 200, 'control_updates': 12 * 200,
            'recovery_updates': (18 * 32 + 12) * 200,
            'total_updates': 3 * 80 * math.ceil(40000 / 128) + (18 * 32 + 12) * 200,
            'Vsearch_rows': (18 * 32 + 15) * 4000,
            'final_roles': 33, 'Vreport_eager_rows': 33 * 5000,
            'Vreport_ts_rows': 33 * 5000, 'test_ts_rows': 33 * 10000,
            'timing_sessions': 3 * 3 * 21, 'timing_pairs': 3 * 3 * 21 * 40}


def derived_seed(domain, base, epoch, batch=0):
    if (type(domain) is not str or not domain
            or any(type(x) is not int or x < 0 for x in (base, epoch, batch))):
        raise ValueError('Explicit seed domain and nonnegative integer coordinates required')
    return int(digest([domain, base, epoch, batch])[:8], 16)


def stage_deadline(started, absolute, stage):
    if stage not in dict(STAGES) or any(type(x) not in (int, float) or not math.isfinite(x)
                                       for x in (started, absolute)):
        raise ValueError('Known stage and finite monotonic deadlines required')
    return min(absolute, started + dict(STAGES)[stage] * 60)


class Reservation(str, Enum):
    CHARGED = 'charged'
    DUPLICATE = 'duplicate'
    STOPPED = 'stopped'


@dataclass(frozen=True)
class AttemptBudget:
    """Failures remain charged; a duplicate consumes only a proposal."""
    deadline: float
    proposed: int = 0
    identities: tuple[str, ...] = ()

    def __post_init__(self):
        if (type(self.deadline) not in (int, float) or not math.isfinite(self.deadline)
                or type(self.proposed) is not int or not len(self.identities) <= self.proposed <= 256
                or len(self.identities) > 32 or len(set(self.identities)) != len(self.identities)
                or any(type(x) is not str or not x for x in self.identities)):
            raise ValueError('Finite deadline and consistent bounded charging state required')

    def reserve(self, identity, now):
        if (type(identity) is not str or not identity or type(now) not in (int, float)
                or not math.isfinite(now)):
            raise ValueError('Canonical identity and finite clock required')
        if now >= self.deadline or self.proposed >= 256 or len(self.identities) >= 32:
            return self, Reservation.STOPPED
        state = replace(self, proposed=self.proposed + 1)
        if identity in self.identities:
            return state, Reservation.DUPLICATE
        return replace(state, identities=(*self.identities, identity)), Reservation.CHARGED


def eligible_finalists(records, *, byte_ceiling):
    if type(byte_ceiling) is not int or byte_ceiling < 1:
        raise ValueError('Positive whole-model byte ceiling required')
    return sorted((r for r in records
                   if r.get('status') == 'succeeded' and r.get('completed_within_deadline') is True
                   and r.get('quality_accepted') is True and r.get('recovery_updates') == 200
                   and type(r.get('tensor_bytes')) is int and 0 < r['tensor_bytes'] <= byte_ceiling
                   and type(r.get('cross_entropy')) in (int, float) and math.isfinite(r['cross_entropy'])
                   and type(r.get('solution_id')) is str and bool(r['solution_id'])),
                  key=lambda r: (r['cross_entropy'], r['tensor_bytes'], r['solution_id']))


def is_digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class SearchOutcome:
    job: str
    status: str
    charged: int
    finalist: str | None
    complete_q32: bool

    def __post_init__(self):
        if (self.job not in {j.name for j in search_jobs()}
                or self.status not in ('completed', 'partial', 'failed', 'not_started')
                or type(self.charged) is not int or not 0 <= self.charged <= 32
                or self.finalist is not None and not is_digest(self.finalist)
                or type(self.complete_q32) is not bool
                or self.complete_q32 and (self.charged != 32 or self.status != 'completed')
                or self.status == 'not_started' and (self.charged != 0 or self.finalist is not None)):
            raise ValueError('Invalid frozen search outcome')


@dataclass(frozen=True)
class GlobalSeal:
    protocol_sha256: str
    data_sha256: str
    source_sha256: str
    outcomes: tuple[SearchOutcome, ...]
    roles: tuple[tuple[str, str], ...]
    artifacts: tuple[tuple[str, str], ...]

    def __post_init__(self):
        mandatory_roles = {f'{m.name}_{role}' for m in MODELS for role in
                           ('dense_original', 'dense_recovery', 'fixed_prune', 'fixed_svd', 'fixed_hybrid')}
        expected_roles = mandatory_roles | {f'{o.job}_finalist' for o in self.outcomes if o.finalist is not None}
        if (not all(is_digest(x) for x in (self.protocol_sha256, self.data_sha256, self.source_sha256))
                or len(self.outcomes) != 18 or {o.job for o in self.outcomes} != {j.name for j in search_jobs()}
                or not self.roles or len(self.roles) > 33
                or len({r for r, _ in self.roles}) != len(self.roles)
                or {r for r, _ in self.roles} != expected_roles
                or any(type(r) is not str or not r or not is_digest(a) for r, a in self.roles)
                or not self.artifacts or len({a for a, _ in self.artifacts}) != len(self.artifacts)
                or any(not is_digest(a) or not is_digest(h) for a, h in self.artifacts)
                or {a for _, a in self.roles} != {a for a, _ in self.artifacts}
                or any(o.finalist is not None and (f'{o.job}_finalist', o.finalist) not in self.roles
                       for o in self.outcomes)):
            raise ValueError('Seal the entire search panel and every final role/artifact together')

    @property
    def sha256(self):
        return digest(asdict(self))


def primary_aggregate(outcomes):
    by_job = {o.job: o for o in outcomes}
    model_complete = {}
    for model in MODELS:
        required = [j for j in search_jobs() if j.model == model.name]
        model_complete[model.name] = all(j.name in by_job and by_job[j.name].complete_q32
                                        and by_job[j.name].finalist is not None for j in required)
    return {'model_complete': model_complete, 'panel_complete': all(model_complete.values()),
            'independent_trainings': 3}


def timing_comparison_admitted(*, initial_dense, own, dense_pre, dense_post, window_completed):
    return all(value is True for value in (initial_dense, own, dense_pre, dense_post, window_completed))


def timing_session_complete(session):
    try:
        if type(session) is not dict or set(session.get('batches', {})) != {'1', '128'}:
            return False
        for batch in session['batches'].values():
            pairs = batch['pairs']
            if (len(pairs) != 20 or [r['pair'] for r in pairs] != list(range(20))
                    or [r['order'] for r in pairs] != ['AB', 'BA']*10
                    or batch['orders'] != ['AB', 'BA']*10):
                return False
            for pair in pairs:
                for side in ('A', 'B'):
                    for clock in ('cuda_event_seconds', 'synchronized_wall_seconds'):
                        value = pair[side][clock]
                        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                            return False
    except (KeyError, TypeError):
        return False
    return True


def aa_session_admitted(session):
    from .cv_sparse_protocol import aa_admission
    if not timing_session_complete(session) or session.get('aa') is not True:
        return False
    return all(aa_admission([p['B'][clock]/p['A'][clock] for p in batch['pairs']],
                            ['AB', 'BA']*10)['accepted']
               for batch in session['batches'].values()
               for clock in ('cuda_event_seconds', 'synchronized_wall_seconds'))


def finalize_timing_admissions(result, *, completed):
    """Recompute every gate from raw observations after terminal completion."""
    result['window_completed'] = completed is True
    for model in result.get('models', {}).values():
        for candidate in model.get('candidates', {}).values():
            candidate['comparison_admitted'] = bool(
                timing_session_complete(candidate.get('AB'))
                and timing_comparison_admitted(
                    initial_dense=aa_session_admitted(model.get('initial_dense_AA')),
                    own=aa_session_admitted(candidate.get('own_AA')),
                    dense_pre=aa_session_admitted(model.get('dense_pre_AA')),
                    dense_post=aa_session_admitted(model.get('dense_post_AA')),
                    window_completed=completed is True and model.get('status') == 'completed'))
    return result
