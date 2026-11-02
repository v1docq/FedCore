"""Frozen Evo-only panel and its accounting. Importing never starts a run."""
from dataclasses import dataclass
from .policy import SearchArm


@dataclass(frozen=True)
class ExperimentProtocol:
    model_seeds: tuple[int, ...] = (20261121, 20261122, 20261123)
    search_seeds: tuple[int, ...] = (101, 211, 307)
    arms: tuple[SearchArm, ...] = tuple(SearchArm)
    fit_rows: int = 40000
    calibration_rows: int = 2000
    search_rows: int = 4000
    report_rows: int = 4000
    batch_size: int = 128
    epochs: int = 80
    attempts: int = 64
    population: int = 8
    recovery_updates: int = 200
    proposal_multiplier: int = 10
    controls_per_model: int = 5

    def __post_init__(self):
        if (type(self.model_seeds) is not tuple or type(self.search_seeds) is not tuple
                or not self.model_seeds or not self.search_seeds
                or any(type(s) is not int or s < 0 for s in self.model_seeds + self.search_seeds)
                or len(set(self.model_seeds)) != len(self.model_seeds)
                or len(set(self.search_seeds)) != len(self.search_seeds)
                or type(self.arms) is not tuple or any(type(a) is not SearchArm for a in self.arms)
                or self.arms != tuple(SearchArm)):
            raise ValueError('Require independent seeds and exactly E0/E1/E2; no Random arm')
        for name in ('fit_rows','calibration_rows','search_rows','report_rows','batch_size',
                     'epochs','attempts','population','recovery_updates','proposal_multiplier','controls_per_model'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Positive integer protocol limits are required')
        if ((self.fit_rows,self.calibration_rows,self.search_rows,self.report_rows) != (40000,2000,4000,4000)
                or any(n % 10 for n in (self.fit_rows,self.calibration_rows,self.search_rows,self.report_rows))
                or self.population >= self.attempts):
            raise ValueError('Require a stratified 50000-row panel and population < attempts')

    def budget(self):
        models, repeats, arms = len(self.model_seeds), len(self.search_seeds), len(self.arms)
        searches = models * repeats * arms
        controls = models * self.controls_per_model
        baseline = models * self.epochs * ((self.fit_rows + self.batch_size - 1) // self.batch_size)
        attempts = searches * self.attempts
        search_updates = attempts * self.recovery_updates
        control_updates = controls * self.recovery_updates
        finalists = searches + controls + models
        return dict(searches=searches, charged_attempts=attempts,
            max_proposals=attempts * self.proposal_multiplier, baseline_sgd=baseline,
            search_sgd=search_updates, control_sgd=control_updates,
            total_sgd=baseline + search_updates + control_updates,
            max_finalists=finalists, latency_sessions=finalists * 2 * 2 + 8 * models,
            primary_contrast='E2-E0', secondary_contrasts=('E1-E0','E2-E1'),
            recovery_tape='one tape per M0 shared by all arms, search seeds and controls',
            cluster_launch_authorized=False, main_gpu_seconds=None)

    def jobs(self):
        # Rotate panel order without changing any model/search pairing.
        return tuple((model, seed, self.arms[(i + j + k) % len(self.arms)])
                     for i, model in enumerate(self.model_seeds)
                     for j, seed in enumerate(self.search_seeds)
                     for k in range(len(self.arms)))


def pilot_budget():
    return dict(models=1, recipes=12, max_seconds=7200,
                baseline_sgd=80 * 313, recovery_sgd=12 * 200,
                total_sgd=27440, cluster_launch_authorized=False)
