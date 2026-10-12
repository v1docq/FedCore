"""Pure, immutable RankDyna descriptions and irreversible rank transitions.

No tensor or optimizer objects belong here. A group ID is its original SVD
column, not its current array offset or a claim about a trained spectrum.
"""
from dataclasses import dataclass, replace
import math

from .method_errors import MethodSpecError


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise MethodSpecError(f"{name} must be an integer >= {minimum}")


def _fraction(value, name, *, include_one=True):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (value > 1 if include_one else value >= 1)):
        endpoint = "[0,1]" if include_one else "[0,1)"
        raise MethodSpecError(f"{name} must be finite in {endpoint}")


@dataclass(frozen=True)
class RankDyna:
    """Bounded observed-label adaptation with a retained *group* fraction.

    ``parameter_budget`` is an optional FedCore extension: it counts exported
    selected factor elements, excluding biases and unselected model parameters.
    It is deliberately distinct from the paper's ceil(c * initial_groups).
    """
    updates: int = 1
    seed: int = 0
    target_fraction: float = 0.5
    beta: float = 0.9
    warmup_steps: int = 0
    cooldown_steps: int = 0
    prune_interval: int = 1
    freeze: str = "factors_only"
    loss: str = "mse"
    reduction: str = "mean"
    parameter_budget: int | None = None

    def __post_init__(self):
        _integer(self.updates, "updates", 1)
        _integer(self.seed, "seed")
        if self.seed >= 2**63:
            raise MethodSpecError("seed must fit a signed 64-bit generator seed")
        _fraction(self.target_fraction, "target_fraction")
        _fraction(self.beta, "beta", include_one=False)
        _integer(self.warmup_steps, "warmup_steps")
        _integer(self.cooldown_steps, "cooldown_steps")
        _integer(self.prune_interval, "prune_interval", 1)
        if self.warmup_steps + self.cooldown_steps >= self.updates:
            raise MethodSpecError("RankDyna requires at least one adjustment step")
        if self.freeze not in ("factors_only", "factors_and_bias"):
            raise MethodSpecError("Declare factors_only or factors_and_bias training")
        if self.loss not in ("mse", "cross_entropy") or self.reduction not in ("mean", "sum"):
            raise MethodSpecError("Declare the observed-label loss and reduction")
        if self.parameter_budget is not None:
            _integer(self.parameter_budget, "parameter_budget")


@dataclass(frozen=True, order=True)
class RankDynaID:
    path: str
    initial_column: int

    def __post_init__(self):
        if type(self.path) is not str:
            raise MethodSpecError("A group path must be a string (including the root '')")
        _integer(self.initial_column, "initial_column")


@dataclass(frozen=True)
class RankDynaGroup:
    identity: RankDynaID
    ema: float
    exported_elements: int

    def __post_init__(self):
        if not isinstance(self.identity, RankDynaID):
            raise MethodSpecError("A typed group identity is required")
        if type(self.ema) not in (int, float) or not math.isfinite(self.ema) or self.ema < 0:
            raise MethodSpecError("Group EMA must be finite and nonnegative")
        _integer(self.exported_elements, "exported_elements", 1)


@dataclass(frozen=True)
class RankDynaState:
    groups: tuple[RankDynaGroup, ...]
    initial_groups: int
    initial_exported_elements: int
    completed_updates: int = 0
    removed_ids: tuple[RankDynaID, ...] = ()

    def __post_init__(self):
        if (type(self.groups) is not tuple or type(self.removed_ids) is not tuple
                or any(not isinstance(g, RankDynaGroup) for g in self.groups)
                or any(not isinstance(i, RankDynaID) for i in self.removed_ids)):
            raise MethodSpecError("Rank state requires immutable typed group tuples")
        _integer(self.initial_groups, "initial_groups", 1)
        _integer(self.initial_exported_elements, "initial_exported_elements", 1)
        _integer(self.completed_updates, "completed_updates")
        active = tuple(g.identity for g in self.groups)
        identities = active + self.removed_ids
        if len(set(identities)) != len(identities) or len(identities) != self.initial_groups:
            raise MethodSpecError("Active and removed IDs must partition the initial groups")
        if sum(g.exported_elements for g in self.groups) > self.initial_exported_elements:
            raise MethodSpecError("Active factor cost cannot exceed its initial cost")

    @property
    def active_ids(self):
        return tuple(g.identity for g in self.groups)


@dataclass(frozen=True)
class RankDynaRemoval:
    state: RankDynaState
    removed_ids: tuple[RankDynaID, ...]
    scheduled_group_budget: int
    scheduled_parameter_budget: int | None
    scheduled: bool


def initial_rankdyna_state(layers):
    """``layers`` contains immutable (path, initial_width, out+in) triples."""
    if type(layers) is not tuple or not layers:
        raise MethodSpecError("Explicit immutable layer descriptions are required")
    groups = []
    seen = set()
    for layer in layers:
        if type(layer) is not tuple or len(layer) != 3:
            raise MethodSpecError("Each layer description is (path, width, group_cost)")
        path, width, cost = layer
        if type(path) is not str or path in seen:
            raise MethodSpecError("Layer paths must be unique strings")
        _integer(width, "initial width", 1)
        _integer(cost, "exported group cost", 1)
        seen.add(path)
        groups.extend(RankDynaGroup(RankDynaID(path, i), 0.0, cost) for i in range(width))
    return RankDynaState(tuple(groups), len(groups), sum(g.exported_elements for g in groups))


def update_rankdyna_ema(state, scores, beta):
    """Observe every active group once, without advancing or reactivating it."""
    if not isinstance(state, RankDynaState):
        raise MethodSpecError("Typed rank state required")
    _fraction(beta, "beta", include_one=False)
    if (type(scores) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                                       or not isinstance(pair[0], RankDynaID) for pair in scores)
            or len(scores) != len(state.groups)):
        raise MethodSpecError("Exactly one immutable score pair per active ID is required")
    values = dict(scores)
    if len(values) != len(scores) or set(values) != set(state.active_ids):
        raise MethodSpecError("Scores must cover active IDs exactly; removed IDs cannot return")
    for score in values.values():
        if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
            raise MethodSpecError("Importance scores must be finite and nonnegative")
    return replace(state, groups=tuple(replace(g, ema=beta*g.ema+(1-beta)*values[g.identity])
                                       for g in state.groups))


def _cubic_budget(initial, target, step, spec):
    if step <= spec.warmup_steps:
        return initial
    end = spec.updates - spec.cooldown_steps
    if step >= end:
        return target
    # Phase boundaries follow the prose of section 3.5. The printed Eq. 12
    # has (t-t0-t1) in its numerator, which does not reach those boundaries.
    progress = (step-spec.warmup_steps)/(end-spec.warmup_steps)
    return target + math.ceil((initial-target)*(1-progress)**3)


def rankdyna_group_budget(initial_groups, step, spec):
    if not isinstance(spec, RankDyna):
        raise MethodSpecError("RankDyna description required")
    _integer(initial_groups, "initial_groups", 1)
    _integer(step, "step")
    if step > spec.updates:
        raise MethodSpecError("Step exceeds the declared training horizon")
    target = math.ceil(spec.target_fraction*initial_groups)
    return _cubic_budget(initial_groups, target, step, spec)


def advance_rankdyna(state, spec):
    """Select globally by EMA, retain original order, and remove irreversibly.

    Ties retain the lexicographically first (path, original-column) IDs. The
    optional cost cap removes the least important retained groups greedily;
    this extension makes no knapsack-optimality claim.
    """
    if not isinstance(state, RankDynaState) or not isinstance(spec, RankDyna):
        raise MethodSpecError("Typed state and RankDyna description required")
    step = state.completed_updates + 1
    group_budget = rankdyna_group_budget(state.initial_groups, step, spec)
    cost_budget = None if spec.parameter_budget is None else _cubic_budget(
        state.initial_exported_elements, min(spec.parameter_budget, state.initial_exported_elements), step, spec)
    boundary = spec.updates-spec.cooldown_steps
    scheduled = (step > spec.warmup_steps and
                 (step % spec.prune_interval == 0 or step == boundary or step == spec.updates))
    ordered = sorted(state.groups, key=lambda group: (-group.ema, group.identity))
    kept = ordered[:group_budget] if scheduled else ordered
    if scheduled and cost_budget is not None:
        cost = sum(g.exported_elements for g in kept)
        while kept and cost > cost_budget:
            cost -= kept.pop().exported_elements
    keep_ids = {g.identity for g in kept}
    removed = tuple(g.identity for g in state.groups if g.identity not in keep_ids)
    result = replace(state, groups=tuple(g for g in state.groups if g.identity in keep_ids),
                     completed_updates=step, removed_ids=state.removed_ids+removed)
    return RankDynaRemoval(result, removed, group_budget, cost_budget, scheduled)
