"""Bounded domain variation; no graph execution, model calls or global RNG."""
from dataclasses import dataclass, replace
from enum import Enum
import math
import random


class SearchArm(str, Enum):
    BASIC = 'E0'
    EXTENDED = 'E1'
    STAGED = 'E2'


class Action(str, Enum):
    LOCAL = 'local'
    RECIPE = 'recipe'
    GROUP = 'group'
    BUDGET = 'budget_transfer'


@dataclass(frozen=True)
class PolicyDecision:
    stage: str
    weights: tuple[tuple[Action, float], ...]


def _limits(charged, maximum, initial):
    if (any(type(x) is not int for x in (charged, maximum, initial))
            or initial < 1 or maximum <= initial or not 0 <= charged <= maximum):
        raise ValueError('Require 0 <= charged <= maximum and 1 <= initial < maximum')


def _stage(index, remaining):
    if index * 10 < 3 * remaining:
        return PolicyDecision('explore', ((Action.LOCAL, .10), (Action.RECIPE, .25),
                                        (Action.GROUP, .45), (Action.BUDGET, .20)))
    if index * 10 < 7 * remaining:
        return PolicyDecision('joint', ((Action.LOCAL, .35), (Action.RECIPE, .20),
                                      (Action.GROUP, .25), (Action.BUDGET, .20)))
    return PolicyDecision('refine', ((Action.LOCAL, .70), (Action.RECIPE, .10),
                                   (Action.GROUP, .10), (Action.BUDGET, .10)))


def policy_for(arm, *, charged, maximum, initial=8):
    """E1 averages the exact discrete E2 schedule after the shared prefix."""
    if type(arm) is not SearchArm:
        raise ValueError('Select an explicit E0/E1/E2 search arm')
    _limits(charged, maximum, initial)
    if arm is SearchArm.BASIC:
        return PolicyDecision('fixed_basic', ((Action.LOCAL, 1.),))
    remaining = maximum - initial
    if arm is SearchArm.STAGED:
        return _stage(min(max(charged - initial, 0), remaining - 1), remaining)
    first, second = (3 * remaining + 9) // 10, (7 * remaining + 9) // 10
    counts = (first, second - first, remaining - second)
    stages = (_stage(0, remaining), _stage(first, remaining), _stage(second, remaining))
    means = tuple((action, math.fsum(count * dict(stage.weights)[action]
                    for count, stage in zip(counts, stages)) / remaining) for action in Action)
    return PolicyDecision('fixed_extended', means)


def _family(recipe):
    return (recipe.keep_channels is not None, recipe.rank is not None, recipe.ptq)


def _proxy(recipe, path):
    # A local, additive weight-count surrogate, never deployment bytes or latency.
    width = 64 * 2 ** (int(path[5]) - 1)
    incoming = width // 2 if int(path[5]) > 1 and path.endswith('.0') else width
    keep = recipe.keep_channels or width
    conv2 = 9 * keep * width if recipe.rank is None else recipe.rank * (9 * keep + width)
    return (9 * keep * incoming + conv2) * (.25 if recipe.ptq else 1.)


def _moves(genotype, space):
    return tuple((i, alternative) for i, gene in enumerate(genotype.genes)
                 for alternative in space.domains[i].alternatives if alternative != gene.recipe)


def mutate_with_policy(genotype, space, *, arm, seed, charged, maximum, initial=8):
    """Generate only whole compatible regions; return an atomic planner decision.

    Budget transfer uses a weight-count proxy. Final compiled deployment cost is
    still measured by the evaluator. The bounded G1 domain has at most 8*64 moves.
    """
    from .cv import normalize_genotype
    from fedcore.algorithm.low_rank.plans import ValidationFailure
    from fedcore.algorithm.low_rank.grammar_search import GrammarMutationDecision
    parent = normalize_genotype(genotype, space)
    if type(parent) is ValidationFailure:
        return parent
    if type(seed) is not int or seed < 0:
        raise ValueError('A nonnegative local seed is required')
    policy = policy_for(arm, charged=charged, maximum=maximum, initial=initial)
    rng, genes = random.Random(seed), list(parent.genes)
    moves = _moves(parent, space)
    recipe_moves = tuple((i, alt) for i, alt in moves if _family(alt) != _family(genes[i].recipe))
    mutable = tuple(sorted({i for i, _ in moves}))
    down, up = [], []
    if arm is not SearchArm.BASIC:
        for i, alt in moves:
            change = _proxy(alt, genes[i].path) - _proxy(genes[i].recipe, genes[i].path)
            if change < 0: down.append((i, alt))
            elif change > 0: up.append((i, alt))
    # Enumerate at most 512^2 tiny metadata pairs; no tensor or solver work.
    transfers = tuple((a, b) for a in down for b in up if a[0] != b[0])
    enabled = {Action.LOCAL: bool(moves), Action.RECIPE: bool(recipe_moves),
               Action.GROUP: len(mutable) >= 2, Action.BUDGET: bool(transfers)}
    available = tuple((action, weight) for action, weight in policy.weights if enabled[action])
    if not available:
        return GrammarMutationDecision(parent, 'no_admissible_move', (('stage', policy.stage),))
    action = rng.choices([a for a, _ in available], [w for _, w in available], k=1)[0]
    if action is Action.LOCAL:
        selected = (rng.choice(moves),)
    elif action is Action.RECIPE:
        selected = (rng.choice(recipe_moves),)
    elif action is Action.GROUP:
        start = rng.randrange(len(mutable) - 1)
        indices = mutable[start:start + rng.randint(2, min(3, len(mutable) - start))]
        selected = tuple((i, rng.choice([alt for index, alt in moves if index == i])) for i in indices)
    else:
        selected = rng.choice(transfers)
    for index, alternative in selected:
        genes[index] = replace(genes[index], recipe=alternative)
    child = normalize_genotype(replace(parent, genes=tuple(genes)), space)
    total = math.fsum(w for _, w in available)
    details = (('stage', policy.stage), ('charged', charged), ('changed_scopes', len(selected)))
    details += tuple(('p_' + a.value, w / total) for a, w in available)
    details += tuple(('configured_p_' + a.value, w) for a, w in policy.weights)
    return GrammarMutationDecision(child, action.value, details)
