"""Finite integer choices under an exact unique-storage budget.

Scores are supplied by existing evaluators. A sum of local scores is explicitly
an additive surrogate; it is not a promise about joint model quality/latency.
Related decisions must be combined into one candidate table before allocation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
import math


@dataclass(frozen=True)
class StorageCost:
    storage_id: str
    numel: int
    bytes_per_element: int = 4
    category: str = "factor"


@dataclass(frozen=True)
class ScoreObjective:
    name: str
    direction: str = "minimize"
    semantics: str = "additive_surrogate"


@dataclass(frozen=True)
class CandidateOption:
    option_id: str
    ranks: tuple[int, ...]
    score: float
    storages: tuple[StorageCost, ...]


@dataclass(frozen=True)
class CandidateTable:
    table_id: str
    options: tuple[CandidateOption, ...]
    target_paths: tuple[str, ...] = ()
    maximum_ranks: tuple[int, ...] = ()
    rank_multiples: tuple[int, ...] = ()


@dataclass(frozen=True)
class CandidateSelection:
    table_id: str
    option: CandidateOption


@dataclass(frozen=True)
class Allocation:
    selected: tuple[CandidateSelection, ...]
    total_cost: int
    objective_score: float
    objective: ScoreObjective
    unit: str
    algorithm: str


@dataclass(frozen=True)
class BudgetInfeasible:
    budget: int
    minimum_cost: int
    unit: str
    reason: str = "No admissible integer combination fits the budget"


def _storage_signature(storage):
    if (not isinstance(storage, StorageCost) or not isinstance(storage.storage_id, str) or not storage.storage_id
            or type(storage.numel) is not int or storage.numel < 0
            or type(storage.bytes_per_element) is not int or storage.bytes_per_element <= 0
            or not isinstance(storage.category, str) or not storage.category):
        raise ValueError("Storage requires a stable id, nonnegative integer size, precision and category")
    return storage.numel, storage.bytes_per_element, storage.category


def exact_unique_cost(storages, unit="parameters"):
    """Count aliases once; inconsistent descriptions of one storage are errors.

    Bias, untouched tensors, maps, shared bases and residuals are represented by
    explicit records just like factors. Byte cost uses each record's precision.
    In parameter mode every declared stored element counts, including maps.
    """
    if unit not in ("parameters", "bytes"):
        raise ValueError("Cost unit must be parameters or bytes")
    unique = {}
    for storage in storages:
        signature = _storage_signature(storage)
        if storage.storage_id in unique and unique[storage.storage_id] != signature:
            raise ValueError("Conflicting descriptions of shared storage")
        unique[storage.storage_id] = signature
    return sum(elements * (precision if unit == "bytes" else 1) for elements, precision, _ in unique.values())


def _validate(tables, objective, budget, fixed_storages, unit):
    if type(budget) is not int or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    if (not isinstance(objective, ScoreObjective) or not isinstance(objective.name, str) or not objective.name
            or objective.direction not in ("minimize", "maximize")
            or objective.semantics != "additive_surrogate"):
        raise ValueError("Name an additive-surrogate objective and its direction")
    if len({table.table_id for table in tables}) != len(tables):
        raise ValueError("Candidate table identifiers must be unique")
    paths = [path for table in tables for path in table.target_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Coupled/overlapping targets must be grouped before allocation")
    storages = list(fixed_storages)
    for table in tables:
        if not table.table_id or not table.options:
            raise ValueError("Each named table requires at least one admissible option")
        if len({option.option_id for option in table.options}) != len(table.options):
            raise ValueError("Option identifiers must be unique within a table")
        if any(type(rank) is not int or rank <= 0 for rank in (*table.maximum_ranks, *table.rank_multiples)):
            raise ValueError("Shape maxima and hardware grid must be positive integers")
        for option in table.options:
            if not isinstance(option.option_id, str) or not option.option_id or not isinstance(option.ranks, tuple) or not option.ranks or any(type(rank) is not int or rank <= 0 for rank in option.ranks):
                raise ValueError("Options must carry admissible positive integer structures")
            if table.maximum_ranks and (len(table.maximum_ranks) != len(option.ranks) or any(rank > maximum for rank, maximum in zip(option.ranks, table.maximum_ranks))):
                raise ValueError("Option rank exceeds its declared operator shape")
            if table.rank_multiples and (len(table.rank_multiples) != len(option.ranks) or any(rank % multiple for rank, multiple in zip(option.ranks, table.rank_multiples))):
                raise ValueError("Option rank does not fit the declared hardware grid")
            if type(option.score) not in (int, float) or not math.isfinite(option.score):
                raise ValueError("Scores must be finite")
            storages.extend(option.storages)
    exact_unique_cost(storages, unit)  # also validates identity across alternatives


def _canonical(tables):
    if any(not isinstance(table, CandidateTable) for table in tables):
        raise ValueError("Finite CandidateTable values required")
    if any(not isinstance(option, CandidateOption) for table in tables for option in table.options):
        raise ValueError("Finite CandidateOption values required")
    return tuple(replace(table, options=tuple(sorted(table.options, key=lambda option: option.option_id)))
                 for table in sorted(tables, key=lambda table: table.table_id))


def _key(score, cost, choices, objective):
    return (score if objective.direction == "minimize" else -score,
            cost, tuple(option.option_id for option in choices))


def _exact_score(choices):
    # Float rounding must not make a discarded DP prefix become preferable
    # after adding a later score (e.g. 1e16 + 1 - 1e16). No score is rounded.
    return sum((Fraction(option.score) for option in choices), Fraction())


def exhaustive_allocate(tables, budget, *, objective, fixed_storages=(), unit="parameters", max_combinations=100000):
    """Small independent oracle, including arbitrary shared-storage choices."""
    tables, fixed_storages = _canonical(tuple(tables)), tuple(fixed_storages)
    _validate(tables, objective, budget, fixed_storages, unit)
    if type(max_combinations) is not int or max_combinations <= 0:
        raise ValueError("Oracle combination limit must be a positive integer")
    if math.prod(len(table.options) for table in tables) > max_combinations:
        raise ValueError("Exhaustive oracle combination limit exceeded")
    best, minimum = None, None
    for choices in product(*(table.options for table in tables)):
        cost = exact_unique_cost((*fixed_storages, *(storage for option in choices for storage in option.storages)), unit)
        minimum = cost if minimum is None else min(minimum, cost)
        score = math.fsum(option.score for option in choices)
        key = _key(_exact_score(choices), cost, choices, objective)
        if cost <= budget and (best is None or key < best[0]):
            best = (key, choices, cost, score)
    if best is None:
        return BudgetInfeasible(budget, minimum if minimum is not None else exact_unique_cost(fixed_storages, unit), unit)
    _, choices, cost, score = best
    return Allocation(tuple(CandidateSelection(table.table_id, option) for table, option in zip(tables, choices)),
                      cost, score, objective, unit, "exhaustive")


def _has_cross_table_sharing(tables, fixed_storages):
    fixed = {storage.storage_id for storage in fixed_storages}
    owners = {}
    for table in tables:
        for option in table.options:
            for storage in option.storages:
                if storage.storage_id in fixed:
                    continue
                if storage.storage_id in owners and owners[storage.storage_id] != table.table_id:
                    return True
                owners[storage.storage_id] = table.table_id
    return False


def allocate(tables, budget, *, objective, fixed_storages=(), unit="parameters", max_shared_combinations=100000):
    """Exact deterministic sparse DP for additive independent cost tables.

    Shared choices use the bounded exhaustive oracle. Large related tables must
    be grouped by the caller; no unreported greedy approximation is introduced.
    The sparse DP retains one best prefix per exact cost and never rounds ranks
    or relaxes the budget. Ties prefer lower cost, then lexical table/option ids.
    """
    tables, fixed_storages = _canonical(tuple(tables)), tuple(fixed_storages)
    _validate(tables, objective, budget, fixed_storages, unit)
    if _has_cross_table_sharing(tables, fixed_storages):
        return exhaustive_allocate(tables, budget, objective=objective, fixed_storages=fixed_storages,
                                   unit=unit, max_combinations=max_shared_combinations)
    base = exact_unique_cost(fixed_storages, unit)
    fixed_ids = {storage.storage_id for storage in fixed_storages}
    costs = {table.table_id: {option.option_id: exact_unique_cost(tuple(storage for storage in option.storages if storage.storage_id not in fixed_ids), unit)
                             for option in table.options} for table in tables}
    minimum = base + sum(min(costs[table.table_id].values()) for table in tables)
    if minimum > budget:
        return BudgetInfeasible(budget, minimum, unit)
    # Prefix decisions compare exact rational values of supplied float scores;
    # the final reported score is rounded once with fsum, as in the oracle.
    states = {base: ()}
    for table in tables:
        next_states = {}
        for previous_cost, choices in states.items():
            for option in table.options:
                cost = previous_cost + costs[table.table_id][option.option_id]
                if cost > budget:
                    continue
                candidate = choices + (option,)
                incumbent = next_states.get(cost)
                if incumbent is None or _key(_exact_score(candidate), cost, candidate, objective) < _key(_exact_score(incumbent), cost, incumbent, objective):
                    next_states[cost] = candidate
        states = next_states
    cost, choices = min(states.items(), key=lambda item: _key(_exact_score(item[1]), item[0], item[1], objective))
    score = math.fsum(option.score for option in choices)
    # Independent recomputation protects changes to the DP's accounting.
    exact_cost = exact_unique_cost((*fixed_storages, *(storage for option in choices for storage in option.storages)), unit)
    if exact_cost != cost or exact_cost > budget:
        raise RuntimeError("Allocation accounting invariant violated")
    return Allocation(tuple(CandidateSelection(table.table_id, option) for table, option in zip(tables, choices)),
                      exact_cost, score, objective, unit, "additive_dynamic_program")
