"""Finite graph-bound candidates and budgets on one frozen source snapshot.

This opt-in boundary compares actual replacements in one unregularized input
second-moment objective. Independent tables are explicitly a frozen-snapshot
surrogate, including ASVD alternatives; they are not current-student curves.
Numerical weights remain runtime-owned and selected candidates are replayed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import pickle

import torch
from torch import nn

from .allocation import (Allocation, BudgetInfeasible, CandidateOption, CandidateTable,
                         ScoreObjective, allocate)
from .artifacts import ArtifactStoreError, StatisticRef, StatisticsKey, StatisticsStore
from .candidate_storage import (CandidateStorageError, PhysicalStorage, capture_inventory,
    clone_preserving_storages, inventory_cost, inventory_fingerprint, project_storages, storage_guard, validate_inventory)
from .compression_graph import (ArtifactKind, ArtifactRef, CalibrationRef, FixedCompressionGraph,
    ModelVersion, Recipe, RuntimeOptions, _canonical, validate_fixed_graph)
from .execution import WeightedProfileError, _fingerprint
from .graph_execution import CompressionGraphError, execute_fixed_graph, prepare_fixed_graph, _allocation_failure, _calibration
from .method_execution import _rng_isolated
from .method_specs import ASVD
from .plans import MetricPolicy, OperatorDescriptor, ValidationFailure, Violation, _metadata_only
from .staged_execution import _weighted_matrices, _rng_fingerprint
from .topology import TopologyError, module_paths, replace_modules_atomically


class CandidateGraphError(WeightedProfileError):
    def __init__(self, violations):
        self.violations = tuple(violations)
        super().__init__("; ".join(f"{v.code}: {v.path}: {v.message}" for v in self.violations))


def _fail(code, path, message):
    raise CandidateGraphError((Violation(code, path, message),))


def _digest(value):
    if not _metadata_only(value):
        _fail("LiveResource", "plan", "Plans and catalogs may contain immutable metadata only")
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CandidateRecipe:
    recipe: Recipe
    rank: int
    representation: str = "two_layers"
    policy: MetricPolicy = MetricPolicy()
    method_spec: ASVD | None = None


@dataclass(frozen=True)
class CandidateEvaluationContext:
    source: ModelVersion
    calibration: CalibrationRef
    source_inventory_fingerprint: str
    graph_version: str
    options: RuntimeOptions
    rng_fingerprint: str
    semantics: str = "frozen_snapshot_surrogate"
    score: str = "local_output_squared_l2_per_observation"
    normalization: str = "weight_sum"
    regularization: str = "none"
    bias_policy: str = "preserve"
    coordinates: str = "original_operator_input_output"


@dataclass(frozen=True)
class GraphCandidate:
    """Metric identity is the pair (graph fingerprint, graph-local metric ref)."""
    option_id: str
    recipe: CandidateRecipe
    graph: FixedCompressionGraph
    statistic_ref: StatisticRef
    metric_ref: ArtifactRef
    score_ref: StatisticRef
    score: float
    storages: tuple[PhysicalStorage, ...]

    @property
    def metric_identity(self):
        return self.graph.fingerprint, self.metric_ref


@dataclass(frozen=True)
class GraphCandidateTable:
    table_id: str
    path: str
    aliases: tuple[str, ...]
    operator: OperatorDescriptor
    score_ref: StatisticRef
    candidates: tuple[GraphCandidate, ...]


@dataclass(frozen=True)
class CandidateCatalog:
    context: CandidateEvaluationContext
    source_inventory: tuple[PhysicalStorage, ...]
    fixed_storages: tuple[PhysicalStorage, ...]
    tables: tuple[GraphCandidateTable, ...]
    max_candidates: int
    fingerprint: str


@dataclass(frozen=True)
class CandidateAllocationPlan:
    catalog: CandidateCatalog
    allocation: Allocation
    budget: int
    unit: str
    fingerprint: str


@dataclass(frozen=True)
class CandidateAllocationResult:
    model: nn.Module
    plan: CandidateAllocationPlan
    evidence: dict


@dataclass(frozen=True)
class CandidateExportResult:
    model: nn.Module
    evidence: dict


def _normalize_recipe(spec):
    if type(spec) is not CandidateRecipe or not _metadata_only(spec):
        _fail("InvalidCandidate", "alternatives", "Immutable CandidateRecipe required")
    try:
        recipe = Recipe(spec.recipe)
    except (TypeError, ValueError):
        _fail("InvalidCandidate", "recipe", "Only weighted_svd and asvd candidates are supported")
    if type(spec.rank) is not int or spec.rank <= 0:
        _fail("InvalidRank", "rank", "Each finite candidate needs an explicit positive integer rank")
    return replace(spec, recipe=recipe, method_spec=ASVD() if recipe is Recipe.ASVD and spec.method_spec is None else spec.method_spec)


def _graph(model, data, context, path, spec):
    return prepare_fixed_graph(model, data, recipe=spec.recipe, method_spec=spec.method_spec,
        rank=spec.rank, policy=spec.policy, representation=spec.representation, target_paths=(path,),
        batch_size=context.options.batch_size, max_workspace_bytes=context.options.max_workspace_bytes,
        max_peak_bytes=context.options.max_peak_bytes, graph_version=context.source.graph_version,
        executor="staged_fixed_recipe_v1")


def _score(source, replacement, descriptor, snapshot, options):
    rows, width = descriptor.shape[0] // descriptor.groups, math.prod(descriptor.shape[1:])
    temporary = descriptor.groups * rows * width * (6 * 8 + 2 * next(source.parameters()).element_size())
    if temporary > options.max_workspace_bytes:
        _fail("ResourceLimitExceeded", descriptor.path, "Common-score operator and float64 residual buffers exceed workspace")
    storage_guard(temporary, options.max_peak_bytes)
    before = _weighted_matrices(source, descriptor)
    after = _weighted_matrices(replacement, descriptor)
    if ((source.bias is None) != (replacement.bias is None) or source.bias is not None
            and not torch.equal(source.bias, replacement.bias)):
        _fail("UnsupportedBiasChange", descriptor.path, "This score requires the fixed preserved bias")
    if len(snapshot.tensors) != descriptor.groups or len(before) != len(after):
        _fail("InvalidScoreContext", descriptor.path, "Moment groups must match both operators")
    errors = []
    for weight, approximation, moment in zip(before, after, snapshot.tensors):
        if moment.shape != (weight.shape[1], weight.shape[1]):
            _fail("InvalidScoreContext", descriptor.path, "The score moment must match input coordinates")
        residual = weight.detach().double() - approximation.detach().double()
        errors.append(float(((residual @ moment) * residual).sum()))
    value = math.fsum(errors)
    if not math.isfinite(value) or value < -1e-10:
        _fail("InvalidScore", descriptor.path, "Local output error must be finite and nonnegative")
    return max(0., value)


def _validate_ref(ref, context, table, *, second_moment=False):
    if type(ref) is not StatisticRef or type(ref.provenance) is not StatisticsKey or ref.key_digest != ref.provenance.digest:
        _fail("InvalidArtifactRef", table.path, "A typed statistics reference is required")
    key = ref.provenance
    if (key.source_fingerprint != context.source.fingerprint or key.graph_version != context.source.graph_version
            or key.calibration_fingerprint != context.calibration.fingerprint or key.path != table.path
            or key.aliases != table.aliases or key.batch_size != context.options.batch_size
            or key.coordinates != context.coordinates or key.normalization != context.normalization
            or key.mask != "none" or key.weights != "none" or key.input_dtype != table.operator.dtype
            or key.rng_fingerprint != context.rng_fingerprint or key.model_fingerprint != context.source.fingerprint
            or key.estimator_version != "fixed_cpu_v1"):
        _fail("IncompatibleProvenance", table.path, "Candidate statistics must refer to the frozen source and common calibration context")
    if second_moment and (key.statistic is not ArtifactKind.INPUT_SECOND_MOMENT
                          or key.model_fingerprint != context.source.fingerprint):
        _fail("InvalidScoreContext", table.path, "Score requires the original frozen input second moment")


def validate_candidate_catalog(catalog):
    """Pure metadata checks; a digest is a content identifier, not a signature."""
    if type(catalog) is not CandidateCatalog or not _metadata_only(catalog):
        _fail("InvalidCatalog", "catalog", "Metadata-only CandidateCatalog required")
    if type(catalog.max_candidates) is not int or not 1 <= catalog.max_candidates <= 4096:
        _fail("InvalidCandidateLimit", "max_candidates", "Finite candidate cap must be in [1,4096]")
    if (type(catalog.context) is not CandidateEvaluationContext or type(catalog.tables) is not tuple or not catalog.tables
            or any(type(table) is not GraphCandidateTable or type(table.operator) is not OperatorDescriptor
                   or type(table.table_id) is not str or not table.table_id or type(table.path) is not str
                   or type(table.aliases) is not tuple or any(type(alias) is not str for alias in table.aliases)
                   or type(table.candidates) is not tuple or any(type(candidate) is not GraphCandidate for candidate in table.candidates)
                   for table in catalog.tables)
            or type(catalog.source_inventory) is not tuple or type(catalog.fixed_storages) is not tuple
            or any(type(record) is not PhysicalStorage for record in (*catalog.source_inventory, *catalog.fixed_storages))):
        _fail("InvalidContext", "catalog", "A common frozen evaluation context and finite tables are required")
    context = catalog.context
    try:
        validate_inventory(catalog.source_inventory)
        validate_inventory(catalog.fixed_storages)
        for table in catalog.tables:
            for candidate in table.candidates:
                validate_inventory(candidate.storages)
    except CandidateStorageError as error:
        raise CandidateGraphError(error.violations) from error
    if (type(context.source) is not ModelVersion or type(context.calibration) is not CalibrationRef
            or type(context.options) is not RuntimeOptions or type(context.graph_version) is not str or not context.graph_version
            or type(context.source.graph_version) is not str or not context.source.graph_version
            or any(type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                   for digest in (context.source.fingerprint, context.calibration.fingerprint,
                                  context.source_inventory_fingerprint, context.rng_fingerprint))):
        _fail("InvalidContext", "context", "Typed source, calibration and runtime metadata required")
    if (context.semantics != "frozen_snapshot_surrogate" or context.regularization != "none"
            or context.score != "local_output_squared_l2_per_observation" or context.bias_policy != "preserve"
            or context.normalization != "weight_sum" or context.coordinates != "original_operator_input_output"):
        _fail("InvalidContext", "context", "The common unregularized frozen score contract cannot be changed")
    source_id = inventory_fingerprint(catalog.source_inventory)
    if source_id != context.source_inventory_fingerprint or context.source.graph_version != context.graph_version + ":inventory=" + source_id:
        _fail("InvalidSourceContext", "context", "Source inventory and graph namespace must agree")
    if len({table.table_id for table in catalog.tables}) != len(catalog.tables):
        _fail("InvalidCatalog", "tables", "Each target requires one alternative table")
    if sum(len(table.candidates) for table in catalog.tables) > catalog.max_candidates:
        _fail("InvalidCandidateLimit", "tables", "Catalog exceeds its finite candidate cap")
    targets = []
    for table in catalog.tables:
        if type(table) is not GraphCandidateTable or not table.candidates or table.path not in table.aliases:
            _fail("InvalidCatalog", "tables", "Typed, nonempty operator tables required")
        targets.extend(table.aliases)
        _validate_ref(table.score_ref, context, table, second_moment=True)
        if len({candidate.option_id for candidate in table.candidates}) != len(table.candidates):
            _fail("InvalidCatalog", table.path, "Alternative identifiers must be unique")
        for candidate in table.candidates:
            graph = candidate.graph
            failure = validate_fixed_graph(graph)
            if isinstance(failure, ValidationFailure):
                raise CandidateGraphError(failure.violations)
            if (type(candidate) is not GraphCandidate or candidate.recipe != _normalize_recipe(candidate.recipe)
                    or len(graph.transform.steps) != 1 or graph.transform.steps[0].operator != table.operator
                    or graph.scopes[0].aliases != table.aliases or graph.source != context.source
                    or graph.calibration != context.calibration or graph.options != context.options
                    or graph.executor != "staged_fixed_recipe_v1" or candidate.recipe.recipe is not graph.recipe
                    or candidate.recipe.rank != graph.transform.steps[0].rank
                    or candidate.recipe.policy != graph.transform.request.metric_policy
                    or candidate.recipe.representation != graph.transform.request.representation
                    or candidate.recipe.method_spec != graph.method_spec
                    or candidate.metric_ref != graph.nodes[1].outputs[0] or candidate.score_ref != table.score_ref
                    or type(candidate.score) not in (int, float) or not math.isfinite(candidate.score) or candidate.score < 0):
                _fail("InvalidCandidateBinding", table.path, "Graph, metric, source, score and candidate description must agree")
            _validate_ref(candidate.statistic_ref, context, table)
            expected_kind = (ArtifactKind.INPUT_SECOND_MOMENT if graph.recipe is Recipe.WEIGHTED_SVD
                else ArtifactKind.INPUT_ABS_MEAN if graph.method_spec.abs_mode == "abs_mean" else ArtifactKind.INPUT_ABS_MAX)
            if candidate.statistic_ref.provenance.statistic is not expected_kind:
                _fail("InvalidCandidateBinding", table.path, "Recipe metric must consume its own semantic statistic type")
    if len(set(targets)) != len(targets) or any(a == "" or b == "" or a.startswith(b + ".") or b.startswith(a + ".")
            for index, a in enumerate(targets) for b in targets[index + 1:]):
        _fail("OverlappingScopes", "tables", "Overlapping or aliased tables require one coherent choice")
    expected_fixed = tuple(record for record in catalog.source_inventory if any(
        not any(root == "" or view.path == root or view.path.startswith(root + ".") for root in targets)
        for view in record.views))
    if catalog.fixed_storages != expected_fixed:
        _fail("InvalidFixedStorages", "catalog", "Untouched/shared backing storages must be counted exactly once")
    if catalog.fingerprint != _digest(replace(catalog, fingerprint="")):
        _fail("CatalogFingerprintMismatch", "catalog", "Catalog metadata changed after preparation")


@_rng_isolated
@torch.no_grad()
def prepare_candidate_catalog(model, calibration, *, alternatives, store, graph_version="candidate-source-v1",
                              runtime_options=RuntimeOptions(), max_candidates=128):
    """Prepare finite alternatives from M0, with actual costs and common scores.

    No candidate weights persist. Only metadata and owned StatisticsStore refs
    survive. The caller must satisfy staged pure-observation capabilities.
    """
    try:
        if (type(alternatives) is not dict or not alternatives or type(max_candidates) is not int
                or not 1 <= max_candidates <= 4096 or any(type(path) is not str or type(specs) is not tuple or not specs for path, specs in alternatives.items())
                or sum(map(len, alternatives.values())) > max_candidates):
            _fail("InvalidCandidateLimit", "alternatives", "A bounded nonempty mapping of target paths to finite recipe tuples is required")
        if (type(store) is not StatisticsStore or type(runtime_options) is not RuntimeOptions
                or runtime_options.parameter_budget is not None or runtime_options.tensor_byte_budget is not None
                or type(graph_version) is not str or not graph_version):
            _fail("InvalidContext", "context", "Explicit bounded store, graph version and unbudgeted RuntimeOptions required")
        if (not _metadata_only(runtime_options) or any(type(getattr(runtime_options, field)) is not int
                or getattr(runtime_options, field) <= 0 for field in ("batch_size", "max_workspace_bytes", "max_peak_bytes"))
                or runtime_options.max_peak_bytes > 12 * 1024**3 or runtime_options.max_workspace_bytes > runtime_options.max_peak_bytes):
            _fail("InvalidResourceLimit", "runtime_options", "Positive CPU runtime limits must fit the declared peak of at most 12 GiB")
        if not isinstance(model, nn.Module):
            _fail("InvalidSource", "model", "A CPU torch Module is required")
        storage_guard(0, runtime_options.max_peak_bytes)
        _calibration(calibration, runtime_options)
        inventory = capture_inventory(model, prefix="source")
        inventory_id = inventory_fingerprint(inventory)
        runtime_version = graph_version + ":inventory=" + inventory_id
        source_id, data_id = _fingerprint(model, calibration, runtime_version)
        context = CandidateEvaluationContext(ModelVersion(runtime_version, source_id),
            CalibrationRef(data_id, tuple(calibration.shape), str(calibration.dtype).removeprefix("torch.")),
            inventory_id, graph_version, runtime_options, _rng_fingerprint())
        specifications = tuple((path, tuple(_normalize_recipe(spec) for spec in specs)) for path, specs in sorted(alternatives.items()))
        # Validate all capabilities/scopes before running any candidate forward.
        graphs = {(path, index): _graph(model, calibration, context, path, spec)
                  for path, specs in specifications for index, spec in enumerate(specs)}
        aliases = [alias for path, _specs in specifications for alias in graphs[(path, 0)].scopes[0].aliases]
        if len(aliases) != len(set(aliases)) or any(a == "" or b == "" or a.startswith(b + ".") or b.startswith(a + ".")
                for index, a in enumerate(aliases) for b in aliases[index+1:]):
            _fail("OverlappingScopes", "alternatives", "Select one table per coherent operator/alias scope")
        tables = []
        paths = module_paths(model)
        for path, specs in specifications:
            score_graph = _graph(model, calibration, context, path, CandidateRecipe(Recipe.WEIGHTED_SVD, 1))
            score_result = execute_fixed_graph(model, calibration, score_graph, store=store)
            score_ref = score_result.statistic_refs[0]
            descriptor = score_graph.transform.steps[0].operator
            snapshot = store.read(score_ref, expected=score_ref.provenance, max_peak_bytes=runtime_options.max_peak_bytes)
            del score_result
            candidates = []
            for index, spec in enumerate(specs):
                graph = graphs[(path, index)]
                result = execute_fixed_graph(model, calibration, graph, store=store)
                replacement = module_paths(result.model)[path]
                option_id = _digest((path, spec))
                costs = capture_inventory(replacement, prefix="candidate:" + option_id)
                value = _score(paths[path], replacement, descriptor, snapshot, runtime_options)
                candidates.append(GraphCandidate(option_id, spec, graph, result.statistic_refs[0],
                    graph.nodes[1].outputs[0], score_ref, value, costs))
                del replacement, result
            tables.append(GraphCandidateTable(path or "<root>", path, graphs[(path, 0)].scopes[0].aliases,
                                             descriptor, score_ref, tuple(candidates)))
        fixed = tuple(record for record in inventory if any(not any(root == "" or view.path.startswith(root + ".")
            for root in aliases) for view in record.views))
        catalog = CandidateCatalog(context, inventory, fixed, tuple(tables), max_candidates, "")
        catalog = replace(catalog, fingerprint=_digest(catalog))
        validate_candidate_catalog(catalog)
        return catalog
    except (CandidateStorageError, ArtifactStoreError, CompressionGraphError) as error:
        raise CandidateGraphError(error.violations) from error
    except TopologyError as error:
        _fail("UnsupportedTopology", "alternatives", str(error))
    except WeightedProfileError as error:
        _fail("InvalidCandidate", "capabilities", str(error))
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            _fail("ResourceLimitExceeded", "candidate.prepare", "Candidate preparation allocation failed after preflight")
        raise


def _allocate(catalog, budget, unit, max_shared_combinations):
    tables = tuple(CandidateTable(table.table_id, tuple(CandidateOption(candidate.option_id,
        (candidate.recipe.rank,), candidate.score, project_storages(candidate.storages, unit))
        for candidate in table.candidates), (table.path,), (min(table.operator.shape[0] // table.operator.groups,
        math.prod(table.operator.shape[1:])),)) for table in catalog.tables)
    return allocate(tables, budget, objective=ScoreObjective("frozen_local_output_error_sum"),
        fixed_storages=project_storages(catalog.fixed_storages, unit), unit=unit,
        max_shared_combinations=max_shared_combinations)


def allocate_candidate_catalog(catalog, budget, *, unit="bytes", max_shared_combinations=100000):
    """Pure projection and delegation to the existing exact allocator."""
    validate_candidate_catalog(catalog)
    try:
        allocation = _allocate(catalog, budget, unit, max_shared_combinations)
    except CandidateStorageError as error:
        raise CandidateGraphError(error.violations) from error
    except (TypeError, ValueError) as error:
        _fail("InvalidAllocation", "allocation", str(error))
    if isinstance(allocation, BudgetInfeasible):
        return allocation
    plan = CandidateAllocationPlan(catalog, allocation, budget, unit, "")
    return replace(plan, fingerprint=_digest(plan))


def _validate_plan(plan):
    if type(plan) is not CandidateAllocationPlan or not _metadata_only(plan):
        _fail("InvalidPlan", "allocation", "Metadata-only CandidateAllocationPlan required")
    validate_candidate_catalog(plan.catalog)
    expected = allocate_candidate_catalog(plan.catalog, plan.budget, unit=plan.unit)
    if isinstance(expected, BudgetInfeasible) or expected != plan:
        _fail("AllocationFingerprintMismatch", "plan", "Selection, budget and allocation must match the catalog")


def _verify_cost(model, plan):
    inventory = capture_inventory(model)
    cost = inventory_cost(inventory, plan.unit)
    if cost > plan.budget:
        _fail("BudgetInfeasible", "actual_cost", f"Actual {plan.unit} cost {cost} exceeds budget {plan.budget}")
    if cost != plan.allocation.total_cost:
        _fail("CostMismatch", "actual_cost", "Actual backing storage cost does not match the selected candidate tables")
    return inventory, cost


@_rng_isolated
@torch.no_grad()
def apply_candidate_allocation(model, calibration, plan, *, store):
    """Replay selected single-scope candidates from M0 and apply them in a batch."""
    try:
        _validate_plan(plan)
        catalog, context = plan.catalog, plan.catalog.context
        if not isinstance(model, nn.Module):
            _fail("InvalidSource", "model", "A CPU torch Module is required")
        _calibration(calibration, context.options)
        if (capture_inventory(model, prefix="source") != catalog.source_inventory
                or _fingerprint(model, calibration, context.source.graph_version) !=
                (context.source.fingerprint, context.calibration.fingerprint)):
            _fail("StaleSource", "source", "Full source state/storage topology or calibration changed")
        if _rng_fingerprint() != context.rng_fingerprint:
            _fail("StaleEvaluationContext", "rng", "Initial CPU RNG state differs from the catalog evaluation context")
        if type(store) is not StatisticsStore:
            _fail("InvalidArtifactStore", "store", "Explicit StatisticsStore required")
        paths, replacements, replay = module_paths(model), {}, []
        selections = {selection.table_id: selection.option.option_id for selection in plan.allocation.selected}
        for table in catalog.tables:
            candidate = next(item for item in table.candidates if item.option_id == selections[table.table_id])
            snapshot = store.read(table.score_ref, expected=table.score_ref.provenance,
                                  max_peak_bytes=context.options.max_peak_bytes)
            # Explicit saved refs must match source/context; never reinterpret a
            # saved current-student snapshot as a frozen curve for another M0.
            store.read(candidate.statistic_ref, expected=candidate.statistic_ref.provenance,
                       max_peak_bytes=context.options.max_peak_bytes)
            result = execute_fixed_graph(model, calibration, candidate.graph, store=store)
            if result.statistic_refs != (candidate.statistic_ref,):
                _fail("StaleCandidateStatistics", table.path, "Replay did not use the catalog's frozen statistics provenance")
            replacement = module_paths(result.model)[table.path]
            if capture_inventory(replacement, prefix="candidate:" + candidate.option_id) != candidate.storages:
                _fail("CandidateCostMismatch", table.path, "Replayed representation/storage differs from the catalog")
            value = _score(paths[table.path], replacement, table.operator, snapshot, context.options)
            if not math.isclose(value, candidate.score, rel_tol=1e-10, abs_tol=1e-12):
                _fail("CandidateScoreMismatch", table.path, "Replayed replacement does not reproduce the common-context score")
            replacements[table.path] = replacement
            replay.append({"table": table.table_id, "option": candidate.option_id,
                           "common_local_score": value, "recipe": candidate.recipe.recipe.value,
                           "observe_forward_calls": result.evidence["observe_forward_calls"]})
            del result, snapshot
        work = clone_preserving_storages(model, max_peak_bytes=context.options.max_peak_bytes)
        work = replace_modules_atomically(work, replacements)
        inventory, cost = _verify_cost(work, plan)
        return CandidateAllocationResult(work, plan, {
            "catalog_fingerprint": catalog.fingerprint, "allocation_fingerprint": plan.fingerprint,
            "semantics": context.semantics, "local_score_name": context.score,
            "local_score_sum": plan.allocation.objective_score,
            "local_score_is_whole_model_quality": False,
            "whole_model_evaluation": {"status": "required_not_run"},
            "unit": plan.unit, "budget": plan.budget, "actual_cost": cost,
            "actual_tensor_bytes": inventory_cost(inventory, "bytes"),
            "actual_parameter_numel": sum(record.parameter_numel for record in inventory),
            "stored_elements_semantics": "all_backing_storage_elements_including_buffers",
            "selected": replay, "export_verification": {"status": "required_not_run"}})
    except (CandidateStorageError, ArtifactStoreError, CompressionGraphError) as error:
        raise CandidateGraphError(error.violations) from error
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            _fail("ResourceLimitExceeded", "candidate.replay", "Allocation replay failed after memory preflight")
        raise


@_rng_isolated
@torch.no_grad()
def verify_candidate_export(result, *, format="torch_save_module"):
    """Real in-memory torch.save/load, then full-storage cost/budget validation.

    Serialized stream length is reported separately and never used as a tensor
    budget. Only this explicit trusted local module format is supported.
    """
    if type(result) is not CandidateAllocationResult or type(format) is not str or format != "torch_save_module":
        _fail("UnsupportedExport", "format", "Use a prepared result and explicit torch_save_module format")
    try:
        _validate_plan(result.plan)
        before, _cost = _verify_cost(result.model, result.plan)
        storage_guard(8 * sum(record.nbytes for record in before), result.plan.catalog.context.options.max_peak_bytes)
        stream = io.BytesIO()
        torch.save(result.model, stream)
        serialized_bytes = stream.tell()
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=False)
        after, cost = _verify_cost(restored, result.plan)
        if inventory_fingerprint(before) != inventory_fingerprint(after):
            _fail("ExportStateMismatch", "export", "Serialization changed registered tensor state or backing storage relationships")
        return CandidateExportResult(restored, {"status": "verified", "format": format,
            "actual_cost": cost, "unit": result.plan.unit, "budget": result.plan.budget,
            "tensor_bytes": inventory_cost(after, "bytes"), "serialized_stream_bytes": serialized_bytes,
            "parameter_numel": sum(record.parameter_numel for record in after)})
    except CandidateStorageError as error:
        raise CandidateGraphError(error.violations) from error
    except (pickle.PickleError, AttributeError, TypeError, EOFError) as error:
        _fail("UnsupportedExport", "export", str(error))
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            _fail("ResourceLimitExceeded", "export", "Serialization allocation failed after preflight")
        _fail("UnsupportedExport", "export", str(error))
