"""Runtime interpreter for the finite frozen-source compression grammar."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import time

import torch
from torch import nn

from .approximation import NumericalDomainViolation
from .artifacts import ArtifactStoreError, StatisticsStore
from .candidate_storage import (CandidateStorageError, capture_inventory,
    clone_preserving_storages, inventory_cost, storage_guard)
from .candidate_storage import inventory_fingerprint
from .compression_grammar import (ASVDChoice, AffinePCAChoice, AffineWholeProfilePlan,
    CheckedWorkflow, GrammarSpace, ScopeDomain, WeightedChoice, compile_genotype,
    validate_grammar_space, validate_workflow)
from .compression_graph import CalibrationRef, ModelVersion, RuntimeOptions
from .execution import WeightedProfileError, _fingerprint, plan_weighted
from .graph_execution import (CompressionGraphError, _allocation_failure,
    _calibration, _copy_calibration, execute_fixed_graph)
from .method_execution import (MethodProfileError, _collect_rows, _explicit_encoder_graph,
    _install, _matrix, _numerics, _rng_isolated, _statistical_solve, plan_method)
from .method_specs import AFM
from .plans import ValidationFailure, Violation
from .topology import TopologyError, inspect_topology, module_paths, replace_modules_atomically
from .statistics import StatisticsPassport
from .staged_execution import _pure_observe


@dataclass(frozen=True)
class StructuralInvalid:
    violations: tuple[Violation, ...]


@dataclass(frozen=True)
class NumericalFailure:
    code: str
    message: str


@dataclass(frozen=True)
class ResourceFailure:
    code: str
    message: str


@dataclass(frozen=True)
class CallbackFailure:
    code: str
    message: str


@dataclass(frozen=True)
class WorkflowSuccess:
    model: nn.Module
    workflow: CheckedWorkflow
    evidence: dict


@dataclass(frozen=True)
class PoorQuality:
    """A finite measured result, distinct from structural or numerical failure."""
    result: WorkflowSuccess
    score: float
    maximum: float


WorkflowOutcome = WorkflowSuccess | StructuralInvalid | NumericalFailure | ResourceFailure | CallbackFailure | PoorQuality


def prepare_grammar_space(model, calibration, *, alternatives, graph_version="compression-grammar-v1",
        options=RuntimeOptions(), max_scopes=32, max_alternatives=4096):
    """Read source metadata and check all finite alternatives without forwards.

    Each input path names a complete module alias scope. Partial parameter ties,
    storage views and coupled multi-operator groups are rejected by the existing
    profile capability checks, rather than being flattened into independent genes.
    """
    if type(alternatives) is not dict or not alternatives or any(type(path) is not str
            or type(choices) is not tuple or not choices for path, choices in alternatives.items()):
        return ValidationFailure((Violation("InvalidAlternatives", "alternatives", "A finite path-to-choice tuple mapping is required"),))
    if type(options) is not RuntimeOptions or type(graph_version) is not str or not graph_version:
        return ValidationFailure((Violation("InvalidSourceContext", "space", "Explicit RuntimeOptions and graph version are required"),))
    if not isinstance(model, nn.Module):
        return ValidationFailure((Violation("InvalidSource", "model", "A CPU torch Module is required"),))
    try:
        _calibration(calibration, options)
        topology = inspect_topology(model)
        paths = module_paths(model)
        domains = []
        for path, choices in alternatives.items():
            if path not in paths:
                raise TopologyError("Unknown module scope")
            aliases = tuple(sorted(next((group for group in topology.module_aliases if path in group), (path,))))
            canonical = aliases[0]
            # rank=1 checks the inherited operator/group/tie capability boundary.
            base = plan_weighted(model, rank=1, target_paths=(canonical,), max_peak_bytes=options.max_peak_bytes)
            for choice in choices:
                if type(choice) is WeightedChoice:
                    plan_weighted(model, rank=choice.rank, target_paths=(canonical,), policy=choice.policy,
                        representation=choice.representation, max_peak_bytes=options.max_peak_bytes)
                elif type(choice) in (ASVDChoice, AffinePCAChoice):
                    plan_method(model, choice.spec if type(choice) is ASVDChoice else AFM(), rank=choice.rank,
                        target_paths=(canonical,), max_peak_bytes=options.max_peak_bytes)
            domains.append(ScopeDomain(base.steps[0].operator, aliases, choices))
        checkpoint, data = _fingerprint(model, calibration, graph_version)
        space = GrammarSpace(ModelVersion(graph_version, checkpoint),
            CalibrationRef(data, tuple(calibration.shape), str(calibration.dtype).removeprefix("torch.")),
            tuple(sorted(domains, key=lambda domain: domain.aliases)), topology.module_aliases,
            options, max_scopes, max_alternatives,
            inventory_fingerprint(capture_inventory(model)))
        failure = validate_grammar_space(space)
        return failure if failure is not None else space
    except (WeightedProfileError, TopologyError, TypeError, ValueError, OverflowError) as error:
        return ValidationFailure(getattr(error, "violations", (Violation("UnsupportedCapability", "space", str(error)),)))


def _structural(code, path, message):
    return StructuralInvalid((Violation(code, path, message),))


@_rng_isolated
@torch.no_grad()
def execute_workflow(model, calibration, workflow, *, store=None, phase_callback=None):
    """Execute complete profiles on M0, then publish one independent model.

    Without ``store``, each fixed profile owns a temporary statistics store.
    An explicit bounded ``StatisticsStore`` belongs to the caller and is never
    closed here. Each fixed graph commits independently: a later scope failure
    does not roll back statistics committed by an earlier scope. AFM never uses
    this store. ``phase_callback`` observes real nested execution intervals.

    Profile observations and numerical evidence are runtime-owned. AFM uses the
    existing affine output-PCA solver and adds its mean correction even when the
    original operator has no bias. No local objective is used as search fitness.
    """
    failure = validate_workflow(workflow)
    if isinstance(failure, ValidationFailure):
        return StructuralInvalid(failure.violations)
    if store is not None and type(store) is not StatisticsStore:
        return _structural("InvalidArtifactStore", "store", "An explicit bounded StatisticsStore is required")
    if phase_callback is not None and not callable(phase_callback):
        return _structural("InvalidObserver", "phase_callback", "A callable phase observer is required")
    space, options = workflow.space, workflow.space.options
    if not isinstance(model, nn.Module):
        return _structural("InvalidSource", "model", "A CPU torch Module is required")
    try:
        _calibration(calibration, options)
        checkpoint, data = _fingerprint(model, calibration, space.source.graph_version)
        if checkpoint != space.source.fingerprint:
            return _structural("StaleModelVersion", "source", "Source state or configuration changed")
        if CalibrationRef(data, tuple(calibration.shape), str(calibration.dtype).removeprefix("torch.")) != space.calibration:
            return _structural("StaleCalibration", "calibration", "Calibration content changed")
        if inspect_topology(model).module_aliases != space.module_aliases:
            return _structural("IncompleteAliasScope", "source", "Source aliases differ from the checked closure")
        if space.source_storage_fingerprint != inventory_fingerprint(capture_inventory(model)):
            return _structural("StaleSourceStorage", "source.storage", "Full source storage, nonpersistent buffers or tensor-object aliases changed")
        if any(value.device.type != "cpu" for value in (*model.parameters(), *model.buffers())):
            return _structural("UnsupportedCapability", "model", "The grammar interpreter requires CPU tensors")
        # Check every planned descriptor against actual source capabilities before
        # cloning, hooks or solver effects. A forged destination cannot rebind it.
        actual = prepare_grammar_space(model, calibration, alternatives={domain.aliases[0]: domain.alternatives
            for domain in space.domains}, graph_version=space.source.graph_version, options=options,
            max_scopes=space.max_scopes, max_alternatives=space.max_alternatives)
        if isinstance(actual, ValidationFailure):
            return StructuralInvalid(actual.violations)
        if actual != space:
            return _structural("CapabilityMismatch", "space", "Current descriptors differ from the checked source")
        owned_data, copy_evidence = _copy_calibration(calibration, options)
        replacements, records = {}, []
        for compiled in workflow.scopes:
            path = compiled.gene.aliases[0]
            profile_started = time.perf_counter() if phase_callback is not None else None
            profile_phase = "afm_whole_profile" if type(compiled.plan) is AffineWholeProfilePlan else "fixed_profile"
            if phase_callback is not None:
                phase_callback({"phase": profile_phase, "scope": path,
                    "started_at": profile_started, "completed_at": None, "state": "started"})
            if type(compiled.plan) is AffineWholeProfilePlan:
                import psutil
                afm_work = _explicit_encoder_graph(clone_preserving_storages(model,
                    max_peak_bytes=options.max_peak_bytes))
                original_mode = module_paths(afm_work)[path].training
                afm_work.eval()
                layer = module_paths(afm_work)[path]
                reserve = psutil.Process(os.getpid()).memory_info().rss + 64 * 1024**2
                rows = _pure_observe(afm_work, owned_data, space.source.graph_version,
                    lambda: _collect_rows(afm_work, layer, owned_data, batch_size=options.batch_size,
                        workspace=options.max_workspace_bytes, peak=options.max_peak_bytes, reserve=reserve))
                output_width = layer.out_features
                needed = (rows.numel()*8*4 + rows.shape[0]*output_width*8*4
                          + 8*output_width*output_width*8 + layer.weight.numel()*8*8)
                if needed > options.max_workspace_bytes:
                    return ResourceFailure("ResourceLimitExceeded", "Named AFM output, covariance and solve buffers exceed workspace")
                storage_guard(needed, options.max_peak_bytes)
                passport = StatisticsPassport(space.source.fingerprint, space.source.graph_version,
                    space.source.fingerprint, path + ":input", space.calibration.fingerprint)
                answer = _statistical_solve(AFM(), _matrix(layer), layer.bias, rows, passport,
                    compiled.gene.choice.rank, None, None)
                replacement = _install(layer, answer)
                replacement.training = original_mode
                evidence = {"executor": "checked_afm_whole_profile_v1",
                    "stage_trace_kind": "declared_inline_whole_profile_stages",
                    "observation_effect_guard": "model,data,modes,CPU_RNG,backing_storage",
                    "numerical_evidence": _numerics(answer)}
                del rows, answer, afm_work, layer
            else:
                profile_store = StatisticsStore(options.max_workspace_bytes) if store is None else store
                try:
                    result = execute_fixed_graph(model, owned_data, compiled.plan, store=profile_store,
                        phase_callback=phase_callback)
                    evidence = result.evidence
                    replacement = module_paths(result.model)[path]
                finally:
                    if store is None:
                        profile_store.close()
                del result
            if phase_callback is not None:
                phase_callback({"phase": profile_phase, "scope": path, "state": "completed",
                    "started_at": profile_started, "completed_at": time.perf_counter()})
            replacements[path] = replacement
            records.append({"aliases": list(compiled.gene.aliases), "source": space.source.fingerprint,
                "profile_evidence": evidence})
            # Account for retained replacements plus the next profile's copy.
            retained = sum(sum(p.numel() * p.element_size() for p in replacement.parameters())
                           for replacement in replacements.values())
            storage_guard(retained, options.max_peak_bytes)
            del replacement
        assembly_started = time.perf_counter() if phase_callback is not None else None
        if phase_callback is not None:
            phase_callback({"phase": "assembly_apply", "started_at": assembly_started,
                "completed_at": None, "state": "started"})
        work = clone_preserving_storages(model, max_peak_bytes=options.max_peak_bytes)
        work = replace_modules_atomically(work, replacements)
        inventory = capture_inventory(work)
        actual_bytes = inventory_cost(inventory, "bytes")
        actual_elements = inventory_cost(inventory, "parameters") if all(record.elements is not None for record in inventory) else None
        if actual_elements is None and options.parameter_budget is not None:
            return _structural("AmbiguousElementCost", "budget", "Mixed-dtype storage has no single stored-element cost; byte budgets remain defined")
        if ((options.parameter_budget is not None and actual_elements > options.parameter_budget)
                or (options.tensor_byte_budget is not None and actual_bytes > options.tensor_byte_budget)):
            return ResourceFailure("BudgetInfeasible", "The actual complete backing-storage inventory exceeds the declared budget")
        if phase_callback is not None:
            phase_callback({"phase": "assembly_apply", "started_at": assembly_started,
                "completed_at": time.perf_counter(), "state": "completed"})
        return WorkflowSuccess(work, workflow, {"workflow_fingerprint": workflow.fingerprint,
            "source_semantics": workflow.source_semantics, "scope_profiles": records,
            "calibration_copy": copy_evidence, "actual_backing_storage_bytes": actual_bytes,
            "actual_stored_elements": actual_elements, "apply_count": 1, "recovery": "none",
            "fitness_role": "caller-owned heldout evaluation; local profile objectives are not search fitness"})
    except NumericalDomainViolation as error:
        return NumericalFailure(type(error).__name__, str(error))
    except (CompressionGraphError, CandidateStorageError, ArtifactStoreError) as error:
        violations = getattr(error, "violations", ())
        numerical = {"NumericalDomainViolation", "NumericalFailure", "SolverFailure", "NonfiniteStatistics", "NonfiniteResult"}
        cause = error.__cause__
        if isinstance(cause, NumericalDomainViolation) or any(v.code in numerical for v in violations):
            return NumericalFailure(next((v.code for v in violations if v.code in numerical), type(cause).__name__), str(error))
        resources = {"ResourceLimitExceeded", "BudgetInfeasible", "StoreCapacityExceeded"}
        if any(v.code in resources for v in violations):
            return ResourceFailure(next(v.code for v in violations if v.code in resources), str(error))
        return StructuralInvalid(violations or (Violation("RuntimeContract", "workflow", str(error)),))
    except (MethodProfileError, WeightedProfileError, TopologyError) as error:
        if str(error) == "Nonfinite layer observations":
            return NumericalFailure("NonfiniteObservations", str(error))
        if "ResourceLimitExceeded" in str(error) or "BudgetInfeasible" in str(error):
            return ResourceFailure("ResourceLimitExceeded" if "ResourceLimitExceeded" in str(error) else "BudgetInfeasible", str(error))
        return _structural("RuntimeContract", "workflow", str(error))
    except ValueError as error:
        # Existing statistics kernels predate NumericalDomainViolation. Translate
        # their exact numerical failures; schema/topology/programming errors
        # continue to propagate rather than being mislabeled as bad numerics.
        statistics_errors = {"Statistics overflow": "StatisticsOverflow",
                             "Observations require finite float32/float64 rows": "NonfiniteObservations"}
        if str(error) in statistics_errors:
            return NumericalFailure(statistics_errors[str(error)], str(error))
        raise
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            return ResourceFailure("ResourceLimitExceeded", str(error))
        if isinstance(error, torch.linalg.LinAlgError):
            return NumericalFailure(type(error).__name__, str(error))
        return CallbackFailure(type(error).__name__, str(error))


def assess_quality(result, score, maximum):
    """A finite quality rejection retains its model and measured score."""
    if type(result) is not WorkflowSuccess:
        return _structural("InvalidQualityContext", "quality", "Quality assessment requires an executed successful workflow")
    try:
        finite = all(type(value) in (int, float) and math.isfinite(value) for value in (score, maximum))
    except OverflowError:
        finite = False
    if not finite:
        return NumericalFailure("NonfiniteQuality", "Heldout quality scores and threshold must be finite")
    return PoorQuality(result, float(score), float(maximum)) if score > maximum else result
