"""Bounded measurement adapters for the existing SVD experiment machinery.

These helpers execute a declared finite grid through ``runner.apply_candidate``.
They do not train a baseline, search for candidates, or evaluate the test role.
Latency observations come from exported, reloaded TorchScript artifacts; KL
observations come from independent single-layer interventions on validation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import platform
from typing import Mapping
from uuid import uuid4

import torch
from torch import nn

from fedcore.algorithm.low_rank.allocation import BudgetInfeasible, StorageCost, exact_unique_cost
from fedcore.algorithm.low_rank.statistical_profiles import (
    LatencyProfile, LatencyProfileKey, LatencyRankSelection, RankMeasurement, select_latency_rank,
)
from fedcore.algorithm.low_rank.topology import inspect_topology, module_paths
from . import measurement, runner
from .protocol import (
    ExperimentBundle, ExperimentProtocol, ProtocolError, json_value, stable_hash, tensor_hash, validate_roles,
)
from .svd_rank_policies import IsolatedRankEvaluation, RankOperatorCost, rank_candidate_schedule


class ProfileMeasurementError(ProtocolError):
    """A measurement cannot establish the requested finite profile."""


class RankProfileInfeasible(ProfileMeasurementError):
    """Actual storage/latency constraints exclude the declared candidate grid."""

    def __init__(self, failure: BudgetInfeasible):
        self.failure = failure
        super().__init__(f"BudgetInfeasible: {failure.reason}; budget={failure.budget}, "
                         f"minimum_cost={failure.minimum_cost}, unit={failure.unit}")


@dataclass(frozen=True)
class FLARRankMeasurements:
    profile: LatencyProfile
    selection: LatencyRankSelection
    evidence: Mapping

    def to_dict(self):
        return {"profile": asdict(self.profile), "selection": asdict(self.selection),
                "evidence": json_value(self.evidence)}


@dataclass(frozen=True)
class IsolatedKLMeasurements:
    operators: tuple[RankOperatorCost, ...]
    rank_grid: Mapping
    evaluations: tuple[IsolatedRankEvaluation, ...]
    fixed_storages: tuple[StorageCost, ...]
    evidence: Mapping

    def to_dict(self):
        return {"operators": [asdict(op) for op in self.operators],
                "rank_grid": json_value(self.rank_grid),
                "evaluations": [asdict(item) for item in self.evaluations],
                "fixed_storages": [asdict(item) for item in self.fixed_storages],
                "evidence": json_value(self.evidence)}


@dataclass(frozen=True)
class _StoredTensor:
    cost: StorageCost
    paths: tuple[str, ...]
    hashes: tuple[str, ...]


def _stores(model):
    """Physical Parameter/buffer storage, including aliases and fixed buffers.

    This finite CPU profile refuses partial storage views rather than counting a
    view's logical size as its whole storage. Aliases of a full tensor are one
    record. The path-based identities survive the independent candidate copy.
    """
    groups = {}
    for path, module in module_paths(model).items():
        for kind, members in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in members.items():
                if tensor is None:
                    continue
                # SVD factors can be column-major full dense tensors. A stride
                # permutation is safe; holes, overlaps and offsets are not.
                dense, expected_stride = True, 1
                for stride, size in sorted((stride, size) for stride, size in zip(tensor.stride(), tensor.shape)
                                           if size > 1):
                    dense = dense and stride == expected_stride
                    expected_stride *= size
                if (tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized
                        or not dense or tensor.storage_offset() != 0
                        or tensor.numel() * tensor.element_size() != tensor.untyped_storage().nbytes()):
                    raise ProfileMeasurementError("Exact profile costs require full dense CPU tensor storage")
                key = tensor.untyped_storage()._cdata
                full_path = f"{path}.{name}" if path else name
                entry = groups.setdefault(key, {"tensor": tensor, "paths": [], "kinds": [], "hashes": []})
                if tensor.dtype != entry["tensor"].dtype:
                    raise ProfileMeasurementError("Mixed-precision views of one storage are unsupported")
                entry["paths"].append(full_path)
                entry["kinds"].append(kind)
                entry["hashes"].append(tensor_hash(tensor))
    records = []
    for entry in groups.values():
        tensor = entry["tensor"]
        ordered = sorted(zip(entry["paths"], entry["hashes"]))
        paths, hashes = tuple(zip(*ordered))
        storage_id = "profile-storage:" + stable_hash(paths)
        category = "parameter" if "parameter" in entry["kinds"] else "buffer"
        records.append(_StoredTensor(StorageCost(storage_id, tensor.numel(), tensor.element_size(), category),
                                    paths, hashes))
    return tuple(sorted(records, key=lambda item: item.paths))


def _belongs(name, path):
    return path == "" or name.startswith(path + ".")


def _partition(records, target_paths):
    fixed, targets = [], []
    for record in records:
        affected = [any(_belongs(name, path) for path in target_paths) for name in record.paths]
        if any(affected) and not all(affected):
            raise ProfileMeasurementError("Target and unmodified modules share tensor storage")
        (targets if any(affected) else fixed).append(record)
    return tuple(fixed), tuple(targets)


def _cost(records, unit="parameters"):
    return exact_unique_cost((record.cost for record in records), unit)


def _validate_context(baseline, bundle, protocol, baseline_id, checkpoint_id):
    if not isinstance(bundle, ExperimentBundle) or not isinstance(protocol, ExperimentProtocol):
        raise ProfileMeasurementError("Explicit experiment bundle and protocol are required")
    if not isinstance(baseline, nn.Module) or protocol.device != "cpu":
        raise ProfileMeasurementError("The first measured SVD profile requires a CPU model/protocol")
    if not all(isinstance(value, str) and value for value in (baseline_id, checkpoint_id)):
        raise ProfileMeasurementError("One trained baseline identity and checkpoint identity are required")
    validate_roles(bundle)
    if inspect_topology(baseline).storage_aliases:
        raise ProfileMeasurementError("Measured candidate copies cannot preserve shared storage views")
    return runner.model_state_hash(baseline), _stores(baseline)


def _operator(baseline, path, *, affine):
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    modules = module_paths(baseline)
    if not isinstance(path, str) or path not in modules or type(modules[path]) not in (nn.Linear, DecomposedLinear):
        raise ProfileMeasurementError("Select an explicit supported Linear target path")
    layer = modules[path]
    if sum(id(module) == id(layer) for module in modules.values()) != 1:
        raise ProfileMeasurementError("Isolated rank profiles require a unique target module path")
    parameter = next(layer.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ProfileMeasurementError("Measured Linear factors require float32/float64")
    bias = layer.out_features if affine or layer.bias is not None else 0
    return RankOperatorCost(path, (layer.out_features, layer.in_features), bias, parameter.element_size())


def _validate_candidate(baseline_hash, baseline, source_stores, current, op, rank):
    # Exact unchanged-state comparison detects hidden calibration/buffer changes,
    # and target costs check the actual two-factor inference representation.
    before_fixed, _ = _partition(source_stores, (op.path,))
    after_stores = _stores(current)
    after_fixed, after_target = _partition(after_stores, (op.path,))
    if before_fixed != after_fixed:
        raise ProfileMeasurementError("An isolated intervention modified unselected stored tensors")
    for unit in ("parameters", "bytes"):
        expected = exact_unique_cost(op.storages(rank), unit)
        if _cost(after_target, unit) != expected:
            raise ProfileMeasurementError("Actual factor/bias storage differs from the declared rank cost")
    if runner.model_state_hash(baseline) != baseline_hash:
        raise ProfileMeasurementError("A rank evaluation modified the common trained baseline")
    return after_stores


def flar_latency_profile_key(baseline, target_path, example_input, *, source_version, threads=1):
    """Bind timing reuse to source, hardware, graph, precision and full input shape.

    ``shape`` is the compressed matrix shape. The graph hash additionally binds
    the full executable input shape, target path, CPU identity and thread count;
    this prevents e.g. sequence length changes from reusing a matrix-only key.
    Caller-supplied source_version must identify the executed source snapshot.
    """
    op = _operator(baseline, target_path, affine=False)
    if (not isinstance(example_input, torch.Tensor) or example_input.device.type != "cpu"
            or example_input.dtype not in (torch.float32, torch.float64)
            or example_input.ndim < 2 or len(example_input) == 0
            or example_input.element_size() != op.bytes_per_element
            or not bool(torch.isfinite(example_input).all())
            or not isinstance(source_version, str) or not source_version
            or type(threads) is not int or threads < 1):
        raise ProfileMeasurementError("Complete finite CPU input, source identity and thread count required")
    cpu = {"machine": platform.machine(), "processor": platform.processor(),
           "system": platform.system(), "release": platform.release()}
    graph = stable_hash({"model_state": runner.model_state_hash(baseline), "target_path": target_path,
                         "input_shape": list(example_input.shape), "input_sha256": tensor_hash(example_input),
                         "threads": threads, "cpu": cpu})
    return LatencyProfileKey(op.shape, len(example_input), str(example_input.dtype).removeprefix("torch."),
                             "torch.jit", str(torch.__version__), "cpu", graph, source_version)


def measure_flar_rank_candidates(baseline, bundle, protocol, *, target_path, rank_grid,
                                 output_dir, baseline_id, checkpoint_id, source_version,
                                 maximum_parameters, method_options=None,
                                 predicted_latency_ms=None, expected_key=None,
                                 maximum_latency_ms=None, minimum_quality=None):
    """Measure all ranks, then enumerate actual feasible whole-model choices.

    ``maximum_parameters`` counts all stored model elements, including fixed
    buffers/bias. Quality is negative existing-runner validation quality loss
    (larger is better); minimum_quality is a floor on this explicitly named
    score. Predicted latencies remain annotations and never replace a failed
    export or measured sample. Joint multi-layer quality still needs PETRA.
    """
    baseline_hash, source_stores = _validate_context(baseline, bundle, protocol, baseline_id, checkpoint_id)
    if protocol.artifact_format != "torchscript":
        raise ProfileMeasurementError("This finite latency profile measures TorchScript only")
    if type(maximum_parameters) is not int or maximum_parameters < 0:
        raise ProfileMeasurementError("Whole-model stored-element budget must be a nonnegative integer")
    if maximum_latency_ms is not None and (type(maximum_latency_ms) not in (int, float)
            or not math.isfinite(maximum_latency_ms) or maximum_latency_ms <= 0):
        raise ProfileMeasurementError("Maximum measured latency must be finite and positive")
    if minimum_quality is not None and (type(minimum_quality) not in (int, float)
            or not math.isfinite(minimum_quality)):
        raise ProfileMeasurementError("Minimum observed quality must be finite")
    op = _operator(baseline, target_path, affine=False)
    grid = {target_path: tuple(rank_grid)}
    schedule, schedule_evidence = rank_candidate_schedule("flar_svd", (op,), grid,
        method_options=method_options, baseline_id=baseline_id, checkpoint_id=checkpoint_id)
    example = bundle.validation.x[:protocol.batch_size]
    key = flar_latency_profile_key(baseline, target_path, example, source_version=source_version,
                                   threads=protocol.threads)
    if expected_key is not None and expected_key != key:
        raise ProfileMeasurementError("Stale latency validity key")
    fixed, _ = _partition(source_stores, (target_path,))
    fixed_cost = _cost(fixed)
    minimum = fixed_cost + min(exact_unique_cost(op.storages(rank)) for rank in grid[target_path])
    if minimum > maximum_parameters:
        raise RankProfileInfeasible(BudgetInfeasible(maximum_parameters, minimum, "parameters"))
    predictions = dict(predicted_latency_ms or {})
    if set(predictions) - set(grid[target_path]) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in predictions.values()):
        raise ProfileMeasurementError("Latency predictions must be finite positive annotations of declared ranks")
    # Repeated measurements must not overwrite artifacts referenced by earlier
    # evidence, even when the baseline and candidate specifications are equal.
    measurement_id = uuid4().hex
    destination = Path(output_dir) / measurement_id
    destination.mkdir(parents=True, exist_ok=True)
    records, observations = [], []
    for rank, candidate in zip(grid[target_path], schedule):
        current, transform = runner.apply_candidate(baseline, bundle, protocol, candidate)
        stores = _validate_candidate(baseline_hash, baseline, source_stores, current, op, rank)
        quality = runner.quality_metrics(current, bundle.validation, bundle.task,
                                         batch_size=protocol.batch_size, device="cpu")
        measured = measurement.measure_artifact(current, example,
            destination / f"flar-rank-{rank}-{candidate.candidate_id}.pt", format="torchscript", device="cpu",
            repeats=protocol.measurement_repeats, warmup=protocol.warmup, threads=protocol.threads)
        if measured.get("status") != "succeeded":
            causes = (measured.get("error") or {}).get("causes", ())
            details = "; ".join(cause.splitlines()[0] for cause in causes if isinstance(cause, str) and cause)
            raise ProfileMeasurementError(f"Rank {rank} artifact measurement failed: {measured.get('reason', 'unknown')}; {details}")
        profile = measured["profile"]
        if (profile["input_shape"] != list(example.shape) or profile["dtype"] != str(example.dtype)
                or profile["device"] != key.device or profile["runtime"] != key.runtime
                or profile["threads"] != protocol.threads or not measured.get("quality_parity_checked")):
            raise ProfileMeasurementError("Actual artifact measurement disagrees with the latency validity domain")
        if measured["metrics"]["tensor_state_bytes"] != _cost(stores, "bytes"):
            raise ProfileMeasurementError("Measured artifact tensor storage disagrees with exact model state cost")
        observations.append(RankMeasurement(rank, -float(quality["loss"]), predictions.get(rank),
                                             tuple(measured["raw_inference_ms"]), measured["artifact"]["sha256"]))
        records.append({"rank": rank, "candidate": candidate.to_dict(), "transform": transform,
                        "validation_quality": quality, "measurement": measured,
                        "local_stored_elements": exact_unique_cost(op.storages(rank)),
                        "whole_model_stored_elements": _cost(stores), "whole_model_tensor_bytes": _cost(stores, "bytes"),
                        "whole_model_budget_feasible": _cost(stores) <= maximum_parameters})
    profile = LatencyProfile(key, grid[target_path], tuple(observations), source_version)
    selected = select_latency_rank(profile, key, maximum_parameters=maximum_parameters-fixed_cost,
        bias_elements=op.bias_elements, maximum_latency_ms=maximum_latency_ms,
        minimum_quality=minimum_quality, score_direction="maximize")
    if isinstance(selected, BudgetInfeasible):
        raise RankProfileInfeasible(BudgetInfeasible(maximum_parameters, selected.minimum_cost+fixed_cost,
                                                    selected.unit, selected.reason))
    selected_total = fixed_cost + selected.parameter_cost
    if selected_total > maximum_parameters:
        raise ProfileMeasurementError("Selected artifact exceeds the actual whole-model storage budget")
    selected = replace(selected, manifest={**selected.manifest,
        "parameter_cost_scope": "selected_target_factors_and_bias",
        "whole_model_stored_elements": selected_total, "fixed_stored_elements": fixed_cost,
        "whole_model_maximum_parameters": maximum_parameters})
    evidence = {**schedule_evidence, "baseline_state_sha256": baseline_hash,
        "validity_key": asdict(key), "input_shape": list(example.shape), "threads": protocol.threads,
        "quality_score": "negative_validation_quality_loss", "quality_direction": "maximize",
        "latency_scope": "complete_exported_reloaded_model", "latency_statistic": "median_raw_inference_ms",
        "measurement_id": measurement_id, "example_input_sha256": tensor_hash(example),
        "predictions_are_measurements": False, "timing_speedup_claim": False,
        "maximum_parameters_scope": "whole_model_stored_elements_including_buffers",
        "maximum_parameters": maximum_parameters, "fixed_stored_elements": fixed_cost,
        "selected_whole_model_stored_elements": selected_total,
        "fixed_storages": [asdict(record.cost) for record in fixed], "evaluations": records,
        "validation": bundle.validation.manifest(), "calibration": bundle.calibration.manifest(),
        "test_evaluated": False, "joint_evaluation_required_for_multiple_targets": True}
    return FLARRankMeasurements(profile, selected, evidence)


def _validation_kl(teacher, student, split, task, batch_size):
    """FP64 KL(teacher || student), streamed over independent validation rows."""
    if task not in ("classification", "language_model"):
        raise ProfileMeasurementError("AAFM KL requires classification or causal-language-model logits")
    split.verify_integrity()
    teacher.eval()
    student.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(split.x), batch_size):
            x = split.x[start:start+batch_size]
            reference, current = teacher(x), student(x)
            if (not isinstance(reference, torch.Tensor) or not isinstance(current, torch.Tensor)
                    or reference.shape != current.shape or reference.ndim < 2 or reference.shape[-1] < 2
                    or not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all())):
                raise ProfileMeasurementError("KL requires equal finite tensor logits with at least two classes")
            if task == "classification":
                if reference.ndim != 2 or len(reference) != len(x):
                    raise ProfileMeasurementError("Classification KL requires [batch,classes] logits")
            else:
                labels = split.y[start:start+batch_size]
                if reference.ndim != 3 or reference.shape[:2] != labels.shape or reference.shape[1] < 2:
                    raise ProfileMeasurementError("Causal LM KL requires [batch,time,vocabulary] logits")
                mask = labels[:, 1:] != -100
                reference, current = reference[:, :-1][mask], current[:, :-1][mask]
            if not len(reference):
                continue
            log_p, log_q = reference.double().log_softmax(-1), current.double().log_softmax(-1)
            terms = (log_p.exp() * (log_p-log_q)).sum(-1)
            total += float(terms.sum())
            count += terms.numel()
    if count == 0 or not math.isfinite(total):
        raise ProfileMeasurementError("KL requires at least one finite nonpadding validation observation")
    value = total/count
    if value < -1e-12:
        raise ProfileMeasurementError("Computed KL is numerically negative")
    return max(0.0, value), count


def collect_aafm_kl_candidates(baseline, bundle, protocol, *, target_paths, rank_grid,
                              baseline_id, checkpoint_id, method_options=None, unit="parameters"):
    """Collect isolated validation KL and exact costs from one trained baseline.

    Every evaluation's cost is LOCAL factor+bias cost, matching the existing
    ``select_isolated_ranks`` contract. Returned fixed_storages exclude all
    selected targets and complete the eventual JOINT model budget. Evidence
    also records each isolated model's actual whole-model cost, which includes
    the other targets still stored densely. This is no joint KL prediction.
    """
    baseline_hash, source_stores = _validate_context(baseline, bundle, protocol, baseline_id, checkpoint_id)
    if unit not in ("parameters", "bytes"):
        raise ProfileMeasurementError("Explicit stored-element or tensor-byte cost unit required")
    if bundle.task not in ("classification", "language_model"):
        raise ProfileMeasurementError("AAFM KL requires classification or causal-language-model logits")
    paths = tuple(target_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ProfileMeasurementError("AAFM requires nonempty unique target paths")
    operators = tuple(_operator(baseline, path, affine=True) for path in paths)
    for op in operators:
        _partition(source_stores, (op.path,))
    if not isinstance(rank_grid, Mapping):
        raise ProfileMeasurementError("AAFM requires an explicit complete per-path rank grid")
    # Capture one-shot grid iterators once, before scheduling the effects.
    grid = {path: tuple(values) for path, values in rank_grid.items()}
    schedule, schedule_evidence = rank_candidate_schedule("afm", operators, grid,
        method_options=method_options, baseline_id=baseline_id, checkpoint_id=checkpoint_id)
    fixed, _ = _partition(source_stores, paths)
    calibration = bundle.calibration.manifest()
    teacher = deepcopy(baseline).eval().requires_grad_(False)
    observations, records = [], []
    index = 0
    for op in operators:
        statistics_id = stable_hash({"baseline_state": baseline_hash, "path": op.path, "method": "afm",
                                     "method_options": json_value(method_options or {}),
                                     "calibration": calibration, "batch_size": protocol.batch_size})
        for rank in grid[op.path]:
            candidate = schedule[index]
            index += 1
            current, transform = runner.apply_candidate(baseline, bundle, protocol, candidate)
            stores = _validate_candidate(baseline_hash, baseline, source_stores, current, op, rank)
            score, count = _validation_kl(teacher, current, bundle.validation, bundle.task, protocol.batch_size)
            observations.append(IsolatedRankEvaluation(op.path, rank, score,
                exact_unique_cost(op.storages(rank), unit), baseline_id, "validation_kl_teacher_to_student",
                "minimize", "validation", checkpoint_id, baseline_hash, statistics_id))
            records.append({"path": op.path, "rank": rank, "candidate": candidate.to_dict(),
                            "transform": transform, "statistics_id": statistics_id, "validation_kl": score,
                            "kl_observations": count, "local_cost": exact_unique_cost(op.storages(rank), unit),
                            "whole_model_cost": _cost(stores, unit), "whole_model_tensor_bytes": _cost(stores, "bytes")})
    if runner.model_state_hash(baseline) != baseline_hash:
        raise ProfileMeasurementError("Validation collection changed the common trained baseline")
    evidence = {**schedule_evidence, "baseline_state_sha256": baseline_hash,
        "score_name": "validation_kl_teacher_to_student", "score_direction": "minimize",
        "kl_definition": "mean_observation_sum_class_p_teacher_log_p_teacher_over_p_student",
        "kl_temperature": 1.0, "kl_accumulation_dtype": "float64",
        "kl_causal_shift": bundle.task == "language_model", "kl_padding_label": -100 if bundle.task == "language_model" else None,
        "cost_unit": unit, "local_cost_scope": "actual_two_factors_and_affine_bias",
        "fixed_cost": _cost(fixed, unit), "baseline_whole_model_cost": _cost(source_stores, unit),
        "joint_cost_formula": "unique_fixed_storage_plus_all_chosen_local_storage",
        "joint_quality_observed": False, "joint_evaluation_required": True,
        "calibration": calibration, "validation": bundle.validation.manifest(), "test_evaluated": False,
        "evaluations": records}
    return IsolatedKLMeasurements(operators, grid, tuple(observations), tuple(record.cost for record in fixed), evidence)
