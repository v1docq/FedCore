"""Owned CPU interpreter for the checked CIFAR structural/SVD graph.

CPU compilation is bounded by declared double-SVD workspaces and the common
12 GiB peak ceiling. A100 inference/recovery belongs to the caller; this profile
does not claim CUDA sparse, N:M, quantized, or training graph capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from .candidate_storage import (CandidateStorageError, capture_inventory, clone_preserving_storages,
    inventory_cost, inventory_fingerprint, storage_guard)
from .compression_graph import CalibrationRef, ModelVersion, RuntimeOptions
from .compression_grammar import _hash
from .cv_grammar import (CVBlockDomain, CVGrammarSpace, CheckedCVWorkflow, DenseCV,
    validate_cv_grammar_space, validate_cv_workflow)
from .execution import _hash_tensor
from .grammar_execution import StructuralInvalid, NumericalFailure, ResourceFailure
from .plans import ValidationFailure, Violation
from .topology import replace_modules_atomically
from fedcore.algorithm.pruning.sparse.cv_channels import (
    BLOCK_PATHS, BLOCK_OUTPUTS, CifarResNet18, validate_cv_model, build_compact_block, select_internal_channels)


@dataclass(frozen=True)
class CVWorkflowSuccess:
    model: CifarResNet18
    workflow: CheckedCVWorkflow
    checkpoint_bytes: bytes
    inventory: tuple
    evidence: dict


def _calibration(calibration):
    if (not isinstance(calibration, torch.Tensor) or calibration.device.type != "cpu"
            or calibration.dtype != torch.float32 or calibration.layout != torch.strided
            or calibration.ndim != 4 or tuple(calibration.shape[1:]) != (3, 32, 32)
            or not 1 <= calibration.shape[0] <= 4096 or not bool(torch.isfinite(calibration).all())
            or calibration.is_conj() or calibration.is_neg()):
        raise ValueError("CV calibration requires finite CPU float32 NCHW CIFAR32 inputs, 1..4096 samples")
    digest = hashlib.sha256()
    _hash_tensor(digest, calibration)
    return digest.hexdigest()


def _source(model):
    cost = validate_cv_model(model, cpu=True)
    if model.conv1.weight.dtype != torch.float32:
        raise ValueError("The first CV execution profile requires float32")
    if model.block_widths != BLOCK_OUTPUTS or any(rank is not None for rank in model.conv2_ranks):
        raise ValueError("CV workflows observe one dense frozen ResNet18, never a previous student")
    if any(layer.training for layer in model.modules()):
        raise ValueError("Freeze source evaluation modes before preparing the CV grammar")
    if model.channel_indices != tuple(tuple(range(width)) for width in BLOCK_OUTPUTS):
        raise ValueError("Dense source coordinates must be original complete block channels")
    inventory = capture_inventory(model)
    storage = inventory_fingerprint(inventory)
    identity = _hash((model.representation_config(), storage,
                      tuple((name, layer.training) for name, layer in model.named_modules())))
    return identity, storage, cost


def prepare_cv_grammar_space(model, calibration, *, alternatives, graph_version="cv-compression-v1",
                             options=RuntimeOptions()):
    """Read complete source/data identities and reject capabilities before effects.

    No model forward, hooks, copies, pruning, or SVD occurs here. Each alternative
    must fit its conservative solver workspace separately. For all eight 512-wide
    blocks, callers can declare 256 MiB workspace; no full activation Gram exists.
    """
    if (type(alternatives) is not dict or not alternatives or any(type(path) is not str
            or type(choices) is not tuple or not choices for path, choices in alternatives.items())
            or type(graph_version) is not str or not graph_version):
        return ValidationFailure((Violation("InvalidAlternatives", "space", "Declare finite exact block alternatives and a graph version"),))
    try:
        source, storage, cost = _source(model)
        data = _calibration(calibration)
        space = CVGrammarSpace(ModelVersion(graph_version, source), CalibrationRef(data, tuple(calibration.shape), "float32"),
            tuple(CVBlockDomain(path, choices) for path, choices in sorted(alternatives.items())),
            storage, cost, model.num_classes, "float32", options)
        failure = validate_cv_grammar_space(space)
        return space if failure is None else failure
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError, CandidateStorageError) as error:
        return ValidationFailure(getattr(error, "violations", (Violation("UnsupportedCapability", "space", str(error)),)))


def _invalid(code, path, message):
    return StructuralInvalid((Violation(code, path, message),))


@torch.no_grad()
def execute_cv_workflow(model, calibration, workflow):
    """Rebind exactly, build independent blocks, atomically apply and round-trip.

    The source is never forwarded or mutated. Calibration is content-bound but
    this weight-only structural/Frobenius profile does not estimate statistics.
    Source tensors, modes, storage and CPU RNG are unchanged on every outcome.
    """
    failure = validate_cv_workflow(workflow)
    if failure is not None:
        return StructuralInvalid(failure.violations)
    space = workflow.space
    checked = prepare_cv_grammar_space(model, calibration,
        alternatives={domain.path: domain.alternatives for domain in space.domains},
        graph_version=space.source.graph_version, options=space.options)
    if type(checked) is ValidationFailure:
        return StructuralInvalid(checked.violations)
    if checked != space:
        return _invalid("StaleSourceContext", "space", "Source configuration, complete storage, modes or calibration content changed")
    try:
        required = 6 * space.source_storage_bytes + max((block.solver_workspace_bytes for block in workflow.blocks), default=0)
        storage_guard(required, space.options.max_peak_bytes)
        with torch.random.fork_rng(devices=[]):
            work = clone_preserving_storages(model, max_peak_bytes=space.options.max_peak_bytes)
            replacements, widths, ranks, coordinates, history = {}, list(work.block_widths), list(work.conv2_ranks), list(work.channel_indices), []
            for block in workflow.blocks:
                gene, path = block.gene, block.gene.path
                if type(gene.choice) is DenseCV:
                    continue
                source = model.get_submodule(path)
                keep = select_internal_channels(source, block.internal_width)
                replacement = build_compact_block(source, keep, block.rank)
                for name, parameter in replacement.named_parameters():
                    source_name = "conv2.weight" if name.startswith("conv2.") else name
                    parameter.requires_grad_(dict(source.named_parameters())[source_name].requires_grad)
                replacements[path] = replacement
                index = BLOCK_PATHS.index(path)
                widths[index], ranks[index], coordinates[index] = block.internal_width, block.rank, keep
                history.append({"path": path, "recipe": type(gene.choice).__name__,
                    "profile": "channel_l1_structural_v1" if len(keep) < BLOCK_OUTPUTS[index] else "dense_internal_channels",
                    "original_channels": list(keep), "internal_width": block.internal_width, "conv2_rank": block.rank,
                    "conv1_shape": list(replacement.conv1.weight.shape),
                    "conv2_shapes": [list(layer.weight.shape) for layer in replacement.conv2] if block.rank is not None else [list(replacement.conv2.weight.shape)],
                    "original_linked_tensor_bytes": block.original_tensor_bytes, "final_linked_tensor_bytes": block.final_tensor_bytes,
                    "svd_semantics": "truncated_Frobenius_SVD_of_channel_selected_conv2" if block.rank is not None else "none"})
            work = replace_modules_atomically(work, replacements)
            work.block_widths, work.conv2_ranks, work.channel_indices = tuple(widths), tuple(ranks), tuple(coordinates)
            validate_cv_model(work, cpu=True)
            inventory = capture_inventory(work)
            physical_bytes = inventory_cost(inventory, "bytes")
            if physical_bytes != workflow.expected_storage_bytes:
                return _invalid("RepresentationMismatch", "model", "Executed physical tensor bytes differ from the pure checked plan")
            evidence = {"schema": "cv-workflow-v1", "workflow_fingerprint": workflow.fingerprint,
                "source_fingerprint": space.source.fingerprint, "calibration_fingerprint": space.calibration.fingerprint,
                "source_storage_fingerprint": space.source_storage_fingerprint,
                "source_tensor_bytes": space.source_storage_bytes, "storage_bytes": physical_bytes,
                "history": history, "source_semantics": workflow.source_semantics,
                "runtime": "ordinary_float32_Conv2d_and_real_two_Conv2d_SVD_factors_CPU_build_CUDA_compatible",
                "recovery": "NoRecovery; caller-owned fixed evaluator stage only"}
            work._fedcore_method_evidence = evidence
            from fedcore.tools.registry.checkpoint_manager import CheckpointManager
            manager = CheckpointManager(".", auto_cleanup=False)
            payload = manager.serialize_to_bytes(work)
            restored = manager.deserialize_from_bytes(payload, device="cpu")
            if (type(restored) is not CifarResNet18 or restored.representation_config() != work.representation_config()
                    or inventory_fingerprint(capture_inventory(restored)) != inventory_fingerprint(inventory)
                    or tuple((name, layer.training) for name, layer in restored.named_modules()) !=
                       tuple((name, layer.training) for name, layer in work.named_modules())):
                return _invalid("CheckpointMismatch", "model", "Exact constructor, storage, tensor state and modes must survive checkpoint restoration")
            # Small local probes exercise every changed residual block after an
            # independent constructor reload, including its true reduced shapes.
            for path in replacements:
                incoming = work.get_submodule(path).conv1.in_channels
                probe = torch.arange(incoming * 4 * 4, dtype=torch.float32).reshape(1, incoming, 4, 4) / (incoming * 16)
                output = restored.get_submodule(path)(probe)
                expected = work.get_submodule(path)(probe)
                if not bool(torch.isfinite(output).all()) or not torch.equal(output, expected):
                    return _invalid("CheckpointMismatch", path, "Reloaded real residual blocks must execute exactly on deterministic probes")
            if _source(model) != (space.source.fingerprint, space.source_storage_fingerprint, space.source_storage_bytes) or _calibration(calibration) != space.calibration.fingerprint:
                return _invalid("SourceMutation", "source", "Source or calibration changed during the owned interpreter")
            restored._fedcore_requires_optimizer_rebuild = True
            return CVWorkflowSuccess(restored, workflow, payload, capture_inventory(restored), evidence)
    except CandidateStorageError as error:
        return ResourceFailure(error.violations[0].code, str(error))
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as error:
        return NumericalFailure("CVExecutionFailed", str(error))
