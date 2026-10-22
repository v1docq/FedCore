"""Checked finite composition of concrete CPU operator representations.

This grammar is independent of the frozen-observation G3 grammar. Each scope
executes its declared order against its own predecessor; independent scopes
are installed together on an owned model. No training or graph tracing occurs.
Masks leave ordinary dense zeros. Quantization means the existing portable
DoBi factor remapping (reconstruct, SVD, encode), never TorchAO or fake quant.
Only a verified, weights-only checkpoint is a supported final artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json

import torch
from torch import nn
from torch.nn.utils import prune

from .candidate_storage import (CandidateStorageError, PhysicalStorage,
    capture_inventory, clone_preserving_storages, inventory_cost,
    inventory_fingerprint, storage_guard)
from .dobi_remapping import RemappedLinear, remapped_tensor_bytes
from .plans import ValidationFailure, Violation, _metadata_only
from .structured_layers import FactorizedLinear, ResidualLinear
from .structured_profiles import FactorPair, _svd
from .topology import module_paths, replace_modules_atomically


class Representation(str, Enum):
    DENSE = "dense"
    DENSE_MASKED = "dense_masked"
    REAL_FACTORS = "real_factors"
    PACKED_FACTORS = "packed_factors"
    RESIDUAL_BRANCH = "residual_branch"


class FinalFormat(str, Enum):
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class PruneMask:
    """Zero explicit row-major weight coordinates; no retained sparse format."""
    zero_indices: tuple[int, ...]


@dataclass(frozen=True)
class LowRank:
    rank: int


@dataclass(frozen=True)
class QuantizeRemapped:
    tail_dtype: str = "float16"
    scale_dtype: str = "float32"
    compute_dtype: str = "float32"


@dataclass(frozen=True)
class MaterializeDense:
    """Decode/merge to a new full dense weight and drop rank/packing promises."""


@dataclass(frozen=True)
class AddResidual:
    """Add a rank-r weight correction toward this scope's original operator.

    The base, including its bias, remains in the deployed graph. No training
    is performed; the correction has no bias and is not an exactness promise.
    """
    rank: int


CompositionStep = PruneMask | LowRank | QuantizeRemapped | MaterializeDense | AddResidual


@dataclass(frozen=True)
class CompositionRecipe:
    path: str
    steps: tuple[CompositionStep, ...]


@dataclass(frozen=True)
class CompositionRequest:
    recipes: tuple[CompositionRecipe, ...]
    max_storage_bytes: int | None = None
    max_peak_bytes: int = 12 * 1024**3
    export_format: FinalFormat = FinalFormat.CHECKPOINT


@dataclass(frozen=True)
class OperatorState:
    representation: Representation
    in_features: int
    out_features: int
    dtype: str
    bias: bool
    rank: int | None
    storage_bytes: int
    runtime: str = "cpu_float"


@dataclass(frozen=True)
class CompositionTransition:
    recipe_index: int
    step_index: int
    path: str
    operation: CompositionStep
    input_state: OperatorState
    output_state: OperatorState


@dataclass(frozen=True)
class CheckedComposition:
    request: CompositionRequest
    source_fingerprint: str
    source_inventory: tuple[PhysicalStorage, ...]
    transitions: tuple[CompositionTransition, ...]
    expected_storage_bytes: int
    fingerprint: str
    schema_version: int = 1


@dataclass(frozen=True)
class CompositionEvidence:
    recipe_index: int
    step_index: int
    path: str
    operation: str
    input_representation: Representation
    output_representation: Representation
    input_storage_bytes: int
    output_storage_bytes: int
    semantics: str


@dataclass(frozen=True)
class CompositionSuccess:
    model: nn.Module
    workflow: CheckedComposition
    inventory: tuple[PhysicalStorage, ...]
    history: tuple[CompositionEvidence, ...]
    checkpoint_bytes: bytes

    @property
    def storage_bytes(self):
        return inventory_cost(self.inventory, "bytes")

    @property
    def fingerprint(self):
        return self.workflow.fingerprint


class _Refused(ValueError):
    def __init__(self, code, path, message):
        self.failure = ValidationFailure((Violation(code, path, message),))
        super().__init__(message)


def _refuse(code, path, message):
    raise _Refused(code, path, message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def _under(path, root):
    return root == "" or path == root or path.startswith(root + ".")


def _state_bytes(state):
    width = 8 if state.dtype == "float64" else 4
    bias = state.out_features * width if state.bias else 0
    if state.representation in (Representation.DENSE, Representation.DENSE_MASKED):
        return state.in_features * state.out_features * width + bias
    if state.representation is Representation.REAL_FACTORS:
        return state.rank * (state.in_features + state.out_features) * width + bias
    raise AssertionError("Packing and residual storage need their explicit component costs")


def _inspect_operator(layer, path, cost):
    path = path or "model"
    if type(layer) is nn.Linear:
        kind, dtype, bias, rank = Representation.DENSE, layer.weight.dtype, layer.bias is not None, None
        if (layer.weight.shape != (layer.out_features, layer.in_features)
                or layer.bias is not None and (layer.bias.shape != (layer.out_features,)
                    or layer.bias.dtype != layer.weight.dtype)
                or layer._modules):
            _refuse("UnsupportedCapability", path, "Linear must retain its declared dense operator topology")
    elif type(layer) is FactorizedLinear:
        if (type(layer.left) is not nn.Linear or type(layer.right) is not nn.Linear
                or set(layer._modules) != {"left", "right"}
                or layer.right.bias is not None
                or layer.left.weight.shape != (layer.out_features, layer.rank)
                or layer.right.weight.shape != (layer.rank, layer.in_features)
                or layer.left.weight.dtype != layer.right.weight.dtype
                or layer.left.bias is not None and (layer.left.bias.shape != (layer.out_features,)
                    or layer.left.bias.dtype != layer.left.weight.dtype)):
            _refuse("UnsupportedCapability", path, "Real factors require the declared two compatible Linear operators")
        kind, dtype, bias, rank = (Representation.REAL_FACTORS, layer.left.weight.dtype,
                                  layer.left.bias is not None, layer.rank)
    elif type(layer) is RemappedLinear:
        layer.validate_state(layer.state_dict())
        kind, dtype, bias, rank = (Representation.PACKED_FACTORS,
            getattr(torch, layer.compute_dtype), layer.bias is not None, layer.rank)
        if layer.bias_dtype != layer.compute_dtype:
            _refuse("UnsupportedCapability", path, "Composition requires the packed compute and bias dtype to agree")
    elif type(layer) is ResidualLinear:
        base = _inspect_operator(layer.base, path + ".base", 0)
        residual = _inspect_operator(layer.residual, path + ".residual", 0)
        if (set(layer._modules) != {"base", "residual"}
                or residual.representation is not Representation.REAL_FACTORS
                or residual.bias or residual.dtype != base.dtype
                or (residual.in_features, residual.out_features) != (base.in_features, base.out_features)
                or (layer.in_features, layer.out_features) != (base.in_features, base.out_features)):
            _refuse("UnsupportedCapability", path, "Residual branch dimensions, dtype and bias must match its base")
        kind, dtype, bias, rank = Representation.RESIDUAL_BRANCH, getattr(torch, base.dtype), base.bias, None
    else:
        _refuse("UnsupportedCapability", path,
            "Only dense Linear, real factors, portable DoBi factors and their explicit residual branch are supported; physical sparse and opaque packed backends require adapters")
    if dtype not in (torch.float32, torch.float64):
        _refuse("UnsupportedCapability", path, "Composition requires finite CPU float32/float64 operator arithmetic")
    if (any(type(value) is not int or value < 1 for value in (layer.in_features, layer.out_features))
            or layer.in_features * layer.out_features > 4 * 1024**2
            or max(layer.in_features, layer.out_features) > 8192):
        _refuse("ResourceLimitExceeded", path, "Operators are bounded to 4M matrix entries and axes of at most 8192")
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
           for value in (*layer.parameters(), *layer.buffers())):
        _refuse("InvalidSource", path, "Operator state must be finite")
    return OperatorState(kind, layer.in_features, layer.out_features, str(dtype)[6:], bias, rank, cost,
        "cpu_factor_decode" if kind is Representation.PACKED_FACTORS else "cpu_float")


def _transition(state, operation, path):
    """Pure capability transition; no model, tensors, callbacks or solver calls."""
    dense = state.representation in (Representation.DENSE, Representation.DENSE_MASKED)
    if type(operation) is PruneMask:
        indices = operation.zero_indices
        if (type(indices) is not tuple
                or any(type(index) is not int or not 0 <= index < state.in_features * state.out_features
                       for index in indices) or len(set(indices)) != len(indices)):
            _refuse("InvalidMask", path, "A tuple of distinct in-range row-major zero indices is required")
        if not dense:
            _refuse("UnsupportedTransition", path, "PruneMask requires dense storage; declare MaterializeDense before masking factors or a residual")
        output = replace(state, representation=Representation.DENSE_MASKED, rank=None)
        return replace(output, storage_bytes=_state_bytes(output))
    if type(operation) in (LowRank, AddResidual):
        if type(operation.rank) is not int or not 1 <= operation.rank <= min(state.in_features, state.out_features):
            _refuse("InvalidRank", path, "Rank must be a positive integer fitting both operator axes")
        if type(operation) is LowRank:
            if not dense:
                _refuse("UnsupportedTransition", path, "LowRank requires dense storage; declare MaterializeDense to decode or merge first")
            output = replace(state, representation=Representation.REAL_FACTORS, rank=operation.rank)
            return replace(output, storage_bytes=_state_bytes(output))
        if state.representation is Representation.RESIDUAL_BRANCH:
            _refuse("UnsupportedTransition", path, "Nested residual branches are outside this finite grammar")
        width = 8 if state.dtype == "float64" else 4
        return replace(state, representation=Representation.RESIDUAL_BRANCH, rank=None,
            storage_bytes=state.storage_bytes + operation.rank * (state.in_features + state.out_features) * width,
            runtime="cpu_float")
    if type(operation) is QuantizeRemapped:
        if state.representation not in (Representation.DENSE, Representation.DENSE_MASKED, Representation.REAL_FACTORS):
            _refuse("UnsupportedTransition", path, "DoBi remapping accepts a dense operator or real factors; declare MaterializeDense for other representations")
        if (operation.tail_dtype not in ("float16", "bfloat16", "float32", "float64")
                or operation.scale_dtype not in ("float32", "float64")
                or operation.compute_dtype != state.dtype):
            _refuse("UnsupportedCapability", path, "Explicit DoBi tail/scales are required and compute dtype must match the operator dtype")
        rank = state.rank if state.representation is Representation.REAL_FACTORS else min(state.in_features, state.out_features)
        cost = remapped_tensor_bytes(state.in_features, state.out_features, rank,
            tail_dtype=operation.tail_dtype, scale_dtype=operation.scale_dtype, bias_dtype=state.dtype, bias=state.bias)
        return replace(state, representation=Representation.PACKED_FACTORS, rank=rank,
                       storage_bytes=cost, runtime="cpu_factor_decode")
    if type(operation) is MaterializeDense:
        output = replace(state, representation=Representation.DENSE, rank=None, runtime="cpu_float")
        return replace(output, storage_bytes=_state_bytes(output))
    _refuse("UnsupportedCapability", path, "Unknown composition step; physical sparse, TorchAO, fake quantization and training are outside this finite grammar")


def _request(request):
    if type(request) is not CompositionRequest or not _metadata_only(request):
        _refuse("InvalidComposition", "request", "An immutable metadata-only CompositionRequest is required")
    if request.export_format != FinalFormat.CHECKPOINT or type(request.export_format) not in (str, FinalFormat):
        _refuse("UnsupportedExport", "request.export_format", "Only a verified weights-only checkpoint is supported")
    if (type(request.max_peak_bytes) is not int or not 0 < request.max_peak_bytes <= 12 * 1024**3
            or request.max_storage_bytes is not None and (type(request.max_storage_bytes) is not int or request.max_storage_bytes < 0)):
        _refuse("InvalidBudget", "request", "Declare a nonnegative tensor-byte budget and a positive peak limit of at most 12 GiB")
    if (type(request.recipes) is not tuple or not 1 <= len(request.recipes) <= 16
            or any(type(recipe) is not CompositionRecipe or type(recipe.path) is not str
                   or type(recipe.steps) is not tuple or not 1 <= len(recipe.steps) <= 8
                   for recipe in request.recipes)):
        _refuse("InvalidComposition", "request.recipes", "Declare 1..16 disjoint scope recipes with 1..8 ordered steps each")
    paths = tuple(recipe.path for recipe in request.recipes)
    if len(set(paths)) != len(paths) or any(_under(a, b) or _under(b, a)
                                           for index, a in enumerate(paths) for b in paths[index + 1:]):
        _refuse("OverlappingScopes", "request.recipes", "Composition scopes must be distinct and cannot contain one another")


def _source(model):
    from fedcore.tools.registry.checkpoint_manager import _describe_model
    if not isinstance(model, nn.Module):
        _refuse("InvalidSource", "model", "A torch Module is required")
    from torch.nn.modules import module as module_runtime
    if any(getattr(module_runtime, name, {}) for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
            "_global_backward_pre_hooks", "_global_module_registration_hooks",
            "_global_parameter_registration_hooks", "_global_buffer_registration_hooks")):
        _refuse("UnsupportedCapability", "model", "Global module hooks are outside the isolated composition runtime")
    paths = module_paths(model)
    # Check every raw child, including unexpected children that a primitive
    # constructor descriptor would otherwise omit. Do this before state_dict
    # or any constructor-specific representation methods can be invoked.
    from .structured_layers import SharedBasisLinear, GroupedEmbedding, GroupedLMHead
    from .coupled_layers import ReferenceAttention, ReferenceMLP
    from .learned_layers import ZeroLinear
    from .projection_layers import (FixedProjectorLinear, CenteredPCAEmbedding,
        CTRLinearSum, CTRFieldGraph, ReLUFactorizedLinear)
    constructors = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding, nn.ReLU,
        nn.Identity, nn.Sigmoid, nn.Tanh, nn.Flatten, nn.Dropout, nn.Sequential,
        nn.ModuleDict, nn.ParameterList, nn.ModuleList, FactorizedLinear,
        ResidualLinear, RemappedLinear, SharedBasisLinear, GroupedEmbedding,
        GroupedLMHead, ReferenceAttention, ReferenceMLP, ZeroLinear,
        FixedProjectorLinear, CenteredPCAEmbedding, CTRLinearSum, CTRFieldGraph,
        ReLUFactorizedLinear)
    methods = ("forward", "state_dict", "_save_to_state_dict", "_load_from_state_dict",
               "representation_config", "validate_state", "decoded_factors", "get_extra_state",
               "set_extra_state", "__deepcopy__", "__getstate__", "__setstate__")
    for path, layer in paths.items():
        if type(layer) not in constructors:
            _refuse("UnsupportedExport", path or "model", "This finite composition profile requires exact allowlisted CPU constructors, including all child modules")
        if type(layer) in (nn.ModuleDict, nn.ModuleList, nn.ParameterList) and (
                not path or type(paths[path.rpartition(".")[0]]) is nn.Sequential):
            _refuse("UnsupportedExport", path or "model", "A collection without forward cannot occupy an executable root or Sequential position")
        if any(callable(value) for value in vars(layer).values()) or any(name in vars(layer) for name in methods) or any(
                getattr(layer, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                    "_backward_pre_hooks", "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                    "_load_state_dict_post_hooks")):
            _refuse("UnsupportedCapability", path or "model", "Composition requires unmodified operators without runtime or state hooks")
        if any(name in layer._buffers and layer._buffers[name] is not None for name in layer._non_persistent_buffers_set):
            _refuse("UnsupportedExport", path or "model", "Nonpersistent tensor buffers cannot be silently dropped from the final checkpoint")
    # The public describe_model first reads state_dict. Establish its exact
    # constructor allowlist before an unknown subclass can supply state hooks.
    description = _describe_model(model, {}, "")
    if description is None:
        _refuse("UnsupportedExport", "model", "The source architecture has no complete allowlisted checkpoint constructor")
    inventory = capture_inventory(model)
    identity = _hash({"inventory": inventory_fingerprint(inventory), "architecture": description,
                      "training": [(path, layer.training) for path, layer in paths.items()]})
    return paths, inventory, identity


def _roundtrip(model, expected_inventory):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    manager = CheckpointManager(".", auto_cleanup=False)
    payload = manager.serialize_to_bytes(model)
    restored = manager.deserialize_from_bytes(payload, device="cpu")
    def signature(value):
        canonical, result = {}, []
        for path, layer in module_paths(value).items():
            canonical.setdefault(id(layer), path)
            result.append((path, type(layer), layer.training, canonical[id(layer)]))
        return tuple(result)
    if signature(model) != signature(restored):
        _refuse("UnsupportedExport", "model", "Checkpoint restoration must preserve the complete ordered module graph, modes and module aliases, including modules without tensor state")
    inventory = capture_inventory(restored)
    if inventory_fingerprint(inventory) != inventory_fingerprint(expected_inventory):
        _refuse("UnsupportedExport", "model", "Checkpoint restoration must preserve all physical tensor storage, aliases, dtypes and views")
    return payload, restored, inventory


def _prepare(model, request):
    _request(request)
    paths, inventory, source = _source(model)
    source_bytes = inventory_cost(inventory, "bytes")
    storage_guard(6 * source_bytes, request.max_peak_bytes)
    affected, transitions, removed, added = set(), [], set(), 0
    for recipe_index, recipe in enumerate(request.recipes):
        if recipe.path not in paths:
            _refuse("InvalidScope", recipe.path, "Unknown composition operator path")
        operator = paths[recipe.path]
        aliases = tuple(path for path, layer in paths.items() if layer is operator)
        roots = {path for path in paths if any(_under(path, alias) for alias in aliases)}
        if affected.intersection(roots):
            _refuse("OverlappingScopes", recipe.path, "Aliased scope recipes cannot overlap")
        affected.update(roots)
        records = []
        for record in inventory:
            hits = [any(_under(view.path, alias) for alias in aliases) for view in record.views]
            if any(hits):
                if not all(hits):
                    _refuse("UnsupportedSharing", recipe.path, "A transformed scope cannot share tensor storage with another scope; unchanged shared storage is counted once")
                records.append(record)
                removed.add(record.storage_id)
        state = _inspect_operator(operator, recipe.path, sum(record.nbytes for record in records))
        # SVD double workspaces, decoded matrices and an original-scope reference
        # coexist. This is a conservative declared floor, not a measured peak.
        entries = state.in_features * state.out_features
        storage_guard(6 * source_bytes + 64 * (entries + min(state.in_features, state.out_features)**2), request.max_peak_bytes)
        for step_index, operation in enumerate(recipe.steps):
            output = _transition(state, operation, f"{recipe.path}.steps[{step_index}]")
            transitions.append(CompositionTransition(recipe_index, step_index, recipe.path, operation, state, output))
            state = output
        added += state.storage_bytes
    expected = sum(record.nbytes for record in inventory if record.storage_id not in removed) + added
    if request.max_storage_bytes is not None and expected > request.max_storage_bytes:
        _refuse("BudgetExceeded", "request.max_storage_bytes", f"Final physical tensor storage needs {expected} bytes including packed codes, scales, tails, bias, maps, indices, residuals and unchanged shared storage")
    # Prove that unchanged state is representable before running any transform.
    with torch.random.fork_rng(devices=[]):
        _roundtrip(model, inventory)
    fingerprint = _hash({"request": asdict(request), "source": source,
                         "transitions": [asdict(item) for item in transitions], "bytes": expected})
    return CheckedComposition(request, source, inventory, tuple(transitions), expected, fingerprint)


def prepare_composition(model, request):
    """Bind a bounded representation grammar to a full physical source snapshot.

    Does not execute model forward, pruning, SVD or remapping. The checkpoint
    constructor round trip is read-only and runs with isolated Torch RNG.
    """
    try:
        return _prepare(model, request)
    except _Refused as error:
        return error.failure
    except CandidateStorageError as error:
        return ValidationFailure(error.violations)
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError, OverflowError) as error:
        return ValidationFailure((Violation("UnsupportedExport", "model", str(error)),))


def _weight_bias(layer):
    if type(layer) is nn.Linear:
        return layer.weight.detach(), None if layer.bias is None else layer.bias.detach()
    if type(layer) is FactorizedLinear:
        return layer.left.weight.detach() @ layer.right.weight.detach(), None if layer.left.bias is None else layer.left.bias.detach()
    if type(layer) is RemappedLinear:
        left, right = layer.decoded_factors()
        return left @ right, None if layer.bias is None else layer.bias.detach().to(left)
    if type(layer) is ResidualLinear:
        base, bias = _weight_bias(layer.base)
        residual, _ = _weight_bias(layer.residual)
        return base + residual, bias
    _refuse("UnsupportedCapability", "operator", "Operator has no checked dense reconstruction")


def _factors(weight, rank):
    # Reuse the existing structured-profile solver and real factor container.
    u, singular, vh = _svd(weight.double(), "composition Frobenius operator")
    return FactorPair((u[:, :rank] * singular[:rank]).to(weight), vh[:rank].to(weight))


def _dense(weight, bias):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


def _apply(layer, operation, original):
    weight, bias = _weight_bias(layer)
    if type(operation) is PruneMask:
        # The standard pruning kernel is temporary; remove leaves dense zeros.
        output = _dense(weight, bias)
        mask = torch.ones_like(weight)
        mask.view(-1)[list(operation.zero_indices)] = 0
        prune.custom_from_mask(output, "weight", mask)
        prune.remove(output, "weight")
        semantics = "explicit_mask_materialized_as_dense_zeros; no_sparse_storage_or_persistent_zero_pattern_promise"
    elif type(operation) is LowRank:
        output = FactorizedLinear.from_factors(_factors(weight, operation.rank), bias)
        semantics = "rank_bounded_real_factors; prior_zero_pattern_is_not_preserved"
    elif type(operation) is QuantizeRemapped:
        rank = layer.rank if type(layer) is FactorizedLinear else min(weight.shape)
        output = RemappedLinear.from_weight(weight, rank, bias, tail_dtype=operation.tail_dtype,
            scale_dtype=operation.scale_dtype, compute_dtype=operation.compute_dtype)
        semantics = "reconstruct_then_SVD_then_existing_DoBi_factor_remapping; packed_codes_scales_and_tail; float_decode_runtime"
    elif type(operation) is MaterializeDense:
        output = _dense(weight, bias)
        semantics = "full_dense_decode_or_residual_merge; factor_rank_and_packed_savings_promises_dropped"
    else:
        pair = _factors(original.to(weight) - weight, operation.rank)
        output = ResidualLinear(layer, pair, freeze_base=True)
        semantics = "real_factor_weight_error_correction_toward_initial_scope; base_and_bias_retained; no_training_or_exactness_promise"
    output.eval()
    output.requires_grad_(False)
    return output, semantics


@torch.no_grad()
def execute_composition(model, workflow):
    """Execute checked transitions on owned state and publish a reloadable model.

    Revalidation rejects stale source or forged plans before copying/transforming.
    Success owns both the deployed model and its verified checkpoint bytes.
    """
    if type(workflow) is not CheckedComposition or not _metadata_only(workflow):
        return ValidationFailure((Violation("InvalidComposition", "workflow", "A checked metadata-only composition is required"),))
    checked = prepare_composition(model, workflow.request)
    if type(checked) is ValidationFailure:
        return checked
    if checked != workflow:
        return ValidationFailure((Violation("StaleComposition", "workflow", "Source snapshot, capabilities or checked transitions changed"),))
    try:
        with torch.random.fork_rng(devices=[]):
            work = clone_preserving_storages(model, max_peak_bytes=workflow.request.max_peak_bytes)
            paths, replacements, history = module_paths(work), {}, []
            for recipe_index, recipe in enumerate(workflow.request.recipes):
                layer = paths[recipe.path]
                original, _ = _weight_bias(layer)
                original = original.detach().clone()
                for transition in (item for item in workflow.transitions if item.recipe_index == recipe_index):
                    layer, semantics = _apply(layer, transition.operation, original)
                    actual = _inspect_operator(layer, recipe.path, inventory_cost(capture_inventory(layer), "bytes"))
                    expected = transition.output_state
                    if expected.representation is Representation.DENSE_MASKED:
                        actual = replace(actual, representation=Representation.DENSE_MASKED)
                    if actual != expected:
                        _refuse("RepresentationMismatch", recipe.path, "Executed representation or physical storage differs from its checked transition")
                    history.append(CompositionEvidence(recipe_index, transition.step_index, recipe.path,
                        type(transition.operation).__name__, transition.input_state.representation,
                        expected.representation, transition.input_state.storage_bytes, actual.storage_bytes, semantics))
                replacements[recipe.path] = layer
            work = replace_modules_atomically(work, replacements)
            inventory = capture_inventory(work)
            cost = inventory_cost(inventory, "bytes")
            if cost != workflow.expected_storage_bytes:
                _refuse("RepresentationMismatch", "model", "Final physical storage differs from the complete checked inventory cost")
            if workflow.request.max_storage_bytes is not None and cost > workflow.request.max_storage_bytes:
                _refuse("BudgetExceeded", "model", "Final deployed tensor storage exceeds its byte budget")
            work._fedcore_method_evidence = {"schema": "composition-v1", "fingerprint": workflow.fingerprint,
                "source_fingerprint": workflow.source_fingerprint, "history": [asdict(item) for item in history],
                "storage_bytes": cost, "artifact": "verified_weights_only_checkpoint"}
            payload, restored, inventory = _roundtrip(work, inventory)
            # Exercise the restored runtime of each replaced operator. Inputs
            # are deterministic and bounded; no calibration or RNG is needed.
            restored_paths, work_paths = module_paths(restored), module_paths(work)
            for recipe in workflow.request.recipes:
                final = restored_paths[recipe.path]
                state = next(item.output_state for item in reversed(workflow.transitions)
                             if item.path == recipe.path)
                probe = torch.arange(state.in_features, dtype=getattr(torch, state.dtype))
                probe = (probe / state.in_features).repeat(2, 1)
                expected_output, output = work_paths[recipe.path](probe), final(probe)
                if (output.shape != (2, state.out_features) or not bool(torch.isfinite(output).all())
                        or not torch.equal(output, expected_output)):
                    _refuse("UnsupportedExport", recipe.path, "The restored operator must execute identically on the deterministic runtime probe")
            restored._fedcore_requires_optimizer_rebuild = True
            return CompositionSuccess(restored, workflow, inventory, tuple(history), payload)
    except _Refused as error:
        return error.failure
    except CandidateStorageError as error:
        return ValidationFailure(error.violations)
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError, OverflowError) as error:
        return ValidationFailure((Violation("CompositionExecutionFailed", "model", str(error)),))
