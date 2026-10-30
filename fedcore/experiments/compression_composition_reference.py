"""Small independent G5 witness for the frozen issue-99 local protocol.

This is an experimental interpreter, not a public validation bypass. Its
literal capability table, byte formulas, tensor traversal and primitive
implementations do not use the production planner, executor or inventory.
The early and late timing arms must call this same interpreter. An early arm
may call the production admission boundary before calling it.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.compression_composition import (
    AddResidual, CompositionRecipe, CompositionRequest, LowRank,
    MaterializeDense, PruneMask, QuantizeRemapped,
)
from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear, ResidualLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair


# Published local contract, deliberately expressed without production enums.
CAPABILITIES = {
    "dense": frozenset("PLQDR"),
    "dense_masked": frozenset("PLQDR"),
    "real_factors": frozenset("QDR"),
    "packed_factors": frozenset("DR"),
    "residual_branch": frozenset("D"),
}
_SYMBOLS = {PruneMask: "P", LowRank: "L", QuantizeRemapped: "Q",
            MaterializeDense: "D", AddResidual: "R"}
_WIDTH = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


@dataclass(frozen=True)
class ReferenceRejection:
    category: str
    code: str
    path: str
    message: str
    recipe_index: int | None = None
    step_index: int | None = None


@dataclass(frozen=True)
class ReferenceState:
    representation: str
    in_features: int
    out_features: int
    dtype: str
    bias: bool
    rank: int | None
    storage_bytes: int


@dataclass(frozen=True)
class ReferenceContract:
    accepted: bool
    rejection: ReferenceRejection | None
    final_states: tuple[tuple[str, ReferenceState], ...] = ()
    expected_storage_bytes: int | None = None


@dataclass(frozen=True)
class ReferenceEvent:
    recipe_index: int
    step_index: int
    path: str
    operation: str
    input_representation: str
    output_representation: str | None
    completed: bool
    check_seconds: float
    numerical_seconds: float
    verification_seconds: float
    decompositions_started: int


@dataclass(frozen=True)
class ReferenceTimings:
    admission_seconds: float = 0.0
    copy_seconds: float = 0.0
    numerical_seconds: float = 0.0
    verification_seconds: float = 0.0
    rejection_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class ReferenceExecution:
    model: nn.Module | None
    rejection: ReferenceRejection | None
    events: tuple[ReferenceEvent, ...]
    timings: ReferenceTimings
    completed_steps: int
    decompositions_started: int
    storage_bytes: int | None
    final_states: tuple[tuple[str, ReferenceState], ...] = ()

    @property
    def accepted(self):
        return self.rejection is None


class _Reject(Exception):
    def __init__(self, category, code, path, message):
        self.rejection = ReferenceRejection(category, code, path or "model", message)
        super().__init__(message)


def _reject(code, path, message, category="structural"):
    raise _Reject(category, code, path, message)


def _paths(model):
    """Traverse raw registrations, including aliases, without user callbacks."""
    result = {}

    def visit(layer, path, ancestors):
        if id(layer) in ancestors:
            _reject("InvalidSource", path, "Cyclic module graphs are unsupported")
        result[path] = layer
        for name, child in layer._modules.items():
            if child is not None:
                visit(child, f"{path}.{name}" if path else name, ancestors | {id(layer)})

    visit(model, "", set())
    return result


def independent_inventory(model):
    """Return (physical bytes, canonical physical-storage/view signatures).

    Real backing allocation sizes are counted, once per CPU storage. Signatures
    use encounter-order numbers so independently copied models can be compared.
    """
    storages, views = {}, []
    for path, layer in _paths(model).items():
        for kind, collection in (("parameter", layer._parameters), ("buffer", layer._buffers)):
            for name, value in collection.items():
                if value is None:
                    continue
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.layout != torch.strided):
                    _reject("UnsupportedCapability", path, "Only strided CPU tensor state is supported")
                storage = value.untyped_storage()
                identity = (str(value.device), storage._cdata)
                if identity not in storages:
                    storages[identity] = (len(storages), storage.nbytes())
                number, size = storages[identity]
                views.append((path, kind, name, number, size, str(value.dtype),
                              tuple(value.shape), tuple(value.stride()), value.storage_offset()))
    return sum(size for _, size in storages.values()), tuple(views)


def dense_weight_bias(layer):
    """Decode registered components directly, never a production decoder."""
    if type(layer) is nn.Linear:
        return layer.weight.detach(), None if layer.bias is None else layer.bias.detach()
    if type(layer) is FactorizedLinear:
        return (layer.left.weight.detach() @ layer.right.weight.detach(),
                None if layer.left.bias is None else layer.left.bias.detach())
    if type(layer) is RemappedLinear:
        dtype = getattr(torch, layer.compute_dtype)
        left = layer.left_codes.to(dtype) * layer.left_scale.to(dtype)[None, :]
        right = layer.right_codes.to(dtype) * layer.right_scale.to(dtype)[:, None]
        if layer.out_features >= layer.in_features:
            left = torch.cat((left, layer.tail.to(dtype)), dim=0)
        else:
            right = torch.cat((right, layer.tail.to(dtype)), dim=1)
        return left @ right, None if layer.bias is None else layer.bias.detach().to(dtype)
    if type(layer) is ResidualLinear:
        base, bias = dense_weight_bias(layer.base)
        residual, _ = dense_weight_bias(layer.residual)
        return base + residual, bias
    _reject("UnsupportedCapability", "operator", "No independent dense interpretation exists")


def _state(layer, path):
    """Check actual shapes, precision and packed components independently."""
    size, _ = independent_inventory(layer)
    if type(layer) is nn.Linear:
        n, m, dtype, bias, rank, kind = (layer.in_features, layer.out_features,
            str(layer.weight.dtype)[6:], layer.bias, None, "dense")
        if (set(layer._parameters) != {"weight", "bias"} or layer._buffers or layer._modules
                or layer.weight.shape != (m, n)):
            _reject("UnsupportedCapability", path, "Malformed dense Linear topology")
    elif type(layer) is FactorizedLinear:
        if (set(layer._modules) != {"left", "right"} or type(layer.left) is not nn.Linear
                or type(layer.right) is not nn.Linear or layer.right.bias is not None):
            _reject("UnsupportedCapability", path, "Expected two explicit real factors")
        n, m, rank = layer.in_features, layer.out_features, layer.rank
        dtype, bias, kind = str(layer.left.weight.dtype)[6:], layer.left.bias, "real_factors"
        if (type(rank) is not int or not 1 <= rank <= min(n, m)
                or layer.left.weight.shape != (m, rank) or layer.right.weight.shape != (rank, n)
                or layer.left.weight.dtype != layer.right.weight.dtype):
            _reject("UnsupportedCapability", path, "Malformed factor dimensions or precision")
    elif type(layer) is RemappedLinear:
        n, m, rank = layer.in_features, layer.out_features, layer.rank
        dtype, bias, kind = layer.compute_dtype, layer.bias, "packed_factors"
        if (dtype not in ("float32", "float64") or layer.bias_dtype != dtype
                or layer.scale_dtype not in ("float32", "float64")
                or layer.tail_dtype not in _WIDTH or type(rank) is not int
                or not 1 <= rank <= min(n, m) or layer.tall != (m >= n)):
            _reject("UnsupportedCapability", path, "Malformed packed precision or dimensions")
        short = min(m, n)
        expected = {
            "left_codes": ((short, rank), torch.int8),
            "right_codes": ((rank, short), torch.int8),
            "left_scale": ((rank,), getattr(torch, layer.scale_dtype)),
            "right_scale": ((rank,), getattr(torch, layer.scale_dtype)),
            "tail": (((m-short, rank) if m >= n else (rank, n-short)), getattr(torch, layer.tail_dtype)),
        }
        if set(layer._buffers) != set(expected) or layer._modules:
            _reject("UnsupportedCapability", path, "Packed components must be explicit named buffers")
        component_storages = set()
        for name, (shape, precision) in expected.items():
            tensor = layer._buffers[name]
            if tensor.shape != shape or tensor.dtype != precision:
                _reject("UnsupportedCapability", path, "Packed component shape or dtype differs")
            storage = tensor.untyped_storage()
            if (not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or storage.nbytes() != tensor.numel()*tensor.element_size()
                    or storage.nbytes() and storage._cdata in component_storages):
                _reject("UnsupportedCapability", path, "Each packed component must own a minimal contiguous storage")
            if storage.nbytes():
                component_storages.add(storage._cdata)
            if name.endswith("codes") and bool((tensor == -128).any()):
                _reject("UnsupportedCapability", path, "Packed codes must lie in [-127,127]")
            if name.endswith("scale") and bool((tensor.to(getattr(torch, dtype)) <= 0).any()):
                _reject("UnsupportedCapability", path, "Packed component scales must remain positive")
    elif type(layer) is ResidualLinear:
        base, correction = _state(layer.base, path + ".base"), _state(layer.residual, path + ".residual")
        if (set(layer._modules) != {"base", "residual"} or correction.representation != "real_factors"
                or correction.bias or base.dtype != correction.dtype
                or (base.in_features, base.out_features) != (correction.in_features, correction.out_features)
                or (layer.in_features, layer.out_features) != (base.in_features, base.out_features)):
            _reject("UnsupportedCapability", path, "Residual must retain a compatible bias-free correction")
        return ReferenceState("residual_branch", base.in_features, base.out_features,
                              base.dtype, base.bias, None, size)
    else:
        _reject("UnsupportedCapability", path, "The local reference supports Linear-family scopes only")
    if (type(n) is not int or type(m) is not int or min(n, m) < 1
            or n*m > 4*1024**2 or max(n, m) > 8192):
        _reject("ResourceLimitExceeded", path, "Matrix exceeds the declared local primitive limits", "resource")
    if dtype not in ("float32", "float64"):
        _reject("UnsupportedCapability", path, "Operator arithmetic must be float32 or float64")
    if bias is not None and (bias.shape != (m,) or str(bias.dtype)[6:] != dtype):
        _reject("UnsupportedCapability", path, "Bias must retain output shape and operator precision")
    return ReferenceState(kind, n, m, dtype, bias is not None, rank, size)


def _under(path, root):
    return root == "" or path == root or path.startswith(root + ".")


def _request_source(request, model):
    if type(request) is not CompositionRequest or type(request.recipes) is not tuple:
        _reject("InvalidComposition", "request", "Expected the immutable composition request schema")
    if (type(request.max_peak_bytes) is not int or not 0 < request.max_peak_bytes <= 12*1024**3
            or request.max_storage_bytes is not None and
            (type(request.max_storage_bytes) is not int or request.max_storage_bytes < 0)):
        _reject("InvalidBudget", "request", "Invalid peak or final tensor storage budget")
    if not 1 <= len(request.recipes) <= 16 or any(
            type(recipe) is not CompositionRecipe or type(recipe.path) is not str
            or type(recipe.steps) is not tuple or not 1 <= len(recipe.steps) <= 8
            for recipe in request.recipes):
        _reject("InvalidComposition", "request.recipes", "Expected 1..16 recipes of 1..8 steps")
    scopes = tuple(recipe.path for recipe in request.recipes)
    if any(_under(a, b) or _under(b, a) for i, a in enumerate(scopes) for b in scopes[i+1:]):
        _reject("OverlappingScopes", "request.recipes", "Scope paths must be disjoint")
    if not isinstance(model, nn.Module):
        _reject("InvalidSource", "model", "A module is required")
    paths = _paths(model)
    allowed = (nn.Sequential, nn.Linear, nn.ReLU, nn.Identity, FactorizedLinear, RemappedLinear, ResidualLinear)
    for path, layer in paths.items():
        if type(layer) not in allowed:
            _reject("UnsupportedExport", path, "Source is outside the bounded local constructor profile")
        if any(callable(value) for value in vars(layer).values()) or any(
                getattr(layer, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks")):
            _reject("UnsupportedCapability", path, "Source operators must have no instance callbacks or hooks")
        if layer._non_persistent_buffers_set:
            _reject("UnsupportedExport", path, "Nonpersistent buffers are outside this reference profile")
    source_bytes, inventory = independent_inventory(model)
    for path, layer in paths.items():
        for tensor in (*layer._parameters.values(), *layer._buffers.values()):
            if tensor is None:
                continue
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                _reject("InvalidSource", path, "Source tensors must be finite")
            if (not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or tensor.untyped_storage().nbytes() != tensor.numel()*tensor.element_size()):
                _reject("UnsupportedExport", path, "The local source requires complete contiguous backing storages")
    states, affected = [], set()
    for recipe in request.recipes:
        if recipe.path not in paths:
            _reject("InvalidScope", recipe.path, "Unknown scope")
        aliases = tuple(path for path, layer in paths.items() if layer is paths[recipe.path])
        if len(aliases) != 1:
            _reject("UnsupportedSharing", recipe.path, "Aliased transformed modules are outside the local profile")
        selected = {view[3] for view in inventory if _under(view[0], recipe.path)}
        if affected & selected or any(view[3] in selected and not _under(view[0], recipe.path) for view in inventory):
            _reject("UnsupportedSharing", recipe.path, "Transformed scopes may not share tensor storage")
        affected.update(selected)
        state = _state(paths[recipe.path], recipe.path)
        # This fixed floor is the package's common safe admission limit. It is
        # not a measured RSS claim and is enforced by both timing branches.
        floor = 6*source_bytes + 64*(state.in_features*state.out_features + min(state.in_features, state.out_features)**2)
        if floor > request.max_peak_bytes:
            _reject("ResourceLimitExceeded", recipe.path, "Declared peak limit is below the common safe workspace floor", "resource")
        states.append((recipe.path, state))
    return paths, source_bytes, tuple(states)


def _next_state(state, operation, path):
    """A literal capability lookup followed by independent parameter rules."""
    symbol = _SYMBOLS.get(type(operation))
    if symbol is None:
        _reject("UnsupportedCapability", path, "Unknown primitive")
    if symbol not in CAPABILITIES[state.representation]:
        _reject("UnsupportedTransition", path, f"{symbol} is unavailable from {state.representation}")
    n, m, width = state.in_features, state.out_features, _WIDTH[state.dtype]
    bias_bytes = m*width if state.bias else 0
    dense_bytes = n*m*width + bias_bytes
    rank, kind, size = None, "dense", dense_bytes
    if symbol == "P":
        indices = operation.zero_indices
        if (type(indices) is not tuple or any(type(i) is not int or not 0 <= i < n*m for i in indices)
                or len(set(indices)) != len(indices)):
            _reject("InvalidMask", path, "Mask coordinates must be distinct in-range row-major integers")
        kind = "dense_masked"
    elif symbol in ("L", "R"):
        if type(operation.rank) is not int or not 1 <= operation.rank <= min(n, m):
            _reject("InvalidRank", path, "Requested rank must fit both axes exactly")
        rank = operation.rank
        if symbol == "L":
            kind, size = "real_factors", rank*(n+m)*width + bias_bytes
        else:
            kind, size, rank = "residual_branch", state.storage_bytes + rank*(n+m)*width, None
    elif symbol == "Q":
        if (operation.tail_dtype not in _WIDTH or operation.scale_dtype not in ("float32", "float64")
                or operation.compute_dtype != state.dtype):
            _reject("UnsupportedCapability", path, "Packed precision must be explicit and compute dtype must match source")
        rank = state.rank if state.representation == "real_factors" else min(n, m)
        kind = "packed_factors"
        size = 2*min(n, m)*rank + abs(n-m)*rank*_WIDTH[operation.tail_dtype] + 2*rank*_WIDTH[operation.scale_dtype] + bias_bytes
    return ReferenceState(kind, n, m, state.dtype, state.bias, rank, size)


def _publication(request, actual_bytes):
    if not isinstance(request.export_format, str) or request.export_format != "checkpoint":
        _reject("UnsupportedExport", "request.export_format", "Only checkpoint publication is supported")
    if request.max_storage_bytes is not None and actual_bytes > request.max_storage_bytes:
        _reject("BudgetExceeded", "request.max_storage_bytes", "Final physical storage exceeds the byte limit", "resource")


def inspect_contract(request, model):
    """Predict the literal contract without cloning, SVD or forward calls."""
    try:
        _, total, states = _request_source(request, model)
        final = []
        for recipe, (path, state) in zip(request.recipes, states):
            original_bytes = state.storage_bytes
            for index, operation in enumerate(recipe.steps):
                state = _next_state(state, operation, f"{path}.steps[{index}]")
            total += state.storage_bytes - original_bytes
            final.append((path, state))
        _publication(request, total)
        return ReferenceContract(True, None, tuple(final), total)
    except _Reject as error:
        return ReferenceContract(False, error.rejection)


def _dense(weight, bias):
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, dtype=weight.dtype)
    result.weight.copy_(weight)
    if bias is not None:
        result.bias.copy_(bias)
    return result


def _svd_pair(weight, rank, on_decomposition):
    if not bool(torch.isfinite(weight).all()):
        _reject("NonfinitePrimitive", "SVD", "SVD input is nonfinite", "numerical")
    on_decomposition()
    u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    # Rank was checked before this point; slicing is never rank validation.
    left, right = (u[:, :rank]*s[:rank]).to(weight.dtype), vh[:rank].to(weight.dtype)
    if not all(bool(torch.isfinite(value).all()) for value in (u, s, vh, left, right)):
        _reject("NonfinitePrimitive", "SVD", "SVD factors overflow the declared precision", "numerical")
    return FactorPair(left, right)


def _pack(weight, bias, rank, operation, on_decomposition):
    """Independent published symmetric int8 encoder, using the runtime class."""
    m, n = weight.shape
    result = RemappedLinear(n, m, rank, bias is not None, tail_dtype=operation.tail_dtype,
        scale_dtype=operation.scale_dtype, compute_dtype=operation.compute_dtype, bias_dtype=str(weight.dtype)[6:])
    if not bool(torch.isfinite(weight).all()):
        _reject("NonfinitePrimitive", "Q", "Packing input is nonfinite", "numerical")
    on_decomposition()
    u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    if m >= n:
        left, right = u[:, :rank]*s[:rank], vh[:rank]
        tail, left = left[n:], left[:n]
    else:
        left, right = u[:, :rank], s[:rank, None]*vh[:rank]
        tail, right = right[:, m:], right[:, :m]
    for name, block, axis in (("left", left, 0), ("right", right, 1)):
        maxima = block.abs().amax(dim=axis)
        scale = torch.where(maxima == 0, torch.ones_like(maxima), maxima/127).to(getattr(torch, operation.scale_dtype))
        core_scale = scale.to(getattr(torch, operation.compute_dtype))
        if not bool(torch.isfinite(core_scale).all()) or bool((core_scale <= 0).any()):
            _reject("NonfinitePrimitive", "Q", "Scale overflow or underflow", "numerical")
        codes = torch.round(block/scale.double().unsqueeze(axis)).clamp(-127, 127).to(torch.int8)
        setattr(result, name + "_codes", codes.contiguous())
        setattr(result, name + "_scale", scale.contiguous())
    result.tail = tail.to(getattr(torch, operation.tail_dtype)).clone(memory_format=torch.contiguous_format)
    if bias is not None:
        result.bias.copy_(bias)
    return result


def _primitive(layer, operation, original, state, on_decomposition):
    weight, bias = dense_weight_bias(layer)
    symbol = _SYMBOLS[type(operation)]
    if symbol == "P":
        masked = weight.clone()
        masked.reshape(-1)[list(operation.zero_indices)] = 0
        result = _dense(masked, bias)
    elif symbol == "L":
        result = FactorizedLinear.from_factors(_svd_pair(weight, operation.rank, on_decomposition), bias)
    elif symbol == "Q":
        result = _pack(weight, bias, state.rank, operation, on_decomposition)
    elif symbol == "D":
        result = _dense(weight, bias)
    else:
        result = ResidualLinear(layer, _svd_pair(original.to(weight.dtype)-weight, operation.rank, on_decomposition), freeze_base=True)
    result.eval()
    result.requires_grad_(False)
    return result


def _verify(layer, expected, original_bias, path):
    actual = _state(layer, path)
    if expected.representation == "dense_masked":
        actual = ReferenceState("dense_masked", actual.in_features, actual.out_features,
            actual.dtype, actual.bias, actual.rank, actual.storage_bytes)
    if actual != expected:
        _reject("RepresentationMismatch", path, "Primitive shapes, precision, rank or physical bytes differ from the independent contract", "shell")
    for module in _paths(layer).values():
        for value in (*module._parameters.values(), *module._buffers.values()):
            if value is not None and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                _reject("NonfinitePrimitive", path, "Primitive produced nonfinite tensor state", "numerical")
    weight, bias = dense_weight_bias(layer)
    if not bool(torch.isfinite(weight).all()):
        _reject("NonfinitePrimitive", path, "Decoded matrix is nonfinite", "numerical")
    if (bias is None) != (original_bias is None) or bias is not None and not torch.equal(bias, original_bias):
        _reject("BiasMismatch", path, "Every primitive must retain the original bias exactly", "shell")
    probe = torch.arange(2*expected.in_features, dtype=weight.dtype).reshape(2, -1)/expected.in_features
    runtime, literal = layer(probe), F.linear(probe, weight, bias)
    tolerance = (2e-12, 2e-12) if weight.dtype == torch.float64 else (3e-5, 3e-6)
    if not bool(torch.isfinite(runtime).all()):
        _reject("NonfinitePrimitive", path, "Runtime probe is nonfinite", "numerical")
    if runtime.shape != literal.shape or not torch.allclose(runtime, literal, rtol=tolerance[0], atol=tolerance[1]):
        _reject("RuntimeMismatch", path, "Runtime differs from the independently decoded affine map", "shell")


def _replace(model, path, layer):
    if not path:
        return layer
    parent_path, _, name = path.rpartition(".")
    _paths(model)[parent_path]._modules[name] = layer
    return model


@torch.no_grad()
def execute_reference(model, request):
    """Execute until a reached operation refuses; preserve source and Torch RNG.

    There is intentionally no call to inspect_contract here. In particular, a
    bad last operation does not erase work already done by its valid prefix,
    and an unsupported export is refused only at publication.
    """
    start = perf_counter()
    durations = dict(admission=0., copy=0., numerical=0., verification=0., rejection=0.)

    @contextmanager
    def measured(phase):
        before = perf_counter()
        try:
            yield
        finally:
            durations[phase] += perf_counter()-before

    events, final, work, rejection = [], [], None, None
    completed = decompositions = 0
    recipe_index = step_index = None
    try:
        with measured("admission"):
            _, source_bytes, states = _request_source(request, model)
        with torch.random.fork_rng(devices=[]):
            with measured("copy"):
                work = deepcopy(model)
            expected_total = source_bytes
            for recipe_index, (recipe, (path, state)) in enumerate(zip(request.recipes, states)):
                layer = _paths(work)[path]
                original, original_bias = dense_weight_bias(layer)
                original = original.clone()
                original_bias = None if original_bias is None else original_bias.clone()
                old_bytes = state.storage_bytes
                for step_index, operation in enumerate(recipe.steps):
                    check_before = durations["admission"]
                    try:
                        with measured("admission"):
                            next_state = _next_state(state, operation, f"{path}.steps[{step_index}]")
                    except _Reject:
                        elapsed = durations["admission"]-check_before
                        events.append(ReferenceEvent(recipe_index, step_index, path,
                            _SYMBOLS.get(type(operation), "?"), state.representation, None, False, elapsed, 0., 0., 0))
                        raise
                    check_seconds = durations["admission"]-check_before
                    step_decompositions = [0]

                    def decomposition_started():
                        nonlocal decompositions
                        step_decompositions[0] += 1
                        decompositions += 1

                    primitive_before = durations["numerical"]
                    verify_before = durations["verification"]
                    try:
                        with measured("numerical"):
                            next_layer = _primitive(layer, operation, original, next_state, decomposition_started)
                        with measured("verification"):
                            _verify(next_layer, next_state, original_bias, path)
                    except BaseException:
                        elapsed = durations["numerical"]-primitive_before
                        verify_elapsed = durations["verification"]-verify_before
                        events.append(ReferenceEvent(recipe_index, step_index, path, _SYMBOLS[type(operation)],
                            state.representation, None, False, check_seconds, elapsed, verify_elapsed, step_decompositions[0]))
                        raise
                    primitive_seconds = durations["numerical"]-primitive_before
                    verify_seconds = durations["verification"]-verify_before
                    events.append(ReferenceEvent(recipe_index, step_index, path, _SYMBOLS[type(operation)],
                        state.representation, next_state.representation, True, check_seconds, primitive_seconds, verify_seconds, step_decompositions[0]))
                    layer, state = next_layer, next_state
                    completed += 1
                work = _replace(work, path, layer)
                expected_total += state.storage_bytes-old_bytes
                final.append((path, state))
            with measured("verification"):
                actual_bytes, _ = independent_inventory(work)
                if actual_bytes != expected_total:
                    _reject("RepresentationMismatch", "model", "Whole-model physical bytes differ from the contract", "shell")
                _publication(request, actual_bytes)
    except _Reject as error:
        with measured("rejection"):
            value = error.rejection
            rejection = ReferenceRejection(value.category, value.code, value.path, value.message, recipe_index, step_index)
    except MemoryError as error:
        rejection = ReferenceRejection("resource", "ResourceLimitExceeded", "model", str(error), recipe_index, step_index)
    except (NumericalDomainViolation, torch.linalg.LinAlgError) as error:
        rejection = ReferenceRejection("numerical", "NumericalFailure", "model", str(error), recipe_index, step_index)
    except RuntimeError as error:
        # Torch 2.2 reports CPU allocator exhaustion as RuntimeError and has no
        # torch.OutOfMemoryError. Keep the recognized allocator surface narrow;
        # an arbitrary RuntimeError must remain an unknown shell failure.
        message = str(error)
        allocation = any(marker in message for marker in (
            "DefaultCPUAllocator: can't allocate memory", "DefaultCPUAllocator: not enough memory",
            "CPUAllocator: not enough memory"))
        rejection = ReferenceRejection("resource" if allocation else "shell",
            "ResourceLimitExceeded" if allocation else "ReferenceExecutionFailed", "model", message, recipe_index, step_index)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        rejection = ReferenceRejection("shell", "ReferenceExecutionFailed", "model", str(error), recipe_index, step_index)
    if rejection is not None:
        work = None
    timings = ReferenceTimings(durations["admission"], durations["copy"], durations["numerical"],
                              durations["verification"], durations["rejection"], perf_counter()-start)
    return ReferenceExecution(work, rejection, tuple(events), timings, completed, decompositions,
                              None if work is None else actual_bytes, tuple(final))


@torch.no_grad()
def assert_reference_parity(reference, candidate, request, *, rtol=None, atol=None):
    """Compare production or restored results against this independent witness."""
    if type(reference) is ReferenceExecution:
        if not reference.accepted:
            raise AssertionError(reference.rejection)
        reference = reference.model
    if independent_inventory(reference) != independent_inventory(candidate):
        raise AssertionError("Physical tensor layouts, aliases, precision or storage bytes differ")
    expected_paths, actual_paths = _paths(reference), _paths(candidate)
    if tuple((path, type(layer), layer.training) for path, layer in expected_paths.items()) != tuple(
            (path, type(layer), layer.training) for path, layer in actual_paths.items()):
        raise AssertionError("Complete module topology or training modes differ")
    for recipe in request.recipes:
        expected_weight, expected_bias = dense_weight_bias(expected_paths[recipe.path])
        actual_weight, actual_bias = dense_weight_bias(actual_paths[recipe.path])
        default = (2e-12, 2e-12) if expected_weight.dtype == torch.float64 else (3e-5, 3e-6)
        relative, absolute = default[0] if rtol is None else rtol, default[1] if atol is None else atol
        torch.testing.assert_close(actual_weight, expected_weight, rtol=relative, atol=absolute)
        if (actual_bias is None) != (expected_bias is None):
            raise AssertionError("Bias presence differs")
        if actual_bias is not None:
            torch.testing.assert_close(actual_bias, expected_bias, rtol=0, atol=0)
        probe = torch.arange(3*expected_weight.shape[1], dtype=expected_weight.dtype).reshape(3, -1)/expected_weight.shape[1]
        torch.testing.assert_close(actual_paths[recipe.path](probe), expected_paths[recipe.path](probe), rtol=relative, atol=absolute)
