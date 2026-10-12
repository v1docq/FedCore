"""Bounded SliceGPT: exact orthogonal graph change, then sequential slicing.

Mathematical scope: row states x, unscaled RMS N(x), orthogonal Q_i.
N(x Q_i)=N(x) Q_i; norm scales are fused into every input consumer.
With input/output bases A,B, PyTorch input weights become W diag(s) A,
output weights become B.T W, output bias B.T b, residual A.T B.
Thus every internal nonlinear argument is unchanged at full width. At k<d,
Q_k Q_k.T is a projection and exact equivalence is no longer claimed.

Pinned primary sources: arXiv:2401.15024v2 (2024-02-09), sections 3.1/3.3/3.4;
Microsoft TransformerCompression 6b12cdee6ad51791d7c776b3a046bc408b9e77e9,
src/slicegpt/rotate.py. This CPU reference variant uses raw residual second
moments, no PCA damping, explicit original-width RMS denominator and a
sharing-preserving compact tied-head bridge. These choices are declared;
they are not an exact implementation of every author's architecture adapter.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import wraps
import hashlib
import json
import math
import time

import torch
from torch import nn

from .coupled_layers import ReferenceAttention, ReferenceMLP
from .slicegpt_layers import (SliceGPTBlock, SliceGPTModel, SliceGPTRMSNorm, _raw_module_paths,
                              MAX_PERSISTENT_ELEMENTS, MAX_RESIDUAL_BLOCKS)
from .slicegpt_specs import SliceGPT, SliceGPTPlan


class SliceGPTProfileError(ValueError):
    """Unsupported source graph, numerical domain or bounded resource request."""


def _isolated(function):
    @wraps(function)
    def run(*args, **kwargs):
        with torch.random.fork_rng(devices=[]):
            return function(*args, **kwargs)
    return run


def _hash_tensor(digest, value):
    value = value.detach().cpu().contiguous()
    digest.update(str((list(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _fingerprint(model):
    digest = hashlib.sha256(json.dumps(model.representation_config(), sort_keys=True, allow_nan=False).encode())
    for name, value in model.state_dict().items():
        digest.update(name.encode()); _hash_tensor(digest, value)
    return digest.hexdigest()


def _calibration(calibration, model, spec, *, required=True):
    if calibration is None:
        if required:
            raise SliceGPTProfileError("SliceGPT calibration token tensors are required")
        return (), None, 0, 0
    batches = (calibration,) if isinstance(calibration, torch.Tensor) else calibration
    if not isinstance(batches, (tuple, list)) or not batches:
        raise SliceGPTProfileError("SliceGPT calibration requires a Tensor or explicit finite list/tuple")
    samples = tokens = 0; digest = hashlib.sha256()
    for index, batch in enumerate(batches):
        # Call the known validator directly; no arbitrary source method is run.
        SliceGPTModel._tokens(model, batch)
        if batch.device.type != "cpu":
            raise SliceGPTProfileError("SliceGPT calibration is CPU only")
        samples += batch.shape[0]; tokens += batch.numel()
        if samples > spec.max_samples or tokens > spec.max_tokens:
            raise SliceGPTProfileError("ResourceLimitExceeded: SliceGPT calibration samples/tokens")
        digest.update(str(index).encode()); _hash_tensor(digest, batch)
    return tuple(batches), digest.hexdigest(), samples, tokens


def _alias_groups(module):
    groups = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        key = (parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
        groups.setdefault(key, []).append((name, parameter.storage_offset(), tuple(parameter.stride())))
    return sorted(tuple(names) for names in groups.values())


def _preflight(model, *, max_peak_bytes=None):
    if type(model) is not SliceGPTModel:
        raise SliceGPTProfileError("SliceGPT supports the exact dedicated SliceGPTModel architecture only")
    from torch.nn.modules import module as module_runtime
    if any(getattr(module_runtime, name, {}) for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
            "_global_backward_pre_hooks")):
        raise SliceGPTProfileError("SliceGPT rejects global hooks affecting its consumer graph")
    allowed = (SliceGPTModel, SliceGPTBlock, SliceGPTRMSNorm, ReferenceAttention, ReferenceMLP,
               nn.ModuleList, nn.Linear, nn.Embedding)
    methods = ("forward", "causal_forward", "_tokens", "apply_branch", "attention",
               "project_query_key", "features", "representation_config", "state_dict",
               "named_modules", "named_parameters", "validate_checkpoint_state")
    try:
        actual_modules = _raw_module_paths(model)
    except ValueError as exc:
        raise SliceGPTProfileError(str(exc)) from exc
    for module in actual_modules.values():
        if type(module) not in allowed or any(name in vars(module) for name in methods):
            raise SliceGPTProfileError("SliceGPT rejects custom modules or instance method overrides")
        if any(getattr(module, name, {}) for name in (
                "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
                "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                "_load_state_dict_post_hooks")):
            raise SliceGPTProfileError("SliceGPT rejects source hooks and hidden consumers")
    if model.fused_norms or model.widths != (model.d_model,) * (2 * model.num_layers + 1):
        raise SliceGPTProfileError("SliceGPT source must be the original unfused full-width graph")
    parameters = tuple(nn.Module.parameters(model))
    if not parameters or parameters[0].dtype not in (torch.float32, torch.float64):
        raise SliceGPTProfileError("SliceGPT requires CPU float32/float64 parameters")
    dtype = parameters[0].dtype
    for value in tuple(parameters) + tuple(nn.Module.buffers(model)):
        if (value.device.type != "cpu" or (value.is_floating_point() and
                (value.dtype != dtype or not bool(torch.isfinite(value).all())))):
            raise SliceGPTProfileError("SliceGPT requires finite matching CPU float32/float64 tensors")
    if max_peak_bytes is not None:
        minimum_copy_bytes = 2 * sum(v.numel() * v.element_size() for v in parameters)
        if minimum_copy_bytes > max_peak_bytes:
            raise SliceGPTProfileError("ResourceLimitExceeded: SliceGPT source/skeleton reservation")
    # Construct a known skeleton under isolated RNG; do not copy or execute the source.
    with torch.random.fork_rng(devices=[]):
        expected = SliceGPTModel(model.vocab_size, model.d_model, model.num_layers, model.num_heads,
            model.num_kv_heads, model.head_dim, model.intermediate_size,
            tie_word_embeddings=model.tie_word_embeddings, bias=model.bias, rope=model.rope,
            eps=model.eps, dtype=dtype)
    expected_modules = dict(expected.named_modules())
    if set(actual_modules) != set(expected_modules):
        raise SliceGPTProfileError("SliceGPT source contains an undeclared consumer graph")
    for name, target in expected_modules.items():
        actual = actual_modules[name]
        if type(actual) is not type(target):
            raise SliceGPTProfileError("SliceGPT source module type mismatch")
        # Public data attributes encode head layout, activations, epsilon and dimensions.
        for key, value in vars(target).items():
            if not key.startswith("_") and key != "training":
                if key not in vars(actual) or vars(actual)[key] != value:
                    raise SliceGPTProfileError(f"SliceGPT source structural setting mismatch: {name}.{key}")
    source_state = model.state_dict(); expected_state = expected.state_dict()
    if set(source_state) != set(expected_state) or _alias_groups(model) != _alias_groups(expected):
        raise SliceGPTProfileError("SliceGPT source parameter paths or sharing do not match the declared graph")
    for key, value in source_state.items():
        if value.shape != expected_state[key].shape or value.dtype != expected_state[key].dtype:
            raise SliceGPTProfileError("SliceGPT source tensor shape/dtype mismatch")
    for name, value in model.named_buffers():
        if not torch.equal(value, dict(expected.named_buffers())[name]):
            raise SliceGPTProfileError("SliceGPT source fixed buffers must match the original graph")
    return dtype


def _costs(model, widths):
    p = model.vocab_size * widths[0]
    floating_buffers = widths[-1] * (widths[0] if model.tie_word_embeddings else widths[-1]) + len(widths)
    integer_buffers = 0
    for i, (ki, ko) in enumerate(zip(widths, widths[1:])):
        floating_buffers += ki * ko
        if i % 2 == 0:
            inner = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
            p += ki * inner + model.num_heads * model.head_dim * ko
            p += (inner + ko) if model.bias else 0
            integer_buffers += model.num_heads + model.num_kv_heads * model.head_dim
        else:
            p += 2 * model.intermediate_size * ki + model.intermediate_size * ko
            p += (2 * model.intermediate_size + ko) if model.bias else 0
    if not model.tie_word_embeddings:
        p += model.vocab_size * widths[-1]
    p += model.vocab_size if model.bias else 0
    return p, (p + floating_buffers) * model.embedding.weight.element_size() + integer_buffers * 8


@_isolated
def plan_slicegpt(model, spec, calibration=None):
    if type(spec) is not SliceGPT:
        raise SliceGPTProfileError("SliceGPT requires its explicit architecture spec")
    _preflight(model, max_peak_bytes=spec.max_peak_bytes)
    widths = spec.widths or (model.d_model,) * (2 * model.num_layers + 1)
    if len(widths) != 2 * model.num_layers + 1 or max(widths) > model.d_model:
        raise SliceGPTProfileError("SliceGPT widths must fit every declared residual boundary")
    batches, fingerprint, samples, tokens = _calibration(calibration, model, spec, required=False)
    parameters, tensor_bytes = _costs(model, widths)
    source_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    # Conservative admission reservation, not an observed peak: two full graphs,
    # output graph, eigensolver/bases, two sets of retained/canonical signals,
    # attention logits and verification logits. LAPACK/allocator overhead excluded.
    d = model.d_model; boundaries = len(widths); size = model.embedding.weight.element_size()
    max_b = max(2, max((v.shape[0] for v in batches), default=2))
    max_t = max(7, max((v.shape[1] for v in batches), default=7))
    inner = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    latent_activations = size * max_b * max_t * (
        4 * inner + 4 * model.num_heads * model.head_dim + 6 * model.intermediate_size + 6 * d)
    full_cost = _costs(model, (d,) * boundaries)[1]
    floor = (2 * source_bytes + full_cost + tensor_bytes + 8 * d * d * (boundaries + 12)
             + 8 * tokens * d * 4 + size * max_b * max_t * max_t * model.num_heads * 6
             + size * max_b * max_t * model.vocab_size * 4 + latent_activations)
    if floor > spec.max_peak_bytes:
        raise SliceGPTProfileError("ResourceLimitExceeded: SliceGPT graph/eigensolver/attention reservation")
    names = ("embedding",) + tuple(f"blocks.{i}.output" for i in range(2 * model.num_layers))
    return SliceGPTPlan("dedicated_causal_rms_transformer_v1", d, widths, names,
        _fingerprint(model), fingerprint, samples, tokens, parameters, tensor_bytes,
        floor, spec.max_peak_bytes, tied_head_policy=(
            "shared_compact_embedding_with_explicit_head_bridge" if model.tie_word_embeddings
            else "materialized_compact_untied_head"))


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise SliceGPTProfileError(f"SliceGPT finite numerical domain exceeded: {label}")
    return value


def residual_second_moment_basis(states):
    """Uncentered float64 Gram eigensystem, largest energy first, fixed signs."""
    if not isinstance(states, (list, tuple)) or not states:
        raise SliceGPTProfileError("Residual PCA requires a finite nonempty sequence of states")
    width = states[0].shape[-1]; gram = torch.zeros(width, width, dtype=torch.float64)
    count = 0
    for value in states:
        if (not isinstance(value, torch.Tensor) or value.ndim != 3 or not value.numel()
                or value.shape[-1] != width or value.device.type != "cpu"
                or value.dtype not in (torch.float32, torch.float64)):
            raise SliceGPTProfileError("Residual PCA requires compatible CPU [batch,tokens,width] states")
        rows = _finite(value.detach().double().reshape(-1, width), "residual rows")
        gram.add_(rows.T @ rows); count += rows.shape[0]
    eigenvalues, vectors = torch.linalg.eigh(_finite(gram, "residual second moment"))
    eigenvalues = eigenvalues.flip(0).clamp_min(0); vectors = vectors.flip(1)
    # Sign choice is immaterial mathematically, but stabilizes serialization.
    indices = vectors.abs().argmax(0)
    signs = torch.sign(vectors[indices, torch.arange(width)]); signs[signs == 0] = 1
    vectors = vectors * signs
    return vectors, eigenvalues, count


def _constructor(model, widths, *, fused=True):
    return SliceGPTModel(model.vocab_size, model.d_model, model.num_layers, model.num_heads,
        model.num_kv_heads, model.head_dim, model.intermediate_size, widths=widths,
        tie_word_embeddings=model.tie_word_embeddings, bias=model.bias, rope=model.rope,
        eps=model.eps, fused_norms=fused, dtype=model.embedding.weight.dtype).eval()


def _inputs(consumer):
    return ((consumer.q_proj, consumer.k_proj, consumer.v_proj) if type(consumer) is ReferenceAttention
            else (consumer.up_proj, consumer.gate_proj))


def _output(consumer):
    return consumer.o_proj if type(consumer) is ReferenceAttention else consumer.down_proj


@torch.no_grad()
def _transform_block(source, target, a, b):
    dtype = target.transition.dtype
    scale = source.norm.weight.detach().double()
    for old, new in zip(_inputs(source.consumer), _inputs(target.consumer)):
        new.weight.copy_(_finite((old.weight.detach().double() * scale) @ a, "fused input weight").to(dtype))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    old, new = _output(source.consumer), _output(target.consumer)
    new.weight.copy_(_finite(b.T @ old.weight.detach().double(), "output weight").to(dtype))
    if old.bias is not None:
        new.bias.copy_((b.T @ old.bias.detach().double()).to(dtype))
    target.transition.copy_(_finite(a.T @ b, "residual transition").to(dtype))


@torch.no_grad()
def _head(source, target, a, b):
    dtype = source.embedding.weight.dtype; scale = source.final_norm.weight.detach().double()
    if source.tie_word_embeddings:
        target.head_bridge.copy_(_finite(b.T @ (scale[:, None] * a), "tied head bridge").to(dtype))
    else:
        target.lm_head.weight.copy_(_finite((source.lm_head.weight.detach().double() * scale) @ b,
                                           "head weight").to(dtype))
    if source.lm_head.bias is not None:
        target.lm_head.bias.copy_(source.lm_head.bias)


def _bases(bases, width, count):
    if not isinstance(bases, (tuple, list)) or len(bases) != count:
        raise SliceGPTProfileError("Full rotation requires one complete orthogonal basis per boundary")
    result = []
    for basis in bases:
        if (not isinstance(basis, torch.Tensor) or basis.shape != (width, width)
                or basis.device.type != "cpu" or basis.dtype not in (torch.float32, torch.float64)):
            raise SliceGPTProfileError("SliceGPT bases must be full square CPU float matrices")
        value = _finite(basis.detach().double(), "basis")
        tolerance = 2e-6 if basis.dtype == torch.float32 else 1e-10
        if not torch.allclose(value.T @ value, torch.eye(width, dtype=torch.float64), atol=tolerance, rtol=tolerance):
            raise SliceGPTProfileError("SliceGPT requires independently orthogonal full-width bases")
        result.append(value)
    return tuple(result)


@torch.no_grad()
def _rotate(source, bases):
    result = _constructor(source, (source.d_model,) * len(bases))
    result.embedding.weight.copy_((source.embedding.weight.detach().double() @ bases[0]).to(source.embedding.weight.dtype))
    for i, block in enumerate(result.blocks):
        _transform_block(source.blocks[i], block, bases[i], bases[i + 1])
    _head(source, result, bases[0], bases[-1])
    return result


def _probes(model, verification_inputs):
    # Always include several independent lengths even for one calibration input.
    probes = [torch.arange(2 * length, dtype=torch.long).reshape(2, length) % model.vocab_size
              for length in (1, 3, 7)]
    if verification_inputs is not None:
        supplied = (verification_inputs,) if isinstance(verification_inputs, torch.Tensor) else verification_inputs
        if not isinstance(supplied, (list, tuple)):
            raise SliceGPTProfileError("Verification inputs require explicit token tensors")
        for value in supplied:
            SliceGPTModel._tokens(model, value)
            probes.append(value)
    return tuple(probes)


@torch.no_grad()
def verify_full_width(source, rotated, verification_inputs=None, *, rtol=None, atol=None):
    """End-to-end logits oracle, also usable to detect a wrong residual transition."""
    if source.widths != rotated.widths or any(v != source.d_model for v in rotated.widths):
        raise SliceGPTProfileError("Exact rotation verification requires full physical widths")
    maximum_tolerance = 5e-5 if source.embedding.weight.dtype == torch.float32 else 2e-10
    rtol = maximum_tolerance if rtol is None else rtol
    atol = maximum_tolerance if atol is None else atol
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum_tolerance
           for v in (rtol, atol)):
        raise SliceGPTProfileError("Full-width verification tolerances may only tighten the dtype defaults")
    cases = []; largest = 0.0
    for tokens in _probes(source, verification_inputs):
        expected, actual = source(tokens), rotated(tokens)
        _finite(expected, "source verification logits"); _finite(actual, "rotated verification logits")
        error = float((actual - expected).abs().max()); largest = max(largest, error)
        if not torch.allclose(expected, actual, rtol=rtol, atol=atol):
            raise SliceGPTProfileError("FullWidthEquivalenceFailed: SliceGPT must pass before slicing")
        cases.append({"batch": tokens.shape[0], "tokens": tokens.shape[1], "max_abs_error": error})
    return {"passed": True, "max_abs_logit_error": largest, "rtol": float(rtol), "atol": float(atol), "cases": cases}


@_isolated
@torch.no_grad()
def rotate_slicegpt(model, bases, *, verification_inputs=None):
    _preflight(model)
    values = _bases(bases, model.d_model, len(model.widths))
    result = _rotate(model, values)
    verify_full_width(model, result, verification_inputs)
    return result


def _actual_cost(model):
    seen = set(); tensor_bytes = 0
    for value in tuple(model.parameters()) + tuple(model.buffers()):
        storage = value.untyped_storage(); key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key); tensor_bytes += storage.nbytes()
    return {"unique_parameters": sum(v.numel() for v in model.parameters()), "tensor_bytes": tensor_bytes,
            "residual_transition_elements": sum(b.transition.numel() for b in model.blocks),
            "head_bridge_elements": model.head_bridge.numel()}


@torch.no_grad()
def _initial_rotation_gate(model, batches, spec, probes):
    states = [model.embedding(tokens).detach() for tokens in batches]
    bases = [residual_second_moment_basis(states)[0]]
    for block in model.blocks:
        states = [block(x) for x in states]
        bases.append(residual_second_moment_basis(states)[0])
    full = _rotate(model, _bases(bases, model.d_model, len(model.widths)))
    return verify_full_width(model, full, probes, rtol=spec.verification_rtol,
                             atol=spec.verification_atol)


@_isolated
@torch.no_grad()
def transform_slicegpt(model, calibration, spec, *, verification_inputs=None):
    started = time.perf_counter()
    plan = plan_slicegpt(model, spec, calibration)
    batches, calibration_hash, _, _ = _calibration(calibration, model, spec)
    # Bounds apply equally to extra verification data.
    if verification_inputs is not None:
        _calibration(verification_inputs, model, spec)
    probes = batches if verification_inputs is None else verification_inputs
    initial_verification = _initial_rotation_gate(model, batches, spec, probes)
    dtype = model.embedding.weight.dtype; d = model.d_model
    canonical = [model.embedding(tokens).detach() for tokens in batches]
    q, eigenvalues, count = residual_second_moment_basis(canonical)
    bases = [q]; statistics = []
    target = _constructor(model, plan.widths)
    a = q[:, :plan.widths[0]]
    target.embedding.weight.copy_((model.embedding.weight.detach().double() @ a).to(dtype))
    states = [(x.double() @ a).to(dtype) for x in canonical]
    statistics.append(_moment_record(plan.boundaries[0], canonical, eigenvalues, plan.widths[0], count))
    del canonical
    for i, source_block in enumerate(model.blocks):
        # Execute the current retained upstream signal, lifted to original
        # coordinates, then the ORIGINAL branch. This is an independent
        # canonical interpreter of the compact graph and observes its current
        # residual outputs before selecting the next output subspace.
        canonical = []
        for z in states:
            x = (z.double() @ a.T).to(dtype)
            branch, _ = source_block.apply_branch(x)
            canonical.append(_finite(x + branch, "current residual output"))
        q, eigenvalues, count = residual_second_moment_basis(canonical)
        bases.append(q); b = q[:, :plan.widths[i + 1]]
        _transform_block(source_block, target.blocks[i], a, b)
        next_states = [target.blocks[i](z) for z in states]
        for observed, reference in zip(next_states, canonical):
            # This checks the sequential calibration interpreter against the
            # installed compact graph, without claiming equivalence to source.
            if not torch.allclose(observed.double(), reference.double() @ b,
                                  atol=2e-5 if dtype == torch.float32 else 2e-10,
                                  rtol=2e-5 if dtype == torch.float32 else 2e-10):
                raise SliceGPTProfileError("CompactSequentialInterpreterMismatch")
        statistics.append(_moment_record(plan.boundaries[i + 1], canonical, eigenvalues,
                                         plan.widths[i + 1], count))
        states = next_states; a = b
    # Full-width transformation is reconstructed separately with the SAME
    # complete bases and checked against the source before returning slicing.
    full = _rotate(model, _bases(bases, d, len(plan.widths)))
    verification = verify_full_width(model, full, probes,
                                    rtol=spec.verification_rtol, atol=spec.verification_atol)
    _head(model, target, bases[0][:, :plan.widths[0]], a)
    approximation = _approximation_record(model, target, batches)
    if _fingerprint(model) != plan.source_fingerprint:
        raise SliceGPTProfileError("SliceGPT source changed during private transformation")
    costs = _actual_cost(target)
    if (costs["unique_parameters"] != plan.estimated_unique_parameters
            or costs["tensor_bytes"] != plan.estimated_tensor_bytes):
        raise SliceGPTProfileError("SliceGPT physical representation cost mismatch")
    evidence = {"method": "SliceGPT", "variant": spec.variant, "architecture": plan.architecture,
        "sources": {"paper": "https://arxiv.org/html/2401.15024v2", "paper_version": "2401.15024v2",
            "author_revision": "6b12cdee6ad51791d7c776b3a046bc408b9e77e9",
            "author_code": "https://github.com/microsoft/TransformerCompression/blob/6b12cdee6ad51791d7c776b3a046bc408b9e77e9/src/slicegpt/rotate.py"},
        "spec": asdict(spec), "plan": plan.to_dict(), "full_width_equivalence": verification,
        "pre_slicing_equivalence_gate": initial_verification,
        "approximation": approximation, "statistics": statistics,
        "provenance": {"source_fingerprint": plan.source_fingerprint,
            "calibration_fingerprint": calibration_hash, "calibration_role": "calibration",
            "calibration_order": "explicit_input_batch_order", "samples": plan.calibration_samples,
            "tokens": plan.calibration_tokens, "statistics_dtype": "float64",
            "padding": "all_supplied_tokens_valid",
            "statistics_kind": "uncentered_current_residual_second_moment",
            "upstream_state": "previous_physical_width_slices_already_installed"},
        "resources": {"estimated_floor_bytes": plan.estimated_floor_bytes,
            "max_peak_bytes": spec.max_peak_bytes, "observed_peak_bytes": None,
            "reservation_excludes": "allocator_and_LAPACK_internal_overhead", "device": "cpu",
            "persistent_element_cap": MAX_PERSISTENT_ELEMENTS,
            "residual_subblock_cap": MAX_RESIDUAL_BLOCKS,
            "elapsed_seconds": time.perf_counter() - started},
        "cost": costs, "source_cost": _actual_cost(model),
        "kv_layout": {"groups": model.num_kv_heads, "qk_width": model.head_dim,
            "value_width": model.head_dim, "bytes_per_batch_token":
                model.num_layers * 2 * model.num_kv_heads * model.head_dim * model.embedding.weight.element_size(),
            "cache_width_reduced": False},
        "declared_variant_choices": ["raw_residual_moments", "no_pca_damping",
            "original_width_rms_denominator", "shared_compact_tied_head_bridge"],
        "algorithm": {"choice": "sequential_dense_cpu_gram_eigendecomposition",
            "alternatives_rejected": ["centered_PCA_changes_the_linear_subspace_objective",
                "original_source_moments_ignore_prior_width_slices"],
            "statistics_work_per_boundary": "O(calibration_tokens*original_width^2)",
            "eigensolve_work_per_boundary": "O(original_width^3)",
            "basis_and_gram_storage": "O(boundaries*original_width^2)",
            "retained_signal_storage": "O(calibration_tokens*original_width)",
            "norm_scale_fusion": "all_QKV_gate_up_input_projections_and_final_head",
            "residual_transition": "Q_previous_retained.T@Q_next_retained",
            "author_implementation_executed": False},
        "environment": {"torch_version": str(torch.__version__), "backend": "torch_cpu_linalg_eigh",
            "execution_dtype": str(dtype), "statistics_dtype": "torch.float64"},
        "unsupported": ["arbitrary_HuggingFace_graphs", "LayerNorm_to_RMSNorm_conversion",
            "parallel_attention_mlp_graphs", "KV_width_pruning", "GPU", "recovery_training",
            "masked_calibration_batches"],
        "claims": {"exact": "full_width_architecture_equivalence_under_declared_RMS_graph",
            "empirical": "finite_calibration_logit_error_only", "quality_guarantee": False}}
    json.dumps(evidence, allow_nan=False)
    # Lazy import keeps the architecture contract independent from per-Linear
    # dispatch and uses the project's familiar immutable result envelope.
    from .method_execution import MethodTransformResult
    return MethodTransformResult(target, plan, evidence)


def _moment_record(boundary, states, eigenvalues, width, count):
    digest = hashlib.sha256()
    for value in states:
        _hash_tensor(digest, value)
    return {"boundary": boundary, "observations": count, "input_width": len(eigenvalues),
            "retained_width": width, "eigenvalues": eigenvalues.tolist(),
            "discarded_second_moment_energy": float(eigenvalues[width:].sum()),
            "state_fingerprint": digest.hexdigest(), "centered": False}


@torch.no_grad()
def _approximation_record(source, target, batches):
    squared = reference = 0.0; largest = 0.0; count = 0
    for tokens in batches:
        expected, actual = source(tokens).double(), target(tokens).double()
        error = _finite(actual - expected, "sliced calibration logits")
        squared += float(error.square().sum()); reference += float(expected.square().sum())
        largest = max(largest, float(error.abs().max())); count += error.numel()
    return {"is_width_reduced": source.widths != target.widths,
            "calibration_logit_mse": squared / count, "calibration_max_abs_logit_error": largest,
            "calibration_relative_squared_logit_error": squared / reference if reference else None,
            "tied_head_extra_projection": source.tie_word_embeddings and target.widths[0] < source.d_model,
            "end_to_end_error_bound_claimed": False}
