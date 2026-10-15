"""Plan, build and transfer a bounded LightFormer reference architecture.

Pinned primary source: ACL Anthology 2023.findings-acl.656, July 2023,
DOI 10.18653/v1/2023.findings-acl.656, sections 3.2/3.3 and appendix C.
No verified author implementation was linked by the paper/Anthology on review.
This is mechanism-level reference code, not a reproduction of paper quality:
encoder/decoder pre-norm layout, explicit decoder groups and fixed dense
embeddings/head are declared reference choices.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import torch
from torch import nn

from .lightformer_layers import (LightFormerTransformer, LightFormerAttention,
                                LightFormerEncoderBlock, LightFormerDecoderBlock,
                                _reference_bound)
from .lightformer_specs import LightFormer, LightFormerArchitecture, LightFormerPlan
from .method_errors import MethodSpecError
from .structured_layers import FactorizedLinear


def _fingerprint(model):
    digest = hashlib.sha256(json.dumps(model.representation_config(), sort_keys=True).encode())
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_contract(source):
    if type(source) is not LightFormerTransformer or source.rank is not None:
        raise MethodSpecError("LightFormer transfer supports only the exact dense LightFormerTransformer reference graph; HF/ASR/arbitrary graphs are unsupported")
    for stack, block_type in (("encoder", LightFormerEncoderBlock), ("decoder", LightFormerDecoderBlock)):
        layers = getattr(source, stack + "_layers")
        if getattr(source, stack + "_groups") != tuple(range(len(layers))):
            raise MethodSpecError("LightFormer source layers must be independent dense blocks")
        if len({id(block) for block in layers}) != len(layers):
            raise MethodSpecError("LightFormer source block aliases are unsupported")
        for block in layers:
            names = {"self_attention", "ffn_up", "ffn_down", "self_norm", "ffn_norm"}
            if stack == "decoder":
                names |= {"cross_attention", "cross_norm"}
            if type(block) is not block_type or set(block._modules) != names:
                raise MethodSpecError("LightFormer source block must have the exact supported topology")
            for name, child in block.named_children():
                if name.endswith("attention"):
                    if (type(child) is not LightFormerAttention or child.rank is not None
                            or child.d_model != source.d_model or child.num_heads != source.num_heads
                            or set(child._modules) != {"q_proj", "k_proj", "v_proj", "o_proj"}):
                        raise MethodSpecError("Unsupported LightFormer source attention")
                    if any(type(proj) is not nn.Linear or proj.weight.shape != (source.d_model, source.d_model)
                           or (proj.bias is not None) != source.bias for proj in child.children()):
                        raise MethodSpecError("Source attention requires exact dense projection shapes/biases")
                elif name.endswith("norm"):
                    if (type(child) is not nn.LayerNorm or child.normalized_shape != (source.d_model,)
                            or child.eps != source.norm_eps or not child.elementwise_affine
                            or child.bias is None):
                        raise MethodSpecError("Unsupported LightFormer source normalization")
                else:
                    shape = ((source.dim_feedforward, source.d_model) if name == "ffn_up"
                             else (source.d_model, source.dim_feedforward))
                    if type(child) is not nn.Linear or child.weight.shape != shape or (child.bias is not None) != source.bias:
                        raise MethodSpecError("Source FFN requires exact dense projection shapes/biases")
    expected_root = {"encoder_layers", "decoder_layers", "src_embedding", "tgt_embedding",
                     "encoder_norm", "decoder_norm", "output_projection"}
    if set(source._modules) != expected_root or any(type(getattr(source, n)) is not nn.Embedding
                                                  for n in ("src_embedding", "tgt_embedding")):
        raise MethodSpecError("Unsupported LightFormer fixed components")
    if (source.src_embedding.weight.shape != (source.src_vocab_size, source.d_model)
            or source.tgt_embedding.weight.shape != (source.tgt_vocab_size, source.d_model)
            or type(source.output_projection) is not nn.Linear
            or source.output_projection.weight.shape != (source.tgt_vocab_size, source.d_model)
            or (source.output_projection.bias is not None) != source.bias):
        raise MethodSpecError("Unsupported LightFormer embedding/head layout")
    for name in ("encoder_norm", "decoder_norm"):
        norm = getattr(source, name)
        if (type(norm) is not nn.LayerNorm or norm.normalized_shape != (source.d_model,)
                or norm.eps != source.norm_eps or not norm.elementwise_affine or norm.bias is None):
            raise MethodSpecError("Unsupported LightFormer final normalization")
    for module in source.modules():
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or "forward" in module.__dict__):
            raise MethodSpecError("LightFormer source hooks or custom forward overrides are unsupported")
    parameters = list(source.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in parameters}) != len(parameters):
        raise MethodSpecError("LightFormer source parameter ties are unsupported")
    if len({(p.device, p.untyped_storage().data_ptr()) for _, p in parameters}) != len(parameters):
        raise MethodSpecError("LightFormer source parameter storage ties are unsupported")
    first = parameters[0][1]
    if first.device.type != "cpu" or first.dtype not in (torch.float32, torch.float64):
        raise MethodSpecError("LightFormer reference transfer supports CPU float32/float64 only")
    for _, value in (*parameters, *source.named_buffers()):
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != first.dtype) or not bool(torch.isfinite(value).all()):
            raise MethodSpecError("LightFormer source tensors must be finite, uniform CPU float32/float64")
    source.validate_checkpoint_state(source.state_dict())
    return first


def _cost(architecture, spec):
    d, f, b = architecture.d_model, architecture.dim_feedforward, int(architecture.bias)
    enc, dec = len(spec.encoder_sources), len(spec.decoder_sources)
    fixed = (architecture.src_vocab_size + 2 * architecture.tgt_vocab_size) * d + b * architecture.tgt_vocab_size + 4 * d
    attention = 8 * d * spec.rank + b * 4 * d
    ffn = 2 * spec.rank * (d + f) + b * (d + f)
    total = fixed + enc * (attention + ffn + 4 * d) + dec * (2 * attention + ffn + 6 * d)
    return fixed, total


def _resources(architecture, spec, source_elements, source_enc_depth, source_dec_depth, dtype):
    size = 4 if dtype == "torch.float32" else 8
    d, f, r = architecture.d_model, architecture.dim_feedforward, spec.rank
    _, total = _cost(architecture, spec)
    metadata_bytes = 8 * (len(spec.encoder_groups) + len(spec.decoder_groups) + len(spec.encoder_sources) + len(spec.decoder_sources))
    tensor_bytes = total * size + architecture.max_positions * d * size + metadata_bytes
    svd_estimate = 8 * (14 * max(d * d, d * f) + 2 * max(d, f) * min(d, f) + min(d, f) ** 2 + r * (d + f))
    source_bytes = source_elements * size + architecture.max_positions * d * size + 16 * (source_enc_depth + source_dec_depth)
    return tensor_bytes, source_bytes + tensor_bytes + svd_estimate


def plan_lightformer(source, spec):
    """Validate source, topology, ranks and resource bounds before construction/SVD.

    The plan contains values only. Peak bytes are a conservative estimate for
    simultaneously live tensors and SVD intermediates; vendor workspace/RSS
    is not measured or guaranteed by this estimate.
    """
    if type(spec) is not LightFormer:
        raise MethodSpecError("An explicit LightFormer architecture spec is required")
    if type(source) is not LightFormerTransformer:
        raise MethodSpecError("LightFormer requires the exact supported dense reference architecture; arbitrary/HF/ASR graphs are unsupported")
    if (spec.rank > min(source.d_model, source.dim_feedforward)
            or 2 * spec.rank >= source.d_model
            or spec.rank * (source.d_model + source.dim_feedforward) >= source.d_model * source.dim_feedforward):
        raise MethodSpecError("LightFormer rank must produce strictly compact factors for every supported projection")
    if (max(spec.encoder_sources) >= len(source.encoder_layers)
            or max(spec.decoder_sources) >= len(source.decoder_layers)):
        raise MethodSpecError("LightFormer source layer index exceeds the supported source depth")
    architecture = LightFormerArchitecture(**{name: getattr(source, name) for name in
        ("src_vocab_size", "tgt_vocab_size", "d_model", "num_heads", "dim_feedforward", "max_positions", "bias", "norm_eps")})
    fixed, total = _cost(architecture, spec)
    try:
        _reference_bound(architecture.src_vocab_size, architecture.tgt_vocab_size, architecture.d_model,
                         architecture.dim_feedforward, spec.encoder_groups, spec.decoder_groups,
                         spec.rank, architecture.bias, architecture.max_positions)
    except ValueError as error:
        raise MethodSpecError(str(error)) from error
    if total > spec.max_parameters:
        raise MethodSpecError("LightFormer architecture exceeds max_parameters before allocation")
    first = _source_contract(source)
    source_elements = sum(p.numel() for p in source.parameters())
    # Double precision input/U/S/V and reconstruction/factor temporaries, plus
    # a ten-matrix workspace allowance. This is deliberately an estimate.
    tensor_bytes, estimated_peak = _resources(architecture, spec, source_elements,
        len(source.encoder_layers), len(source.decoder_layers), str(first.dtype))
    if estimated_peak > spec.max_peak_bytes:
        raise MethodSpecError("LightFormer tensor/SVD planning estimate exceeds max_peak_bytes before allocation")
    return LightFormerPlan(architecture, spec, _fingerprint(source), source_elements, total, fixed,
                           len(source.encoder_layers), len(source.decoder_layers), str(first.dtype),
                           tensor_bytes, estimated_peak)


def build_lightformer(plan, *, device=None, dtype=None):
    """Build compact architecture independently of source weight transfer."""
    if type(plan) is not LightFormerPlan:
        raise MethodSpecError("An immutable LightFormerPlan is required")
    if (type(plan.architecture) is not LightFormerArchitecture or type(plan.spec) is not LightFormer
            or plan.dtype not in ("torch.float32", "torch.float64")):
        raise MethodSpecError("LightFormer plan architecture/spec/dtype is invalid")
    if device is not None and torch.device(device).type != "cpu":
        raise MethodSpecError("LightFormer reference architecture is CPU only")
    dtype = getattr(torch, plan.dtype[6:]) if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64) or str(dtype) != plan.dtype:
        raise MethodSpecError("Architecture dtype must match the validated plan")
    fixed, total = _cost(plan.architecture, plan.spec)
    tensor_bytes, estimated_peak = _resources(plan.architecture, plan.spec, plan.source_parameter_elements,
        plan.source_encoder_depth, plan.source_decoder_depth, plan.dtype)
    if (total != plan.target_parameter_elements or fixed != plan.fixed_parameter_elements
            or tensor_bytes != plan.target_tensor_bytes or estimated_peak != plan.estimated_peak_bytes
            or total > plan.spec.max_parameters or estimated_peak > plan.spec.max_peak_bytes):
        raise MethodSpecError("LightFormer plan resource metadata is inconsistent")
    with torch.random.fork_rng(devices=[]):
        result = LightFormerTransformer(**asdict(plan.architecture), rank=plan.spec.rank,
            encoder_groups=plan.spec.encoder_groups, decoder_groups=plan.spec.decoder_groups,
            encoder_sources=plan.spec.encoder_sources, decoder_sources=plan.spec.decoder_sources,
            device=device, dtype=dtype)
    if sum(p.numel() for p in result.parameters()) != total:
        raise RuntimeError("LightFormer exact unique parameter accounting disagrees with construction")
    return result


def _transfer_linear(source, target, rank):
    # Paper convention U_r, Sigma_r V_r^T; no unrelated source layers averaged.
    u, s, vh = torch.linalg.svd(source.weight.detach().double(), full_matrices=False)
    left, right = u[:, :rank].to(target.left.weight), (s[:rank, None] * vh[:rank]).to(target.right.weight)
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise MethodSpecError("Nonfinite SVD factors cannot initialize LightFormer")
    residual = float(s[rank:].square().sum())
    if not 0 <= residual < float("inf"):
        raise MethodSpecError("SVD residual exceeds the JSON-safe finite numerical domain")
    target.left.weight.copy_(left)
    target.right.weight.copy_(right)
    if source.bias is not None:
        target.left.bias.copy_(source.bias)
    return residual


def transfer_lightformer(source, spec, *, plan=None):
    """Transfer selected source blocks into one actual object per target group.

    Source parameters, modes, hooks and RNG are preserved. Returned factors and
    fixed parameters share no storage with the source.
    """
    validated = plan_lightformer(source, spec)
    if plan is not None and plan != validated:
        raise MethodSpecError("LightFormer plan is stale or disagrees with source/spec")
    plan = validated
    target = build_lightformer(plan)
    residuals = []
    with torch.no_grad():
        for name in ("src_embedding", "tgt_embedding", "encoder_norm", "decoder_norm", "output_projection"):
            getattr(target, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
        target.positions.copy_(source.positions)
        for stack in ("encoder", "decoder"):
            source_layers, target_layers = getattr(source, stack + "_layers"), getattr(target, stack + "_layers")
            groups = getattr(spec, stack + "_groups")
            for group, source_index in enumerate(getattr(spec, stack + "_sources")):
                source_block, target_block = source_layers[source_index], target_layers[groups.index(group)]
                for name, child in source_block.named_children():
                    installed = getattr(target_block, name)
                    if type(child) is nn.LayerNorm:
                        installed.load_state_dict(child.state_dict(), strict=True)
                    elif type(child) is LightFormerAttention:
                        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                            error = _transfer_linear(getattr(child, projection), getattr(installed, projection), spec.rank)
                            residuals.append({"stack": stack, "group": group, "source_layer": source_index,
                                              "projection": name + "." + projection, "spectral_tail_squared": error})
                    else:
                        error = _transfer_linear(child, installed, spec.rank)
                        residuals.append({"stack": stack, "group": group, "source_layer": source_index,
                                          "projection": name, "spectral_tail_squared": error})
    target.train(source.training)
    target.validate_checkpoint_state(target.state_dict())
    parameter_bytes = sum(p.numel() * p.element_size() for p in target.parameters())
    tensor_bytes = parameter_bytes + sum(b.numel() * b.element_size() for b in target.buffers())
    if tensor_bytes != plan.target_tensor_bytes:
        raise RuntimeError("LightFormer tensor-byte plan disagrees with constructed topology")
    enc, dec = len(spec.encoder_groups), len(spec.decoder_groups)
    evidence = {"method": "LightFormer", "variant": spec.variant, "goal": "build_architecture",
        "paper": {"anthology_id": "2023.findings-acl.656", "doi": "10.18653/v1/2023.findings-acl.656",
                  "url": "https://aclanthology.org/2023.findings-acl.656.pdf", "version": "ACL July 2023",
                  "author_code": None, "author_code_status": "no_verified_author_repository_found"},
        "scope": "bounded_reference_mechanism_not_paper_quality_reproduction",
        "source_fingerprint": plan.source_fingerprint,
        "architecture": target.representation_config(), "source_encoder_depth": plan.source_encoder_depth,
        "source_decoder_depth": plan.source_decoder_depth,
        "encoder_applications": enc, "decoder_applications": dec,
        "self_attention_applications": enc + dec, "cross_attention_applications": dec,
        "ffn_applications": enc + dec, "factor_linear_applications": 2 * (6 * enc + 10 * dec),
        "output_projection_applications": 1, "embedding_applications": 2,
        "associated_norm_applications": 2 * enc + 3 * dec, "final_norm_applications": 2,
        "linear_operator_applications": 2 * (6 * enc + 10 * dec) + 1,
        "unique_encoder_groups": len(spec.encoder_sources), "unique_decoder_groups": len(spec.decoder_sources),
        "unique_parameters": plan.target_parameter_elements, "source_unique_parameters": plan.source_parameter_elements,
        "fixed_parameters": plan.fixed_parameter_elements, "parameter_tensor_bytes": parameter_bytes,
        "tensor_bytes": tensor_bytes, "estimated_peak_bytes": plan.estimated_peak_bytes,
        "peak_estimate_policy": "live_source_target_tensors_and_conservative_svd_workspace_estimate_not_measured_RSS",
        "training_resources": {"trainable_unique_parameters": plan.target_parameter_elements,
            "gradient_bytes_estimate": parameter_bytes, "adam_two_moment_bytes_estimate": 2 * parameter_bytes,
            "optimizer_step_counters_included": False, "activation_bytes": None, "kv_cache_bytes": None,
            "note": "parameter sharing does not reduce layer applications or alias application-specific activations/caches"},
        "sharing": "whole_block_object_attention_ffn_associated_norms",
        "kv_cache_policy": "unsupported_refuse_cache_requests", "svd_residuals": residuals}
    from .method_execution import MethodTransformResult
    return MethodTransformResult(target, plan, evidence)
