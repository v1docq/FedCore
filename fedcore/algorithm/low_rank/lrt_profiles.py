"""BuildArchitecture profile for LRT and its dense from-scratch control."""
from dataclasses import asdict

import torch

from .lrt_specs import (LRT, LRTBuildPlan, lrt_parameter_counts,
                        lrt_metadata_values)
from .lrt_layers import LRTProfileError, LRTTransformer, _runtime


PAPER_URL = "https://arxiv.org/html/1910.13923v3"
AUTHOR_COMMIT = "2f78c6bd15b4026c8ddecd428800fcaa4331f6ad"
AUTHOR_ATTENTION_URL = ("https://github.com/gentaiscool/end2end-asr-pytorch/blob/"
                        + AUTHOR_COMMIT + "/models/common_layers.py#L202-L225")


def plan_lrt(spec, *, dtype=torch.float32, factorized=True):
    """Validate full model storage and a build reserve before tensor allocation.

    ``tensor_bytes`` is exact persistent parameter+buffer storage, not an RSS
    estimate. The build reserve is a conservative tensor reservation including
    sinusoid temporaries; training activations, gradients and optimizer state
    depend on the caller's batch and training protocol and are reported apart.
    """
    if type(spec) is not LRT or type(factorized) is not bool:
        raise LRTProfileError("A validated LRT spec and explicit factorized mode are required")
    _, dtype = _runtime("cpu", dtype)
    counts = lrt_parameter_counts(spec)
    itemsize = 4 if dtype == torch.float32 else 8
    fixed_bytes = spec.max_seq_len*spec.d_model*itemsize+8*len(lrt_metadata_values(spec, factorized))
    parameter_count = counts["factorized" if factorized else "dense"]
    tensor_bytes = parameter_count*itemsize+fixed_bytes
    dense_bytes = counts["dense"]*itemsize+fixed_bytes
    # Never claim this is a hard bound on native allocator/process RSS.
    reserve = 4*tensor_bytes+32*spec.max_seq_len*spec.d_model
    if parameter_count > spec.max_parameters:
        raise LRTProfileError("Whole-model parameters exceed max_parameters")
    if tensor_bytes > spec.max_tensor_bytes:
        raise LRTProfileError("Whole-model tensor storage exceeds max_tensor_bytes")
    if reserve > spec.max_peak_bytes:
        raise LRTProfileError("Build tensor reserve exceeds max_peak_bytes")
    return LRTBuildPlan(spec, factorized, str(dtype).removeprefix("torch."),
                        parameter_count, tensor_bytes, counts["dense"], dense_bytes, reserve)


def lrt_costs(spec, *, factorized=True, dtype=torch.float32, batch_size=1, sequence_length=None):
    """Exact model storage and analytical forward multiply-accumulate counts.

    Linear and attention MACs exclude bias/norm/softmax elementwise arithmetic.
    No measured latency, RSS or quality result is inferred from these counts.
    Attention score size and hypothetical KV tensor dimensions stay unchanged
    as factor ranks vary. This implementation does not maintain a KV cache.
    """
    plan = plan_lrt(spec, dtype=dtype, factorized=factorized)
    length = spec.max_seq_len if sequence_length is None else sequence_length
    if (type(batch_size) is not int or batch_size < 1
            or type(length) is not int or length < 1 or length > spec.max_seq_len):
        raise LRTProfileError("Cost inputs require positive batch_size and bounded sequence_length")
    counts = lrt_parameter_counts(spec)
    projection_weights = counts[("factorized" if factorized else "dense")+"_projection_weights_per_layer"]
    linear_macs = batch_size*length*(spec.num_layers*projection_weights+spec.d_model*spec.vocab_size)
    attention_macs = batch_size*spec.num_layers*spec.num_heads*length*length*(spec.d_k+spec.d_v)
    return {"parameter_count": plan.parameter_count, "tensor_bytes": plan.tensor_bytes,
            "dense_parameter_count": plan.dense_parameter_count,
            "dense_tensor_bytes": plan.dense_tensor_bytes,
            "linear_macs": linear_macs, "attention_macs": attention_macs,
            "forward_macs": linear_macs+attention_macs,
            "attention_score_elements_per_layer": batch_size*spec.num_heads*length*length,
            "hypothetical_kv_elements_all_layers": batch_size*spec.num_layers*spec.num_heads*length*(spec.d_k+spec.d_v),
            "kv_cache_implemented": False, "batch_size": batch_size, "sequence_length": length,
            "counting_scope": "whole persistent model; forward MACs exclude elementwise operations"}


def _build(spec, factorized, *, device="cpu", dtype=torch.float32, source=None, calibration=None):
    if source is not None or calibration is not None:
        raise LRTProfileError("From-scratch LRT accepts neither source weights nor calibration")
    device, dtype = _runtime(device, dtype)
    plan = plan_lrt(spec, dtype=dtype, factorized=factorized)
    model = LRTTransformer(spec, factorized=factorized, device=device, dtype=dtype)
    evidence = {"method": "lrt", "goal": "build_architecture", "schema_version": 1,
        "profile": "causal_token_transformer_v1", "factorized": factorized,
        "initialized_from_scratch": True, "training_mode": "from_scratch", "training_performed": False,
        "source_weights_required": False, "svd_required": False, "calibration_required": False,
        "architecture": asdict(spec), "dtype": plan.dtype, "device": "cpu",
        "costs": lrt_costs(spec, factorized=factorized, dtype=dtype),
        "build_peak_bytes_reserve": plan.build_peak_bytes_reserve,
        "source": {"paper": PAPER_URL, "paper_version": "1910.13923v3",
            "anchors": ["#S3.SS1", "#S3.SS2", "#S3.SS3", "#S3.SS4"],
            "author_commit": AUTHOR_COMMIT, "author_attention": AUTHOR_ATTENTION_URL,
            "status": "FedCore bounded mechanism reconstruction; not paper ASR reproduction",
            "attention_formula": "softmax(Q K^T / sqrt(d_k)) V",
            "printed_equation_2_corrected": True,
            "author_master_lrt_implementation_found": False},
        "limits": ["CPU float32/float64; bounded token self-attention",
            "No ASR acoustic front end, cross-attention, CER or beam search",
            "Factor ranks do not reduce sequence attention matrix or KV dimensions",
            "No claimed accuracy or latency reproduction; training uses existing FedCore runner"]}
    # Avoid making architecture descriptions depend eagerly on compression
    # interpreter/solver imports. This is an existing result carrier only.
    from .method_execution import MethodTransformResult
    return MethodTransformResult(model, plan, evidence)


def build_lrt(spec, *, device="cpu", dtype=torch.float32, source=None, calibration=None):
    return _build(spec, True, device=device, dtype=dtype, source=source, calibration=calibration)


def build_dense_lrt(spec, *, device="cpu", dtype=torch.float32, source=None, calibration=None):
    """Matched architecture/training dimensions; independently dense weights."""
    return _build(spec, False, device=device, dtype=dtype, source=source, calibration=calibration)
