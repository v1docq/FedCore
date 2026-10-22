"""Explicit construction/transfer boundary for bounded P5 architectures.

These descriptions are deliberately separate from per-operator compression:
LRT has no pretrained weight, LightFormer transfers complete shared groups,
and SliceGPT transforms a closed residual graph.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from .slicegpt_specs import SliceGPT, SliceGPTPlan
    from .lightformer_specs import LightFormer, LightFormerPlan
    from .lrt_specs import LRT, LRTBuildPlan


class ArchitectureProfileError(ValueError):
    """An explicit architecture capability or output budget was violated."""


@dataclass(frozen=True)
class ArchitectureResult:
    model: nn.Module
    plan: SliceGPTPlan | LightFormerPlan | LRTBuildPlan
    evidence: dict


def _types():
    from .slicegpt_specs import SliceGPT
    from .lightformer_specs import LightFormer
    from .lrt_specs import LRT
    return {"slicegpt": SliceGPT, "lightformer": LightFormer, "lrt": LRT}


def architecture_name(spec):
    for name, cls in _types().items():
        if type(spec) is cls:
            return name
    raise ArchitectureProfileError("A validated explicit P5 architecture description is required")


def architecture_payload(spec):
    return {"architecture": architecture_name(spec), "version": 1, "options": asdict(spec)}


def parse_architecture(name, options=None, *, version=1):
    types = _types()
    if type(version) is not int or version != 1 or not isinstance(name, str) or name not in types:
        raise ArchitectureProfileError("Unsupported architecture or version")
    cls = types[name]
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - {field.name for field in fields(cls)}:
        raise ArchitectureProfileError("Unknown architecture option or invalid options object")
    options = dict(options)
    for key in ("widths", "encoder_groups", "decoder_groups", "encoder_sources", "decoder_sources"):
        if key in options:
            if not isinstance(options[key], (list, tuple)):
                raise ArchitectureProfileError(f"{key} requires a sequence")
            options[key] = tuple(options[key])
    try:
        return cls(**options)
    except TypeError as error:
        raise ArchitectureProfileError("Missing or invalid architecture options") from error


def _arguments(spec, source, calibration, dtype, device, verification_inputs):
    name = architecture_name(spec)
    if str(device) != "cpu":
        raise ArchitectureProfileError("P5 architecture profiles require CPU")
    if name == "lrt":
        if any(value is not None for value in (source, calibration, verification_inputs)):
            raise ArchitectureProfileError("LRT is built from scratch without pretrained source or calibration")
    else:
        if source is None:
            raise ArchitectureProfileError("SliceGPT and LightFormer require their explicit source graph")
        if dtype is not None:
            raise ArchitectureProfileError("Source architecture profiles inherit their source dtype")
        if name == "lightformer" and (calibration is not None or verification_inputs is not None):
            raise ArchitectureProfileError("LightFormer transfer does not consume calibration or rotation inputs")
    return name


def plan_architecture(spec, *, source=None, calibration=None, dtype=None, device="cpu"):
    """Read metadata/capabilities; never construct the target or train it."""
    name = _arguments(spec, source, calibration, dtype, device, None)
    if name == "lrt":
        import torch
        from .lrt_profiles import plan_lrt
        return plan_lrt(spec, dtype=torch.float32 if dtype is None else dtype)
    if name == "lightformer":
        from .lightformer_profiles import plan_lightformer
        return plan_lightformer(source, spec)
    from .slicegpt_profiles import plan_slicegpt
    return plan_slicegpt(source, spec, calibration)


def _budget(value, name):
    if value is not None and (type(value) is not int or value < 0):
        raise ArchitectureProfileError(f"{name} must be a nonnegative integer")


def _check_budget(parameters, tensor_bytes, parameter_budget, tensor_byte_budget):
    if parameter_budget is not None and parameters is not None and parameters > parameter_budget:
        raise ArchitectureProfileError("BudgetInfeasible: actual architecture parameters")
    if tensor_byte_budget is not None and tensor_bytes is not None and tensor_bytes > tensor_byte_budget:
        raise ArchitectureProfileError("BudgetInfeasible: actual architecture tensor bytes")


def build_architecture(spec, *, source=None, calibration=None, dtype=None, device="cpu",
                       verification_inputs=None, parameter_budget=None, tensor_byte_budget=None):
    """Build an independent graph and publish actual unique costs, without training."""
    _budget(parameter_budget, "parameter_budget")
    _budget(tensor_byte_budget, "tensor_byte_budget")
    name = _arguments(spec, source, calibration, dtype, device, verification_inputs)
    plan = plan_architecture(spec, source=source, calibration=calibration, dtype=dtype, device=device)
    _check_budget(getattr(plan, "parameter_count", None), getattr(plan, "tensor_bytes", None),
                  parameter_budget, tensor_byte_budget)
    from .method_execution import _rng_isolated
    return _rng_isolated(_execute)(spec, name, source, calibration, dtype, verification_inputs,
                                   parameter_budget, tensor_byte_budget)


def _execute(spec, name, source, calibration, dtype, verification_inputs,
             parameter_budget, tensor_byte_budget):
    import torch
    from .allocation import exact_unique_cost
    from .execution import _storage_records
    from .method_execution import _tensor_state_bytes
    if name == "lrt":
        from .lrt_profiles import build_lrt
        raw = build_lrt(spec, dtype=torch.float32 if dtype is None else dtype)
    elif name == "lightformer":
        from .lightformer_profiles import transfer_lightformer
        raw = transfer_lightformer(source, spec)
    else:
        from .slicegpt_profiles import transform_slicegpt
        raw = transform_slicegpt(source, calibration, spec, verification_inputs=verification_inputs)
    model = raw.model
    if model is source or any(value.is_floating_point() and not bool(torch.isfinite(value).all())
                              for value in model.state_dict().values()):
        raise ArchitectureProfileError("Architecture result must be finite and independent")
    if source is not None:
        source_storage = {(str(value.device), value.untyped_storage().data_ptr())
                          for value in source.state_dict().values() if value.numel()}
        if any((str(value.device), value.untyped_storage().data_ptr()) in source_storage
               for value in model.state_dict().values() if value.numel()):
            raise ArchitectureProfileError("Architecture result shares tensor storage with its source")
    storages = _storage_records(model)
    parameters = exact_unique_cost(storages)
    tensor_bytes = _tensor_state_bytes(model)
    _check_budget(parameters, tensor_bytes, parameter_budget, tensor_byte_budget)
    evidence = dict(raw.evidence, architecture_spec=architecture_payload(spec),
                    parameters_after=parameters, tensor_bytes_after=tensor_bytes,
                    parameter_bytes_after=exact_unique_cost(storages, "bytes"),
                    training_cost_separate=True, optimizer_rebuild_required=True)
    evidence = json.loads(json.dumps(evidence, allow_nan=False))
    model._fedcore_method_evidence = evidence
    model._fedcore_transform_plan = raw.plan
    model._fedcore_requires_optimizer_rebuild = True
    return ArchitectureResult(model, raw.plan, evidence)
