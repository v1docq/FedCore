"""Exact coupled FFN coordinates and the bounded SoLA reference profile.

All weights use [output, input] orientation. Selection never refits ``down``.
SoLA keeps every marginal neuron and changes only its linear representations.
The supplied activation point and normalized second moments are explicit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.allocation import (
    StorageCost, CandidateOption, CandidateTable, ScoreObjective,
    BudgetInfeasible, allocate, exact_unique_cost,
)
from fedcore.algorithm.low_rank.approximation import solve_weighted
from fedcore.algorithm.low_rank.coupled_layers import ReferenceMLP
from fedcore.algorithm.low_rank.plans import MetricPolicy
from fedcore.algorithm.low_rank.statistics import (
    StatisticsPassport, empty_second_moment, update_second_moment,
    finish_second_moment, validate_passport,
)
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair


MAX_FFN_ELEMENTS = 1_000_000


def _indices(values, width, name="channels"):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be an ordered integer sequence")
    result = tuple(values)
    if (any(type(i) is not int or not 0 <= i < width for i in result)
            or len(result) != len(set(result))):
        raise ValueError(f"{name} must contain distinct indices within the current width")
    return result


def _linear(n, m, bias, *, device=None, dtype=None):
    # Zero-width branches are genuine empty operators, including their biases.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Initializing zero-element tensors")
        return nn.Linear(n, m, bias=bias, device=device, dtype=dtype)


def _validate_tensors(module):
    parameters = tuple(module.parameters())
    if (not parameters or any(p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64)
                              or not bool(torch.isfinite(p).all()) for p in parameters)
            or len({p.dtype for p in parameters}) != 1):
        raise ValueError("Bounded FFN profile requires finite CPU float32/float64 parameters")
    if sum(p.numel() for p in parameters) > MAX_FFN_ELEMENTS:
        raise ValueError("FFN persistent-element limit exceeded before copying")
    for child in module.modules():
        if ("forward" in child.__dict__ or child._forward_hooks or child._forward_pre_hooks
                or child._backward_hooks or getattr(child, "_backward_pre_hooks", {})):
            raise ValueError("FFN adapters require unmodified operators without hooks")


@dataclass(frozen=True)
class FFNView:
    d_model: int
    width: int
    gated: bool
    activation: str
    original_channels: tuple[int, ...]
    up: nn.Linear
    gate: nn.Linear | None
    down: nn.Linear
    adapter: str


def inspect_mlp(module):
    """Allowlist actual classes; matching attribute names do not define an adapter."""
    if type(module) is ReferenceMLP or type(module) is SparseMLP:
        activation, gated = module.activation, module.gated
        up, gate, down = module.up_proj, module.gate_proj, module.down_proj
        channels = tuple(module.channel_indices)
        adapter = type(module).__name__
    else:
        try:
            from transformers.models.llama.modeling_llama import LlamaMLP
        except ImportError:
            LlamaMLP = None
        if LlamaMLP is None or type(module) is not LlamaMLP:
            raise ValueError("Supported FFN adapters are exact ReferenceMLP, SparseMLP and Transformers LlamaMLP")
        if getattr(module.config, "pretraining_tp", 1) != 1:
            raise ValueError("LlamaMLP adapter requires pretraining_tp=1")
        activation, gated = module.config.hidden_act, True
        up, gate, down = module.up_proj, module.gate_proj, module.down_proj
        channels = tuple(range(up.out_features))
        adapter = "transformers-4.44-LlamaMLP"
    _validate_tensors(module)
    if activation not in ("silu", "gelu", "relu"):
        raise ValueError("FFN adapter requires an explicitly supported activation")
    if any(type(layer) is not nn.Linear for layer in (up, down, *((gate,) if gated else ()))):
        raise ValueError("Source FFN projections must be unmodified dense Linear")
    for layer in (up, down, *((gate,) if gated else ())):
        if (layer._modules or layer._buffers or set(layer._parameters) != {"weight", "bias"}
                or layer.weight.shape != (layer.out_features, layer.in_features)
                or (layer.bias is not None and layer.bias.shape != (layer.out_features,))):
            raise ValueError("Source FFN projections cannot contain extra state or undeclared shapes")
    expected_children = {"up_proj", "down_proj"} | ({"gate_proj"} if gated else set())
    if adapter == "transformers-4.44-LlamaMLP":
        from transformers.activations import ACT2FN
        if (type(module.act_fn) is not type(ACT2FN[activation])
                or tuple(module.act_fn.parameters()) or tuple(module.act_fn.buffers())
                or getattr(module.act_fn, "inplace", False)):
            raise ValueError("LlamaMLP activation must match its explicit configuration")
        expected_children.add("act_fn")
    if {name for name, child in module._modules.items() if child is not None} != expected_children or module._parameters:
        raise ValueError("Source FFN cannot contain undeclared children or parameters")
    expected_buffers = {"_channel_map", "_current_map"} if type(module) is SparseMLP else set()
    if set(module._buffers) != expected_buffers:
        raise ValueError("Source FFN cannot contain undeclared buffers")
    d, h = up.in_features, up.out_features
    if (down.in_features != h or down.out_features != d or len(channels) != h
            or (gated and (gate.in_features != d or gate.out_features != h))):
        raise ValueError("Coupled FFN projection shapes disagree")
    if len(set(channels)) != h or any(type(i) is not int or i < 0 for i in channels):
        raise ValueError("FFN original coordinates must be distinct nonnegative integers")
    # General shared projection layouts need their own declared topology.
    weights = [layer.weight for layer in (up, down, *((gate,) if gated else ()))]
    if len({p.untyped_storage().data_ptr() for p in weights if p.numel()}) != sum(bool(p.numel()) for p in weights):
        raise ValueError("FFN source has unsupported shared projection storage")
    return FFNView(d, h, gated, activation, channels, up, gate, down, adapter)


def validate_tensor_state(module, state=None, prefix=""):
    if state is None:
        state = module.state_dict()
    if not isinstance(state, Mapping) or not isinstance(prefix, str):
        raise ValueError("Checkpoint requires a tensor mapping and string prefix")
    expected = module.state_dict()
    keys = {prefix + key for key in expected}
    if not keys.issubset(state) or (not prefix and keys != set(state)):
        raise ValueError("Checkpoint keys disagree with declared representation")
    for name, reference in expected.items():
        value = state[prefix + name]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != reference.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Checkpoint tensor mismatch: {prefix + name}")
        if reference.dtype in (torch.int32, torch.int64) and not torch.equal(value.cpu(), reference.cpu()):
            raise ValueError(f"Immutable coordinate map mismatch: {prefix + name}")
    aliases = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if id(parameter) in aliases:
            previous = aliases[id(parameter)]
            if not torch.equal(state[prefix + previous], state[prefix + name]):
                raise ValueError("Saved values disagree for shared parameter aliases")
        else:
            aliases[id(parameter)] = name


def _projection_config(layer):
    if type(layer) is nn.Linear:
        return {"layer_type": "Linear", "in_features": layer.in_features,
                "out_features": layer.out_features, "bias": layer.bias is not None}
    if type(layer) is FactorizedLinear:
        return layer.representation_config()
    raise ValueError("Unsupported FFN projection representation")


def _projection_from_config(config, *, device=None, dtype=None):
    if not isinstance(config, Mapping):
        raise ValueError("Projection config must be a mapping")
    if config.get("layer_type") == "Linear" and set(config) == {"layer_type", "in_features", "out_features", "bias"}:
        n, m, bias = config["in_features"], config["out_features"], config["bias"]
        if any(type(v) is not int or v < 0 for v in (n, m)) or type(bias) is not bool:
            raise ValueError("Invalid Linear projection config")
        return _linear(n, m, bias, device=device, dtype=dtype)
    if config.get("layer_type") == "FactorizedLinear" and set(config) == {"layer_type", "version", "in_features", "out_features", "rank", "bias"}:
        if type(config["version"]) is not int or config["version"] != 1 or type(config["bias"]) is not bool:
            raise ValueError("Invalid factor projection config")
        if any(type(config[k]) is not int or config[k] < 1 for k in ("in_features", "out_features", "rank")):
            raise ValueError("Invalid factor dimensions")
        if config["rank"] > min(config["in_features"], config["out_features"]):
            raise ValueError("Factor rank exceeds dimensions")
        return FactorizedLinear(config["in_features"], config["out_features"], config["rank"],
                                config["bias"], device=device, dtype=dtype)
    raise ValueError("Unknown FFN projection constructor config")


class SparseMLP(nn.Module):
    """Compact linked channels, including a valid empty side of a partition."""
    def __init__(self, d_model, intermediate_size, *, gated=True, activation="silu",
                 up_bias=True, gate_bias=True, down_bias=True, channel_indices=None,
                 current_indices=None, source_width=None, device=None, dtype=None):
        super().__init__()
        if (type(d_model) is not int or d_model < 1 or type(intermediate_size) is not int or intermediate_size < 0
                or any(type(v) is not bool for v in (gated, up_bias, gate_bias, down_bias))
                or activation not in ("silu", "gelu", "relu")):
            raise ValueError("Invalid compact FFN dimensions, activation or bias flags")
        if (device is not None and torch.device(device).type != "cpu") or dtype not in (None, torch.float32, torch.float64):
            raise ValueError("Compact FFN profile requires CPU float32/float64")
        n = intermediate_size
        source_width = n if source_width is None else source_width
        if type(source_width) is not int or source_width < n:
            raise ValueError("Source width must cover compact coordinates")
        original = tuple(range(n)) if channel_indices is None else tuple(channel_indices)
        current = tuple(range(n)) if current_indices is None else _indices(current_indices, source_width)
        if (len(original) != n or len(current) != n or len(set(original)) != n
                or any(type(i) is not int or i < 0 for i in original)):
            raise ValueError("Compact FFN requires exact original and current coordinate maps")
        elements = (3 if gated else 2) * n * d_model + n * (int(up_bias) + int(gated and gate_bias)) + int(down_bias)*d_model + 2*n
        if elements > MAX_FFN_ELEMENTS:
            raise ValueError("Compact FFN persistent-element limit exceeded before allocation")
        self.d_model, self.intermediate_size = d_model, n
        self.gated, self.activation = gated, activation
        self.channel_indices, self.current_indices, self.source_width = original, current, source_width
        self.up_proj = _linear(d_model, n, up_bias, device=device, dtype=dtype)
        self.gate_proj = _linear(d_model, n, gate_bias, device=device, dtype=dtype) if gated else None
        self.down_proj = _linear(n, d_model, down_bias, device=device, dtype=dtype)
        self.register_buffer("_channel_map", torch.tensor(original, dtype=torch.int64, device=device))
        self.register_buffer("_current_map", torch.tensor(current, dtype=torch.int64, device=device))

    def features(self, x):
        activation = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[self.activation]
        up = self.up_proj(x)
        return activation(self.gate_proj(x)) * up if self.gated else activation(up)

    def forward(self, x):
        return self.down_proj(self.features(x))

    def representation_config(self):
        return {"layer_type": "SparseMLP", "version": 1, "d_model": self.d_model,
                "intermediate_size": self.intermediate_size, "gated": self.gated,
                "activation": self.activation, "channel_indices": list(self.channel_indices),
                "current_indices": list(self.current_indices), "source_width": self.source_width,
                "up": _projection_config(self.up_proj), "gate": None if not self.gated else _projection_config(self.gate_proj),
                "down": _projection_config(self.down_proj)}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        required = {"layer_type", "version", "d_model", "intermediate_size", "gated", "activation",
                    "channel_indices", "current_indices", "source_width", "up", "gate", "down"}
        if (not isinstance(config, Mapping) or set(config) != required or config["layer_type"] != "SparseMLP"
                or type(config["version"]) is not int or config["version"] != 1):
            raise ValueError("Exact version-1 SparseMLP constructor config required")
        d, h = config["d_model"], config["intermediate_size"]
        # Validate every nested shape before allocating any projection.
        for name, shape in (("up", (d, h)), ("down", (h, d)), *(( ("gate", (d, h)),) if config["gated"] else ())):
            part = config[name]
            if not isinstance(part, Mapping) or (part.get("in_features"), part.get("out_features")) != shape:
                raise ValueError("FFN constructor projection dimensions disagree")
        if not config["gated"] and config["gate"] is not None:
            raise ValueError("Ordinary FFN cannot declare a gate projection")
        with torch.random.fork_rng(devices=[]):
            result = cls(d, h, gated=config["gated"], activation=config["activation"],
                         up_bias=config["up"]["bias"], gate_bias=False if config["gate"] is None else config["gate"]["bias"],
                         down_bias=config["down"]["bias"], channel_indices=config["channel_indices"],
                         current_indices=config["current_indices"], source_width=config["source_width"], device=device, dtype=dtype)
            for name in ("up", "gate", "down"):
                if config[name] is not None:
                    setattr(result, name + "_proj", _projection_from_config(config[name], device=device, dtype=dtype))
        return result

    def validate_state(self, state=None, prefix=""):
        validate_tensor_state(self, state, prefix)
        if self._channel_map.tolist() != list(self.channel_indices) or self._current_map.tolist() != list(self.current_indices):
            raise ValueError("Live FFN coordinate maps changed")

    validate_checkpoint_state = validate_state


def _copy_slice(view, indices, *, output_bias):
    reference = view.up.weight
    with torch.random.fork_rng(devices=[]):
        result = SparseMLP(view.d_model, len(indices), gated=view.gated, activation=view.activation,
                           up_bias=view.up.bias is not None, gate_bias=view.gate is not None and view.gate.bias is not None,
                           down_bias=output_bias and view.down.bias is not None,
                           channel_indices=[view.original_channels[i] for i in indices], current_indices=indices,
                           source_width=view.width, device=reference.device, dtype=reference.dtype)
    index = torch.tensor(indices, device=reference.device, dtype=torch.long)
    with torch.no_grad():
        for target, source in ((result.up_proj, view.up), *(( (result.gate_proj, view.gate),) if view.gated else ())):
            target.weight.copy_(source.weight.index_select(0, index))
            if source.bias is not None:
                target.bias.copy_(source.bias.index_select(0, index))
        result.down_proj.weight.copy_(view.down.weight.index_select(1, index))
        if result.down_proj.bias is not None:
            result.down_proj.bias.copy_(view.down.bias)
    return result


def slice_mlp(module, channels):
    view = inspect_mlp(module)
    indices = _indices(channels, view.width)
    result = _copy_slice(view, indices, output_bias=True)
    result.train(module.training)
    return result, {"method": "exact_channel_slice", "adapter": view.adapter,
                    "current_indices": indices, "original_indices": result.channel_indices,
                    "source_width": view.width, "compact_width": len(indices), "down_refit": False}


class PartitionedMLP(nn.Module):
    def __init__(self, prime, marginal, output_bias=None):
        super().__init__()
        if (type(prime) is not SparseMLP or type(marginal) is not SparseMLP
                or (prime.d_model, prime.gated, prime.activation, prime.source_width) !=
                   (marginal.d_model, marginal.gated, marginal.activation, marginal.source_width)
                or _projection_config(prime.down_proj)["bias"] or _projection_config(marginal.down_proj)["bias"]
                or set(prime.current_indices) & set(marginal.current_indices)
                or set(prime.channel_indices) & set(marginal.channel_indices)
                or set(prime.current_indices) | set(marginal.current_indices) != set(range(prime.source_width))):
            raise ValueError("Prime and marginal branches must be a complete disjoint coupled FFN partition")
        self.prime, self.marginal = prime, marginal
        self.d_model, self.gated, self.activation = prime.d_model, prime.gated, prime.activation
        reference = next(prime.parameters())
        if output_bias is not None and (not isinstance(output_bias, torch.Tensor) or output_bias.shape != (self.d_model,)
                                       or not bool(torch.isfinite(output_bias).all())):
            raise ValueError("Partition output bias requires a finite external-width vector")
        self.output_bias = None if output_bias is None else nn.Parameter(output_bias.detach().clone().to(reference))

    def forward(self, x):
        result = self.prime(x) + self.marginal(x)
        return result if self.output_bias is None else result + self.output_bias

    def representation_config(self):
        return {"layer_type": "PartitionedMLP", "version": 1, "prime": self.prime.representation_config(),
                "marginal": self.marginal.representation_config(), "output_bias": self.output_bias is not None}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        if (not isinstance(config, Mapping) or set(config) != {"layer_type", "version", "prime", "marginal", "output_bias"}
                or config["layer_type"] != "PartitionedMLP" or type(config["version"]) is not int
                or config["version"] != 1 or type(config["output_bias"]) is not bool):
            raise ValueError("Exact version-1 PartitionedMLP constructor config required")
        prime = SparseMLP.from_config(config["prime"], device=device, dtype=dtype)
        marginal = SparseMLP.from_config(config["marginal"], device=device, dtype=dtype)
        bias = next(prime.parameters()).new_zeros(prime.d_model) if config["output_bias"] else None
        return cls(prime, marginal, bias)

    def validate_state(self, state=None, prefix=""):
        validate_tensor_state(self, state, prefix)
        self.prime.validate_state(state, prefix + "prime." if state is not None else "")
        self.marginal.validate_state(state, prefix + "marginal." if state is not None else "")

    validate_checkpoint_state = validate_state


def partition_mlp(module, prime_channels):
    view = inspect_mlp(module)
    prime = _indices(prime_channels, view.width, "prime channels")
    selected = set(prime)
    marginal = tuple(i for i in range(view.width) if i not in selected)
    result = PartitionedMLP(_copy_slice(view, prime, output_bias=False),
                            _copy_slice(view, marginal, output_bias=False), view.down.bias)
    result.train(module.training)
    return result, {"method": "exact_channel_partition", "adapter": view.adapter,
                    "prime_current": prime, "marginal_current": marginal,
                    "prime_original": result.prime.channel_indices,
                    "marginal_original": result.marginal.channel_indices,
                    "down_refit": False, "output_bias_applications": int(view.down.bias is not None)}


@dataclass(frozen=True)
class SoLASpec:
    prime_count: int = 1
    activity_point: str = "gated_product"
    rank: int | None = None
    energy_target: float = .95
    rank_multiple: int = 1
    parameter_budget: int | None = None
    ridge: float = 1e-6
    allow_dense: bool = True
    excluded_projections: tuple[str, ...] = ()
    max_workspace_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if (type(self.prime_count) is not int or self.prime_count < 0
                or self.activity_point not in ("gated_product", "post_activation", "up_linear")
                or (self.rank is not None and (type(self.rank) is not int or self.rank < 1))
                or type(self.rank_multiple) is not int or self.rank_multiple < 1
                or type(self.energy_target) not in (int, float) or not 0 < self.energy_target <= 1
                or type(self.ridge) not in (int, float) or not math.isfinite(self.ridge) or self.ridge < 0
                or type(self.allow_dense) is not bool
                or not isinstance(self.excluded_projections, tuple)
                or len(set(self.excluded_projections)) != len(self.excluded_projections)
                or any(p not in ("up_proj", "gate_proj", "down_proj") for p in self.excluded_projections)
                or (self.parameter_budget is not None and (type(self.parameter_budget) is not int or self.parameter_budget < 0))
                or type(self.max_workspace_bytes) is not int or self.max_workspace_bytes < 1):
            raise ValueError("Invalid explicit SoLA profile")


def _fingerprint_tensor(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _fingerprint_mlp(module, view):
    digest = hashlib.sha256(repr((view.adapter, view.gated, view.activation, view.original_channels)).encode())
    for name, tensor in module.state_dict().items():
        digest.update(repr((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _moment(rows, passport):
    return finish_second_moment(update_second_moment(empty_second_moment(rows.shape[1], passport), rows))


def _records(module, prefix=""):
    return tuple(StorageCost(prefix + name, value.numel(), value.element_size(),
                             "map" if not value.is_floating_point() else "parameter")
                 for name, value in module.state_dict().items())


def sola(module, inputs, spec=SoLASpec(), *, passport=None, expected_passport=None):
    """Independent formula implementation, not a claim of full-paper reproduction.

    ``up_linear`` measures pre-activation up values. ``gated_product`` measures
    SiLU(gate)*up. The ordinary profile requires ``post_activation`` explicitly.
    Zero spectral energy has normalized retained energy one at every valid rank.
    """
    if type(spec) is not SoLASpec:
        raise ValueError("SoLASpec required")
    view = inspect_mlp(module)
    if spec.prime_count > view.width:
        raise ValueError("Prime count exceeds FFN width")
    if ((view.gated and spec.activity_point == "post_activation")
            or (not view.gated and spec.activity_point == "gated_product")
            or (not view.gated and "gate_proj" in spec.excluded_projections)):
        raise ValueError("Activity point/exclusions disagree with the declared FFN")
    if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 2 or inputs.shape[-1] != view.d_model
            or not inputs.numel() or inputs.dtype != view.up.weight.dtype or inputs.device != view.up.weight.device
            or not bool(torch.isfinite(inputs).all())):
        raise ValueError("SoLA requires finite compatible nonempty FFN calibration inputs")
    rows_count = inputs.numel() // view.d_model
    reserve = 8 * (8 * (view.width**2 + view.d_model**2) + 8 * view.width * view.d_model + rows_count * (view.d_model + 4*view.width))
    if reserve > spec.max_workspace_bytes:
        raise ValueError("SoLA workspace limit exceeded before statistics or copying")
    if passport is None:
        if expected_passport is not None:
            raise ValueError("Expected provenance requires an actual passport")
        passport = StatisticsPassport(_fingerprint_mlp(module, view), "sola-v1", "source",
                                      "mlp_input", _fingerprint_tensor(inputs), method_version="sola-v1")
    if (not isinstance(passport, StatisticsPassport) or passport.observation_point != "mlp_input"
            or passport.data_role != "calibration"):
        raise ValueError("SoLA requires a calibration passport at mlp_input")
    if expected_passport is not None and validate_passport(passport, expected_passport):
        raise ValueError("StaleStatistics: SoLA passport does not match the expected source")
    with torch.no_grad():
        rows = inputs.detach().reshape(-1, view.d_model)
        activation = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[view.activation]
        up = view.up(rows)
        features = activation(view.gate(rows)) * up if view.gated else activation(up)
        measured = up if spec.activity_point == "up_linear" else features
        scores = measured.double().square().sum(0)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("SoLA activity exceeds the finite numerical domain")
        prime = tuple(sorted(range(view.width), key=lambda i: (-float(scores[i]), i))[:spec.prime_count])
        partition, evidence = partition_mlp(module, prime)
        marginal = partition.marginal.current_indices
        evidence.update(method="sola", source_profile="AAAI-2025-independent-bounded-v1", activity_point=spec.activity_point,
                        activity_scores=scores.tolist(), marginal_deleted=False, normalized_metric="weight_sum",
                        excluded_projections=spec.excluded_projections, untouched_layers="outside_target_ffn",
                        passport=asdict(replace(passport, count=rows_count, weight_sum=float(rows_count))),
                        zero_energy_policy="retained_energy_one", workspace_reserved_bytes=reserve)
        if not marginal:
            cost = exact_unique_cost(_records(partition))
            if spec.parameter_budget is not None and cost > spec.parameter_budget:
                raise ValueError("SoLA budget is below fixed prime storage")
            evidence.update(parameter_elements=cost, ranks={}, retained_energy={})
            return partition, evidence
        input_moment = _moment(rows, passport).matrix
        feature_passport = replace(passport, observation_point="mlp_features", mask_id=str(marginal))
        full_feature_moment = _moment(features, feature_passport).matrix
        index = torch.tensor(marginal, dtype=torch.long)
        # Select G[M,M] first; never slice a precomputed Cholesky factor.
        down_moment = full_feature_moment.index_select(0, index).index_select(1, index)
        projections = ("up_proj", *(("gate_proj",) if view.gated else ()), "down_proj")
        tables, spectra, moments = [], {}, {}
        fixed = list(_records(partition.prime, "prime."))
        fixed.extend(StorageCost("marginal." + name, value.numel(), value.element_size(), "bias_or_map")
                     for name, value in partition.marginal.state_dict().items() if not name.endswith("weight"))
        if partition.output_bias is not None:
            fixed.append(StorageCost("output_bias", partition.output_bias.numel(), partition.output_bias.element_size(), "bias"))
        for name in projections:
            layer = getattr(partition.marginal, name)
            if name in spec.excluded_projections:
                fixed.append(StorageCost("marginal." + name + ".weight", layer.weight.numel(), layer.weight.element_size(), "untouched"))
                continue
            moment = down_moment if name == "down_proj" else input_moment
            moments[name] = moment
            c = moment.double() + spec.ridge * torch.eye(moment.shape[0], dtype=torch.float64)
            eig, vec = torch.linalg.eigh(c)
            root = (vec * eig.clamp_min(0).sqrt()) @ vec.T
            singular = torch.linalg.svdvals(layer.weight.double() @ root)
            spectra[name] = singular
            energy = singular.square().cumsum(0)
            total = float(energy[-1])
            maximum = min(layer.weight.shape)
            ranks = [r for r in range(1, maximum + 1) if r % spec.rank_multiple == 0]
            if spec.rank is not None:
                if spec.rank > maximum or spec.rank % spec.rank_multiple:
                    raise ValueError("SoLA explicit rank violates dimensions or rank multiple")
                ranks = [spec.rank]
            elif spec.parameter_budget is None:
                sufficient = [r for r in ranks if total == 0 or float(energy[r-1]) / total >= spec.energy_target]
                ranks = sufficient[:1]
            options = []
            for r in ranks:
                score = 1. if total == 0 else float(energy[r-1]) / total
                options.append(CandidateOption(f"factor-{r:06}", (r,), score,
                    (StorageCost(f"{name}-rank-{r}", r*sum(layer.weight.shape), layer.weight.element_size(), "factor"),)))
            if spec.allow_dense and spec.rank is None:
                options.append(CandidateOption("dense", (maximum,), 1.,
                    (StorageCost(name + "-dense", layer.weight.numel(), layer.weight.element_size(), "dense"),)))
            if not options:
                raise ValueError("No admissible SoLA rank/dense candidate")
            tables.append(CandidateTable(name, tuple(options)))
        # Without a global budget, choose the least storage meeting each local
        # energy target; a dense option must not silently dominate every score.
        local_budget = exact_unique_cost(fixed) + sum(min(exact_unique_cost(option.storages) for option in table.options) for table in tables)
        allocation = allocate(tables, local_budget if spec.parameter_budget is None else spec.parameter_budget,
                              objective=ScoreObjective("normalized_retained_energy", "maximize"), fixed_storages=fixed)
        if isinstance(allocation, BudgetInfeasible):
            raise ValueError(f"SoLA budget infeasible: minimum {allocation.minimum_cost}")
        ranks, retained = {}, {}
        for selection in allocation.selected:
            name, option = selection.table_id, selection.option
            retained[name] = option.score
            if option.option_id == "dense":
                ranks[name] = "dense"
                continue
            layer = getattr(partition.marginal, name)
            rank = option.ranks[0]
            solved = solve_weighted(layer.weight, moments[name], rank, MetricPolicy(ridge=spec.ridge))
            pair = FactorPair(solved.u * solved.s.sqrt(), solved.s.sqrt()[:, None] * solved.vh)
            replacement = FactorizedLinear.from_factors(pair, layer.bias)
            setattr(partition.marginal, name, replacement)
            ranks[name] = rank
        cost = exact_unique_cost(_records(partition))
        if cost != allocation.total_cost or (spec.parameter_budget is not None and cost > spec.parameter_budget):
            raise RuntimeError("SoLA installed storage disagrees with allocated budget")
        evidence.update(ranks=ranks, retained_energy=retained, parameter_elements=cost,
                        prime_elements=exact_unique_cost(_records(partition.prime)),
                        down_metric_shape=tuple(down_moment.shape), metric_policy={"ridge": spec.ridge, "nullspace": "support_only"})
        return partition, evidence
