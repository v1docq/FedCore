"""Fixed constructors for portable quantization; no dynamic class imports."""
import torch
from .tensor_formats import PackedTensor, QuantizedTensorSpec
from .portable_layers import PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear


def _types():
    from .tender import TenderLinear
    return (PackedTensor, PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear, TenderLinear)


def describe_quantized(layer):
    # Tender is added by P3; importing this adapter before then stays harmless.
    try:
        types = _types()
    except ImportError:
        types = (PackedTensor, PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear)
    if type(layer) not in types:
        return None
    if (any(name in layer.__dict__ for name in ("forward", "validate_state", "state_dict", "decode", "codes"))
            or layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks):
        raise ValueError("Modified portable representation is not checkpointable")
    if type(layer) is PackedTensor:
        layer.validate_state(layer.state_dict())
        config = layer.config()
    elif type(layer) is QuantizedSumLinear:
        config = {"base": describe_quantized(layer.base), "correction": describe_quantized(layer.correction)}
    elif type(layer) is PortableQuantizedLinear:
        config = {"in_features": layer.in_features, "out_features": layer.out_features,
                  "bias": layer.bias is not None, "spec": layer.weight.spec.payload()}
    elif type(layer) is PortableQuantizedFactorLinear:
        config = {"in_features": layer.in_features, "out_features": layer.out_features, "rank": layer.rank,
                  "bias": layer.bias is not None, "left_spec": layer.left.spec.payload(), "right_spec": layer.right.spec.payload()}
    else:
        layer.validate_state(layer.state_dict())
        config = layer.config()
    return {"type": "QuantizationGraph", "schema": "quantization_graph_v1", "kind": type(layer).__name__, "config": config}


def build_quantized(description):
    if (type(description) is not dict or set(description) != {"type", "schema", "kind", "config"}
            or description["type"] != "QuantizationGraph" or description["schema"] != "quantization_graph_v1"):
        raise ValueError("Unknown quantization checkpoint schema")
    kind, config = description["kind"], description["config"]
    if type(config) is not dict:
        raise ValueError("Mapping config required")
    if kind == "PackedTensor" and set(config) == {"shape", "spec"}:
        return PackedTensor(config["shape"], QuantizedTensorSpec.parse(config["spec"]))
    if kind == "PortableQuantizedLinear" and set(config) == {"in_features", "out_features", "bias", "spec"}:
        return PortableQuantizedLinear(config["in_features"], config["out_features"], config["bias"], spec=QuantizedTensorSpec.parse(config["spec"]))
    if kind == "PortableQuantizedFactorLinear" and set(config) == {"in_features", "out_features", "rank", "bias", "left_spec", "right_spec"}:
        return PortableQuantizedFactorLinear(config["in_features"], config["out_features"], config["rank"], config["bias"],
            left_spec=QuantizedTensorSpec.parse(config["left_spec"]), right_spec=QuantizedTensorSpec.parse(config["right_spec"]))
    if kind == "QuantizedSumLinear" and set(config) == {"base", "correction"}:
        return QuantizedSumLinear(build_quantized(config["base"]), build_quantized(config["correction"]))
    if kind == "TenderLinear":
        from .tender import TenderLinear
        return TenderLinear.from_config(config)
    raise ValueError("Unknown portable constructor or fields")


def validate_quantized_state(layer, state, prefix=""):
    if type(layer) is PackedTensor:
        layer.validate_state(state, prefix)
    if type(layer) in (PortableQuantizedLinear, PortableQuantizedFactorLinear):
        if layer.bias is not None:
            bias = state[prefix+"bias"]
            if bias.dtype != torch.float64 or bias.shape != (layer.out_features,) or not bool(torch.isfinite(bias).all()):
                raise ValueError("Invalid portable bias")
    if type(layer).__name__ == "TenderLinear" and type(layer).__module__ == "fedcore.algorithm.quantization.tender":
        layer.validate_state(state, prefix)
