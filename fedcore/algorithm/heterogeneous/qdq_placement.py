"""Prove and remove isolated output Q/DQ pairs from the calibrated CV graph."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np

from .contracts import BackendContractError, PTQPlacement, fingerprint, payload


LEGACY_SCHEMA = "heterogeneous_deployment_v1"
PLACEMENT_SCHEMA = "heterogeneous_deployment_v2"
PLACEMENT_METADATA = "fedcore.ptq_placement_fingerprint"
CALIBRATION_METHOD = "onnxruntime_static_minmax_on_folded_recovered_fp32"
CALIBRATION_RANGE_USE = "omitted_output_ranges_unused_no_recalibration"
LINEAGE_IDENTITIES = ("source_M0", "recovered_float_fingerprint", "calibration_fingerprint")
LINEAGE_FILES = ("folded_fp32_onnx_sha256", "standard_qdq_onnx_sha256")


def _require(condition, message, code="UnsupportedPTQBoundary"):
    if not condition:
        raise BackendContractError(code, message)


class _GraphIndex:
    def __init__(self, model):
        import onnx

        graph = model.graph
        self.nodes = {node.name: node for node in graph.node}
        self.producers = {value: node for node in graph.node for value in node.output if value}
        self.initializers = {value.name: value for value in graph.initializer}
        self.interface = {value.name for value in (*graph.input, *graph.output)}
        self.uses = defaultdict(list)
        _require(len(self.nodes) == len(graph.node) and "" not in self.nodes
                 and not (model.functions or model.training_info or graph.sparse_initializer
                          or graph.quantization_annotation)
                 and self.interface.isdisjoint(self.initializers),
                 "PTQ placement requires a flat graph with unique named nodes")
        for node in graph.node:
            _require(node.domain in ("", "ai.onnx") and not any(
                attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for attr in node.attribute), "PTQ placement requires standard flat ONNX operations")
            for slot, value in enumerate(node.input):
                if value:
                    self.uses[value].append((node, slot))
        self.tensors = (set(self.producers) | set(self.uses) | set(self.initializers) | self.interface
                        | {value.name for value in graph.value_info})

    def only_use(self, tensor):
        uses = self.uses.get(tensor, ())
        _require(len(uses) == 1 and tensor not in self.interface,
                 "Boundary tensor must have one internal consumer input: " + tensor)
        return uses[0]


@dataclass(frozen=True)
class _Boundary:
    logical_path: str
    physical_path: str
    conv_node: str
    conv_output: str
    consumer_node: str
    consumer_op: str
    consumer_input_index: int


@dataclass(frozen=True)
class _Pair:
    nodes: tuple[str, str]
    values: tuple[str, str]
    parameters: tuple[str, str]


def _single_output(node):
    _require(len(node.output) == 1 and bool(node.output[0]), "Expected one named output from " + node.name)
    return node.output[0]


def _boundaries(index, selected, policy):
    boundaries = []
    for logical in policy.omit_output_qdq:
        candidates = [path for path in (logical, logical + ".1") if path in selected]
        first_factor = logical + ".0" in selected
        _require(len(candidates) == 1 and first_factor == (candidates[0] != logical),
                 "No unique selected terminal Conv for " + logical)
        physical, = candidates
        node = index.nodes.get(selected[physical])
        _require(node is not None and node.op_type == "Conv", "Missing selected Conv for " + physical)
        output = _single_output(node)
        consumer, slot = index.only_use(output)
        expected_op, arity = ("Relu", 1) if logical.endswith(".conv1") else ("Add", 2)
        _require(consumer.op_type == expected_op and len(consumer.input) == arity,
                 "Expected an exclusive " + expected_op + " consumer for " + logical)
        boundaries.append(_Boundary(logical, physical, node.name, output, consumer.name, consumer.op_type, slot))
    return tuple(boundaries)


def plan_output_placement(float_graph, selected, policy):
    """Resolve a logical path through the exact float exporter binding."""
    return _boundaries(_GraphIndex(float_graph), selected, policy)


def _scalar_qdq(index, quantize, dequantize, output):
    import onnx

    _require(quantize is not None and quantize.op_type == "QuantizeLinear" and len(quantize.input) == 3
             and not quantize.attribute, "Expected a per-tensor QuantizeLinear for " + output)
    _require(dequantize.op_type == "DequantizeLinear" and len(dequantize.input) == 3
             and not dequantize.attribute and tuple(quantize.input[1:]) == tuple(dequantize.input[1:]),
             "Expected matching per-tensor DequantizeLinear for " + output)
    scale_name, zero_name = quantize.input[1:]
    scale, zero = index.initializers.get(scale_name), index.initializers.get(zero_name)
    _require(scale is not None and scale.data_type == onnx.TensorProto.FLOAT and not scale.dims
             and zero is not None and zero.data_type == onnx.TensorProto.INT8 and not zero.dims,
             "Expected scalar FP32 scale and signed INT8 zero point for " + output)
    scale_value, zero_value = onnx.numpy_helper.to_array(scale), onnx.numpy_helper.to_array(zero)
    _require(np.isfinite(scale_value) and scale_value > 0 and zero_value == 0,
             "Expected positive finite scale and symmetric zero point for " + output)
    return scale_name, zero_name


def _output_pair(index, output):
    quantize, q_slot = index.only_use(output)
    quantized = _single_output(quantize)
    dequantize, dq_slot = index.only_use(quantized)
    _require(q_slot == dq_slot == 0, "Output Q/DQ must use the data inputs: " + output)
    parameters = _scalar_qdq(index, quantize, dequantize, output)
    return _Pair((quantize.name, dequantize.name), (quantized, _single_output(dequantize)), parameters)


def _receipt(boundary, pair):
    return dict(payload(boundary), removed_qdq_nodes=list(pair.nodes), removed_qdq_values=list(pair.values),
                removed_qdq_parameters=list(pair.parameters))


def remove_output_pairs(graph, boundaries):
    """Apply only disjoint, private pairs proved against the float edge plan."""
    index = _GraphIndex(graph)
    edits = []
    for boundary in boundaries:
        conv = index.nodes.get(boundary.conv_node)
        _require(conv is not None and conv.op_type == "Conv" and _single_output(conv) == boundary.conv_output,
                 "Calibrated graph changed selected Conv output: " + boundary.logical_path)
        pair = _output_pair(index, boundary.conv_output)
        consumer, slot = index.only_use(pair.values[1])
        _require((consumer.name, consumer.op_type, slot) ==
                 (boundary.consumer_node, boundary.consumer_op, boundary.consumer_input_index),
                 "Calibrated graph changed the expected consumer of " + boundary.logical_path)
        for parameter in pair.parameters:
            _require(parameter not in index.interface
                     and tuple(node.name for node, _ in index.uses[parameter]) == pair.nodes,
                     "Omitted output parameters must be private to their Q/DQ pair: " + parameter)
        edits.append((boundary, pair, consumer))
    removed_nodes, removed_values, removed_parameters = set(), set(), set()
    for boundary, pair, consumer in edits:
        consumer.input[boundary.consumer_input_index] = boundary.conv_output
        removed_nodes.update(pair.nodes)
        removed_values.update(pair.values)
        removed_parameters.update(pair.parameters)
    for items, removed in ((graph.graph.node, removed_nodes), (graph.graph.value_info, removed_values),
                           (graph.graph.initializer, removed_parameters)):
        for position in reversed(range(len(items))):
            if items[position].name in removed:
                del items[position]
    return [_receipt(boundary, pair) for boundary, pair, _ in edits]


def _constant(index, name):
    visited = set()
    while name in index.producers:
        node = index.producers[name]
        _require(node.op_type == "Identity" and len(node.input) == 1 and name not in visited,
                 "Expected a constant initializer or Identity alias: " + name)
        visited.add(name)
        name = node.input[0]
    return index.initializers.get(name)


def _float_bias(index, conv, path, channels):
    import onnx

    if path.endswith(".conv2.0"):
        _require(len(conv.input) == 2, "First Conv2 factor must remain bias-free: " + path)
        return None
    _require(len(conv.input) == 3, "Selected Conv must retain its folded FP32 bias: " + path)
    bias = _constant(index, conv.input[2])
    _require(bias is not None and bias.data_type == onnx.TensorProto.FLOAT and tuple(bias.dims) == (channels,),
             "Selected Conv bias must retain FP32 channel coordinates: " + path)
    value = onnx.numpy_helper.to_array(bias)
    _require(np.isfinite(value).all(), "Selected Conv bias must be finite: " + path)
    return value.tobytes()


def _verify_selected(index, selected, omitted, expected_biases):
    import onnx

    selected_names = set(selected.values())
    for conv in index.nodes.values():
        if conv.op_type == "Conv" and conv.name not in selected_names:
            weight = _constant(index, conv.input[1]) if len(conv.input) >= 2 else None
            _require(weight is not None and weight.data_type == onnx.TensorProto.FLOAT,
                     "Unselected Conv must retain FP32 weights: " + conv.name)
    for path, name in selected.items():
        conv = index.nodes.get(name)
        _require(conv is not None and conv.op_type == "Conv" and len(conv.input) >= 2,
                 "Missing selected INT8 Conv: " + name)
        dq = index.producers.get(conv.input[1])
        weight_name = path + ".weight_quantized"
        weight = index.initializers.get(weight_name)
        _require(dq is not None and dq.op_type == "DequantizeLinear"
                 and len(dq.input) == 3 and dq.input[0] == weight_name and weight is not None
                 and weight.data_type == onnx.TensorProto.INT8 and len(weight.dims) == 4,
                 "Selected Conv lost its exact INT8 weight initializer: " + path)
        scale, zero = (index.initializers.get(value) for value in dq.input[1:])
        _require(scale is not None and scale.data_type == onnx.TensorProto.FLOAT
                 and zero is not None and zero.data_type == onnx.TensorProto.INT8
                 and tuple(scale.dims) == tuple(zero.dims) == (weight.dims[0],),
                 "Selected Conv must retain per-channel weight scale and zero point: " + path)
        activation_dq = index.producers.get(conv.input[0])
        _require(activation_dq is not None and activation_dq.op_type == "DequantizeLinear"
                 and len(activation_dq.input) == 3, "Selected Conv must retain input DequantizeLinear: " + path)
        _scalar_qdq(index, index.producers.get(activation_dq.input[0]), activation_dq, conv.input[0])
        bias = _float_bias(index, conv, path, weight.dims[0])
        if expected_biases is not None:
            _require(bias == expected_biases[path], "Placement changed the recovered FP32 Conv bias: " + path)
        if name not in omitted:
            _output_pair(index, _single_output(conv))


def _identity(manifest):
    return fingerprint({key: manifest[key] for key in (
        "workflow_fingerprint", "selected_conv_nodes", "backend_spec", "versions", "ptq_placement")})


def _calibration_batch_size(manifest):
    spec = manifest["backend_spec"]
    if (spec["runtime"] != "onnxruntime" or type(spec["opset"]) is not int or spec["opset"] != 17
            or any(type(spec[key]) is not int or spec[key] < 1 for key in ("calibration_batch_size", "max_batch_size"))):
        raise ValueError("Invalid ONNX Runtime calibration configuration")
    return min(spec["calibration_batch_size"], spec["max_batch_size"])


def bind_placement_evidence(graph, manifest, policy, receipts, *, folded_fp32_sha, standard_qdq_sha,
                            calibration_rows):
    manifest["schema"] = PLACEMENT_SCHEMA
    manifest["ptq_placement"] = dict(
        policy=payload(policy), resolved_boundaries=receipts, calibration_lineage=dict(
            method=CALIBRATION_METHOD, range_use=CALIBRATION_RANGE_USE,
            onnxruntime_version=manifest["versions"]["onnxruntime"], calibration_rows=calibration_rows,
            effective_calibration_batch_size=_calibration_batch_size(manifest),
            folded_fp32_onnx_sha256=folded_fp32_sha, standard_qdq_onnx_sha256=standard_qdq_sha,
            **{key: manifest[key] for key in LINEAGE_IDENTITIES}))
    digest = _identity(manifest)
    manifest["ptq_placement_fingerprint"] = digest
    _require(not any(entry.key == PLACEMENT_METADATA for entry in graph.metadata_props),
             "The calibrated graph already declares a placement policy")
    entry = graph.metadata_props.add()
    entry.key, entry.value = PLACEMENT_METADATA, digest


def _sha256_text(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_placement_evidence(graph, manifest, *, expected_biases=None):
    """Verify the policy-bound final graph at export and independently on load."""
    markers = [entry.value for entry in graph.metadata_props if entry.key == PLACEMENT_METADATA]
    if manifest["schema"] == LEGACY_SCHEMA:
        _require(not markers, "An explicit placement graph requires a placement manifest", "InvalidPTQPlacementManifest")
        return
    try:
        evidence = manifest["ptq_placement"]
        if type(evidence) is not dict or set(evidence) != {"policy", "resolved_boundaries", "calibration_lineage"}:
            raise ValueError("Expected policy, resolved boundaries and calibration lineage")
        policy = PTQPlacement.from_payload(evidence["policy"])
        lineage = evidence["calibration_lineage"]
        if (type(lineage) is not dict or set(lineage) != {
                *LINEAGE_IDENTITIES, *LINEAGE_FILES, "method", "range_use", "calibration_rows",
                "onnxruntime_version", "effective_calibration_batch_size"}
                or lineage["method"] != CALIBRATION_METHOD or lineage["range_use"] != CALIBRATION_RANGE_USE
                or type(lineage["calibration_rows"]) is not int or lineage["calibration_rows"] < 1
                or type(lineage["effective_calibration_batch_size"]) is not int
                or lineage["effective_calibration_batch_size"] != _calibration_batch_size(manifest)
                or type(lineage["onnxruntime_version"]) is not str or not lineage["onnxruntime_version"]
                or lineage["onnxruntime_version"] != manifest["versions"]["onnxruntime"]
                or any(not _sha256_text(lineage[key]) for key in (*LINEAGE_IDENTITIES, *LINEAGE_FILES))
                or any(lineage[key] != manifest[key] for key in LINEAGE_IDENTITIES)
                or not _sha256_text(manifest["workflow_fingerprint"])):
            raise ValueError("Placement calibration lineage must bind this candidate and calibration")
        digest = _identity(manifest)
        if manifest["ptq_placement_fingerprint"] != digest or markers != [digest]:
            raise ValueError("Placement fingerprint must match the ONNX graph and complete manifest evidence")
        selected = manifest["selected_conv_nodes"]
        paths = manifest["selected_conv_paths"]
        if (type(selected) is not dict or type(paths) is not list or paths != sorted(selected)
                or any(not re.fullmatch(r"layer[1-4]\.[01]\.conv(?:1|2(?:\.[01])?)", path) for path in paths)
                or any(type(name) is not str or not name for name in selected.values())
                or len(set(selected.values())) != len(selected)):
            raise ValueError("Placement requires canonical selected physical Conv bindings")
        for block in {path.rsplit(".conv", 1)[0] for path in paths}:
            members = {path.removeprefix(block + ".") for path in paths if path.startswith(block + ".")}
            if members not in ({"conv1", "conv2"}, {"conv1", "conv2.0", "conv2.1"}):
                raise ValueError("Selected block requires complete Conv coverage: " + block)
        index = _GraphIndex(graph)
        boundaries = _boundaries(index, selected, policy)
        receipts = evidence["resolved_boundaries"]
        if type(receipts) is not list or len(receipts) != len(boundaries):
            raise ValueError("Placement requires one receipt per target")
        for boundary, receipt in zip(boundaries, receipts):
            fields = payload(boundary)
            if (type(receipt) is not dict or set(receipt) != {*fields, "removed_qdq_nodes", "removed_qdq_values",
                                                            "removed_qdq_parameters"}
                    or any(type(receipt[key]) is not type(value) or receipt[key] != value
                           for key, value in fields.items())):
                raise ValueError("Resolved placement differs from the graph: " + boundary.logical_path)
            for key, live in (("removed_qdq_nodes", index.nodes), ("removed_qdq_values", index.tensors),
                              ("removed_qdq_parameters", index.tensors)):
                names = receipt[key]
                if (type(names) is not list or len(names) != 2 or any(type(name) is not str or not name
                                                                    or name in live for name in names)
                        or names[0] == names[1]):
                    raise ValueError("Omitted Q/DQ values, nodes and parameters must be absent")
        _verify_selected(index, selected, {boundary.conv_node for boundary in boundaries}, expected_biases)
    except BackendContractError:
        raise
    except (TypeError, KeyError, ValueError) as error:
        raise BackendContractError("InvalidPTQPlacementManifest", str(error)) from error
