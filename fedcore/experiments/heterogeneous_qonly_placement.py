"""Prepare and independently verify the fixed Conv1 output-Q/DQ diagnostic."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .heterogeneous_qonly_diagnostic import SAVED_SHA256, WORKFLOW, _sha


SOURCE_ONNX_SHA256 = SAVED_SHA256["model.onnx"]
SOURCE_ENGINE_SHA256 = SAVED_SHA256["model.engine"]
CANDIDATE_ONNX_SHA256 = "762231509175d9038cb19236981ee706714d4dbf17504b66dfb1ac2055bd26cd"
CANDIDATE_ONNX_NAME = "qonly-conv1-output-qdq-omitted.onnx"
RECEIPT_NAME = "graph-receipt.json"

_LOGICAL_PATH = "layer1.0.conv1"
_CONV = "/layer1/layer1.0/conv1/Conv"
_CONV_OUTPUT = _CONV + "_output_0"
_RELU = "/layer1/layer1.0/relu/Relu"
_PAIR_NODES = tuple(_CONV_OUTPUT + "_" + operation for operation in ("QuantizeLinear", "DequantizeLinear"))
_PAIR_VALUES = tuple(node + "_Output" for node in _PAIR_NODES)
_PAIR_PARAMETERS = (_CONV_OUTPUT + "_scale", _CONV_OUTPUT + "_zero_point")
_SELECTED_CONV_NODES = {
    f"layer{stage}.{block}.conv{conv}": f"/layer{stage}/layer{stage}.{block}/conv{conv}/Conv"
    for stage in range(1, 5) for block in (0, 1) for conv in (1, 2)
}


def _source_paths(source_model, source_engine):
    model, engine = Path(source_model), Path(source_engine)
    if model.is_symlink() or engine.is_symlink():
        raise ValueError("The frozen Q-only ONNX and engine must be regular files")
    model, engine = model.resolve(strict=True), engine.resolve(strict=True)
    root = model.parent
    if (model.name != "model.onnx" or engine != root / "model.engine"
            or root.name != "artifact" or root.parent.name != WORKFLOW or root.parent.parent.name != "Q_only"
            or not model.is_file() or not engine.is_file()
            or (root / "manifest.json").exists() or (root / "manifest.json").is_symlink()):
        raise ValueError("Expected the frozen rejected Q-only artifact directory")
    return model, engine


def _read_frozen_model(path, expected_sha256):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("ONNX SHA-256 differs from the fixed diagnostic identity: " + str(path))
    return raw


def _check_frozen_engine(engine):
    if _sha(engine) != SOURCE_ENGINE_SHA256:
        raise ValueError("Engine SHA-256 differs from the frozen Q-only identity")


def _load_graph(raw):
    import onnx

    model = onnx.load_model_from_string(raw)
    onnx.checker.check_model(model, full_check=True)
    return model


def _serialized(message):
    return message.SerializeToString(deterministic=True)


def _boundary():
    from fedcore.algorithm.heterogeneous.qdq_placement import _Boundary

    return _Boundary(_LOGICAL_PATH, _LOGICAL_PATH, _CONV, _CONV_OUTPUT, _RELU, "Relu", 0)


def _prove_delta(source, candidate):
    """Compare the independently expected ordered protobuf delta."""
    from fedcore.algorithm.heterogeneous.contracts import payload
    from fedcore.algorithm.heterogeneous.qdq_placement import _GraphIndex, _output_pair, _verify_selected

    source_index, candidate_index = _GraphIndex(source), _GraphIndex(candidate)
    _verify_selected(source_index, _SELECTED_CONV_NODES, set(), None)
    _verify_selected(candidate_index, _SELECTED_CONV_NODES, {_CONV}, None)

    pair = _output_pair(source_index, _CONV_OUTPUT)
    old_relu, slot = source_index.only_use(pair.values[1])
    new_relu, new_slot = candidate_index.only_use(_CONV_OUTPUT)
    if ((pair.nodes, pair.values, pair.parameters) != (_PAIR_NODES, _PAIR_VALUES, _PAIR_PARAMETERS)
            or (old_relu.name, old_relu.op_type, tuple(old_relu.input), slot)
            != (_RELU, "Relu", (_PAIR_VALUES[1],), 0)
            or (new_relu.name, new_slot) != (_RELU, 0)):
        raise ValueError("The exact private Conv1 output Q/DQ and ReLU input are required")
    for offset, parameter in enumerate(_PAIR_PARAMETERS, start=1):
        uses = source_index.uses[parameter]
        if (parameter in source_index.interface
                or tuple((node.name, slot) for node, slot in uses)
                != tuple((node, offset) for node in _PAIR_NODES)):
            raise ValueError("Only private Conv1 output Q/DQ parameters can be removed")

    expected_relu = deepcopy(old_relu)
    expected_relu.input[0] = _CONV_OUTPUT
    expected_nodes = [_serialized(expected_relu if node.name == _RELU else node)
                      for node in source.graph.node if node.name not in _PAIR_NODES]
    expected_initializers = [_serialized(tensor) for tensor in source.graph.initializer
                             if tensor.name not in _PAIR_PARAMETERS]
    if [_serialized(node) for node in candidate.graph.node] != expected_nodes:
        raise ValueError("Unexpected node change outside the single ReLU input and omitted Q/DQ")
    if [_serialized(tensor) for tensor in candidate.graph.initializer] != expected_initializers:
        raise ValueError("Unexpected initializer change outside the private Q/DQ scale and zero point")

    source_envelope, candidate_envelope = deepcopy(source), deepcopy(candidate)
    for model in (source_envelope, candidate_envelope):
        model.graph.ClearField("node")
        model.graph.ClearField("initializer")
    if _serialized(source_envelope) != _serialized(candidate_envelope):
        raise ValueError("Unexpected graph interface, value info or model metadata change")

    return dict(
        boundary=dict(payload(_boundary()), removed_qdq_nodes=list(_PAIR_NODES),
                      removed_qdq_values=list(_PAIR_VALUES), removed_qdq_parameters=list(_PAIR_PARAMETERS)),
        selected_conv_nodes=dict(_SELECTED_CONV_NODES),
        node_counts=[len(source.graph.node), len(candidate.graph.node)],
        initializer_counts=[len(source.graph.initializer), len(candidate.graph.initializer)],
        value_info_counts=[len(source.graph.value_info), len(candidate.graph.value_info)],
        input_names=[value.name for value in source.graph.input],
        output_names=[value.name for value in source.graph.output],
        model_metadata={entry.key: entry.value for entry in source.metadata_props},
        retained_nodes_match_except_relu_input=True,
        retained_initializers_match=True,
        graph_interface_value_info_and_metadata_match=True,
        input_qdq_int8_weights_and_fp32_biases_verified=True,
    )


def verify_diagnostic_graph(source_model, source_engine, candidate_model):
    """Verify checksums and exact serialized graph delta without consulting a receipt."""
    model, engine = _source_paths(source_model, source_engine)
    candidate = Path(candidate_model)
    if candidate.is_symlink():
        raise ValueError("The candidate ONNX must be a regular file")
    candidate = candidate.resolve(strict=True)
    if candidate.name != CANDIDATE_ONNX_NAME or not candidate.is_file() or model.parent.parent in candidate.parents:
        raise ValueError("Expected the fixed diagnostic ONNX outside the source workflow directory")
    source_raw = _read_frozen_model(model, SOURCE_ONNX_SHA256)
    _check_frozen_engine(engine)
    candidate_raw = _read_frozen_model(candidate, CANDIDATE_ONNX_SHA256)
    proof = _prove_delta(_load_graph(source_raw), _load_graph(candidate_raw))
    if _sha(model) != SOURCE_ONNX_SHA256 or _sha(candidate) != CANDIDATE_ONNX_SHA256:
        raise ValueError("An ONNX file changed during diagnostic graph verification")
    _check_frozen_engine(engine)
    return _receipt(model, engine, candidate, len(candidate_raw), proof)


def _receipt(model, engine, candidate, candidate_size, proof):
    return dict(
        schema="heterogeneous-qonly-placement-diagnostic-graph-v1",
        diagnostic_only=True,
        source_workflow_fingerprint=WORKFLOW,
        source_onnx=str(model),
        source_onnx_sha256=SOURCE_ONNX_SHA256,
        source_engine=str(engine),
        source_engine_sha256=SOURCE_ENGINE_SHA256,
        candidate_onnx=str(candidate),
        candidate_onnx_sha256=CANDIDATE_ONNX_SHA256,
        candidate_onnx_bytes=candidate_size,
        source_files_verified_before_and_after=True,
        **proof,
    )


def prepare_diagnostic_graph(source_model, source_engine, output_directory):
    """Write the checked diagnostic ONNX and receipt to a fresh directory."""
    from fedcore.algorithm.heterogeneous.qdq_placement import remove_output_pairs

    model, engine = _source_paths(source_model, source_engine)
    destination = Path(output_directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("The diagnostic output directory must be fresh")
    destination = destination.resolve()
    if model.parent.parent in destination.parents:
        raise ValueError("The diagnostic output directory must be outside the source workflow directory")

    source_raw = _read_frozen_model(model, SOURCE_ONNX_SHA256)
    _check_frozen_engine(engine)
    source = _load_graph(source_raw)
    graph = deepcopy(source)
    remove_output_pairs(graph, (_boundary(),))
    candidate_raw = _serialized(graph)
    if hashlib.sha256(candidate_raw).hexdigest() != CANDIDATE_ONNX_SHA256:
        raise ValueError("The constructed ONNX differs from the fixed diagnostic identity")
    proof = _prove_delta(source, _load_graph(candidate_raw))

    destination.mkdir()
    candidate = destination / CANDIDATE_ONNX_NAME
    with candidate.open("xb") as stream:
        stream.write(candidate_raw)
    if _sha(model) != SOURCE_ONNX_SHA256 or _sha(candidate) != CANDIDATE_ONNX_SHA256:
        raise ValueError("An ONNX file changed during diagnostic graph preparation")
    _check_frozen_engine(engine)
    receipt = _receipt(model, engine, candidate, len(candidate_raw), proof)
    with (destination / RECEIPT_NAME).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-engine", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare_diagnostic_graph(args.source_model, args.source_engine, args.output_directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0
