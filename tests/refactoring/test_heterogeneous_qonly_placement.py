"""Tests for the static Q-only diagnostic preparation and exact delta proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from fedcore.experiments import heterogeneous_qonly_placement as placement


def _serialize(model):
    return model.SerializeToString(deterministic=True)


@pytest.fixture
def graphs(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    helper, tensor = onnx.helper, onnx.TensorProto
    prefix = "/layer1/layer1.0/"
    conv, relu = prefix + "conv1/Conv", prefix + "relu/Relu"
    conv_output = conv + "_output_0"
    target_q = conv_output + "_QuantizeLinear"
    target_dq = conv_output + "_DequantizeLinear"
    nodes, initializers = [], []

    def add_pair(source, quantize, dequantize):
        scale, zero = source + "_scale", source + "_zero_point"
        initializers.extend([
            helper.make_tensor(scale, tensor.FLOAT, [], [.25]),
            helper.make_tensor(zero, tensor.INT8, [], [0]),
        ])
        codes, result = quantize + "_Output", dequantize + "_Output"
        nodes.extend([
            helper.make_node("QuantizeLinear", [source, scale, zero], [codes], name=quantize),
            helper.make_node("DequantizeLinear", [codes, scale, zero], [result], name=dequantize),
        ])
        return result

    activation = add_pair("images", "InputQ", "InputDQ")
    logical = "layer1.0.conv1"
    weight = logical + ".weight"
    initializers.extend([
        helper.make_tensor(weight + "_quantized", tensor.INT8, [1, 1, 1, 1], [2]),
        helper.make_tensor(weight + "_scale", tensor.FLOAT, [1], [.5]),
        helper.make_tensor(weight + "_zero_point", tensor.INT8, [1], [0]),
        helper.make_tensor(logical + ".bias", tensor.FLOAT, [1], [.125]),
    ])
    nodes.extend([
        helper.make_node("DequantizeLinear", [weight + "_quantized", weight + "_scale", weight + "_zero_point"],
                         ["weights"], name="WeightsDQ", axis=0),
        helper.make_node("Conv", [activation, "weights", logical + ".bias"], [conv_output], name=conv),
    ])
    quantized_output = add_pair(conv_output, target_q, target_dq)
    nodes.append(helper.make_node("Relu", [quantized_output], ["logits"], name=relu))
    interface = [helper.make_tensor_value_info(name, tensor.FLOAT, ["batch", 1, 2, 2])
                 for name in ("images", "logits")]

    def model(graph_nodes, graph_initializers):
        graph = helper.make_graph(graph_nodes, "qonly", interface[:1], interface[1:], graph_initializers)
        result = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
        helper.set_model_props(result, {"onnx.infer": "onnxruntime.quant"})
        onnx.checker.check_model(result, full_check=True)
        return result

    source = model(nodes, initializers)
    direct_relu = helper.make_node("Relu", [conv_output], ["logits"], name=relu)
    candidate = model([*nodes[:-3], direct_relu], initializers[:-2])
    artifact = tmp_path / "Q_only" / placement.WORKFLOW / "artifact"
    artifact.mkdir(parents=True)
    model_path, engine_path = artifact / "model.onnx", artifact / "model.engine"
    model_path.write_bytes(_serialize(source))
    engine_path.write_bytes(b"frozen engine fixture")
    for field, raw in (("SOURCE_ONNX_SHA256", _serialize(source)), ("SOURCE_ENGINE_SHA256", engine_path.read_bytes()),
                       ("CANDIDATE_ONNX_SHA256", _serialize(candidate))):
        monkeypatch.setattr(placement, field, hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(placement, "_SELECTED_CONV_NODES", {logical: conv})
    return model_path, engine_path, candidate


def test_prepare_reproduces_the_expected_static_delta(graphs, tmp_path):
    model, engine, expected = graphs
    original = model.read_bytes(), engine.read_bytes()
    destination = tmp_path / "new-diagnostic"
    receipt = placement.prepare_diagnostic_graph(model, engine, destination)
    candidate = destination / placement.CANDIDATE_ONNX_NAME
    assert candidate.read_bytes() == _serialize(expected)
    assert receipt == json.loads((destination / placement.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt == placement.verify_diagnostic_graph(model, engine, candidate)
    assert receipt["diagnostic_only"] and receipt["source_files_verified_before_and_after"]
    assert receipt["node_counts"] == [7, 5]
    assert receipt["initializer_counts"] == [8, 6]
    assert receipt["input_names"] == ["images"] and receipt["output_names"] == ["logits"]
    assert receipt["model_metadata"] == {"onnx.infer": "onnxruntime.quant"}
    assert receipt["boundary"]["logical_path"] == "layer1.0.conv1"
    assert Path(receipt["candidate_onnx"]) == candidate
    assert set(path.name for path in destination.iterdir()) == {candidate.name, placement.RECEIPT_NAME}
    assert (model.read_bytes(), engine.read_bytes()) == original
    with pytest.raises(FileExistsError, match="must be fresh"):
        placement.prepare_diagnostic_graph(model, engine, destination)


@pytest.mark.parametrize("source_member", [0, 1])
def test_prepare_rejects_changed_source_without_creating_output(graphs, tmp_path, source_member):
    model, engine, _ = graphs
    changed = (model, engine)[source_member]
    changed.write_bytes(changed.read_bytes() + b"changed")
    destination = tmp_path / "unpublished"
    with pytest.raises(ValueError, match="SHA-256 differs"):
        placement.prepare_diagnostic_graph(model, engine, destination)
    assert not destination.exists()


@pytest.mark.parametrize(("change", "message"), [
    ("bias", "Unexpected initializer change"),
    ("input_scale", "Unexpected initializer change"),
    ("input_node", "Unexpected node change"),
    ("tap", "Unexpected graph interface"),
    ("target_tap", "one internal consumer input"),
    ("metadata", "Unexpected graph interface"),
])
def test_independent_proof_rejects_changes_even_with_recomputed_digest(graphs, tmp_path, monkeypatch, change, message):
    model, engine, expected = graphs
    changed = deepcopy(expected)
    if change == "bias":
        next(value for value in changed.graph.initializer if value.name == "layer1.0.conv1.bias").float_data[0] += 1
    elif change == "input_scale":
        next(value for value in changed.graph.initializer if value.name == "images_scale").float_data[0] *= 2
    elif change == "input_node":
        changed.graph.node[0].doc_string = "extra node change"
    elif change in ("tap", "target_tap"):
        tap = changed.graph.output.add()
        tap.CopyFrom(changed.graph.output[0])
        tap.name = "InputDQ_Output" if change == "tap" else "/layer1/layer1.0/conv1/Conv_output_0"
    elif change == "metadata":
        changed.metadata_props[0].value = "altered"
    candidate = tmp_path / placement.CANDIDATE_ONNX_NAME
    candidate.write_bytes(_serialize(changed))
    monkeypatch.setattr(placement, "CANDIDATE_ONNX_SHA256", hashlib.sha256(candidate.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match=message):
        placement.verify_diagnostic_graph(model, engine, candidate)


def test_prepare_requires_delta_proof_before_writing(graphs, tmp_path, monkeypatch):
    from fedcore.algorithm.heterogeneous import qdq_placement

    model, engine, expected = graphs
    expected.graph.node[0].doc_string = "unrelated change"
    monkeypatch.setattr(placement, "CANDIDATE_ONNX_SHA256", hashlib.sha256(_serialize(expected)).hexdigest())
    remove = qdq_placement.remove_output_pairs

    def faulty_remove(graph, boundaries):
        remove(graph, boundaries)
        graph.graph.node[0].doc_string = "unrelated change"

    monkeypatch.setattr(qdq_placement, "remove_output_pairs", faulty_remove)
    destination = tmp_path / "no-graph-published"
    with pytest.raises(ValueError, match="Unexpected node change"):
        placement.prepare_diagnostic_graph(model, engine, destination)
    assert not destination.exists()
