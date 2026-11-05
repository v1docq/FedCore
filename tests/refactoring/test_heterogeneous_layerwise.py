"""CPU-only landmark, residual closure and instrumentation contracts."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from fedcore.experiments import heterogeneous_layerwise as layerwise


def selection():
    ort = np.zeros((2000, 10), np.float32)
    trt = ort.copy()
    for i, margin in ((0, 100.), (3, 6.), (7, 9.), (9, 9.), (20, 8.), (40, 7.), (60, 5.), (80, 4.)):
        ort[i, 0] = margin
        trt[i, 1] = margin
    return layerwise.select_calibration_rows(tuple(range(100, 2100)), ort, trt)


def graph():
    nodes = [helper.make_node("Relu", ["images"], ["stem"], name="/relu/Relu"),
             helper.make_node("QuantizeLinear", ["stem", "scale", "zero"], ["stem_q"], name="Q"),
             helper.make_node("DequantizeLinear", ["stem_q", "scale", "zero"], ["stem_dq"], name="DQ"),
             helper.make_node("Identity", ["stem_dq"], ["main"], name="main_branch"),
             helper.make_node("Add", ["main", "stem"], ["sum"], name="residual_add")]
    previous = "sum"
    for i, (_, name) in enumerate(layerwise.LANDMARK_NODES[1:9]):
        output = f"block{i}"
        nodes.append(helper.make_node("Relu", [previous], [output], name=name))
        previous = output
    nodes.extend((helper.make_node("GlobalAveragePool", [previous], ["pool"], name="/avgpool/GlobalAveragePool"),
                  helper.make_node("Flatten", ["pool"], ["flat"], name="/Flatten"),
                  helper.make_node("Gemm", ["flat", "weight"], ["logits"], name="/fc/Gemm")))
    initializers = [helper.make_tensor("scale", TensorProto.FLOAT, [], [.5]),
                    helper.make_tensor("zero", TensorProto.INT8, [], [0]),
                    helper.make_tensor("weight", TensorProto.FLOAT, [3, 10], [0.] * 30)]
    return helper.make_model(helper.make_graph(nodes, "test", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 2, 2])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])], initializers),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def make_plan(model=None):
    model = model or graph()
    inspector = json.dumps(dict(Layers=[
        dict(Name="fusion", LayerType="Conv", Metadata="[ONNX Layer: /relu/Relu]", Outputs=[{"Name": "stem_q", "Format/Datatype": "Int8"}]),
        dict(Name="wrong_substring", LayerType="Conv", Metadata="[ONNX Layer: /layer1/layer1.0/relu_1/Relu_suffix]", Outputs=[{"Name": "unrelated"}]),
        dict(Name="fc", LayerType="Gemm", Metadata="[ONNX Layer: /fc/Gemm]", Outputs=[{"Name": "internal_fc"}]),
        dict(Name="reshape", LayerType="NoOp", Metadata="[ONNX Layer: /fc/Gemm]", Outputs=[{"Name": "logits"}]),
    ])).encode()
    plan = layerwise.build_layerwise_plan(model, inspector, selection(),
        source_onnx_sha256=hashlib.sha256(model.SerializeToString()).hexdigest(),
        source_engine_sha256="a" * 64, source_inspector_sha256=hashlib.sha256(inspector).hexdigest())
    return model, plan


def test_selection_preserves_original_two_then_six_margin_ranked_unique_disagreements():
    rows = selection()
    assert [row.calibration_position for row in rows] == [0, 1, 7, 9, 20, 40, 3, 60]
    assert [row.train_index for row in rows] == [100, 101, 107, 109, 120, 140, 103, 160]
    assert all(row.reason == "largest_ORT_margin_disagreement" for row in rows[2:])


def test_selection_with_no_disagreements_keeps_only_original_two():
    logits = np.zeros((2000, 10), np.float32)
    assert len(layerwise.select_calibration_rows(tuple(range(2000)), logits, logits)) == 2
    with pytest.raises(ValueError):
        layerwise.select_calibration_rows(tuple(range(2000)), logits, logits, max_rows=9)


def test_real_cpu_ort_instrumentation_preserves_logits_and_exposes_planned_values():
    ort = pytest.importorskip("onnxruntime")
    model = graph()
    # Nonconstant logits expose shape/order mistakes that zero weights would hide.
    weight = model.graph.initializer[-1]
    weight.CopyFrom(onnx.numpy_helper.from_array(np.arange(30, dtype=np.float32).reshape(3, 10)/30, name="weight"))
    _, plan = make_plan(model)
    source = model.SerializeToString()
    enriched = layerwise.instrument_onnx_copy(model, plan)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    original_session = ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])
    enriched_session = ort.InferenceSession(enriched.SerializeToString(), options, providers=["CPUExecutionProvider"])
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)/10 - 1
    expected = original_session.run(["logits"], {"images": inputs})[0]
    names = [value.name for value in enriched.graph.output]
    observed = dict(zip(names, enriched_session.run(names, {"images": inputs})))
    np.testing.assert_allclose(observed["logits"], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(observed["stem"], np.maximum(inputs, 0))
    assert np.ptp(expected) > 0 and len(observed) == 12
    assert model.SerializeToString() == source
    for landmark in plan.landmarks:
        assert observed[landmark.tensor].shape == (2, *landmark.shape[1:])


def test_landmarks_keep_complete_residual_qdq_closure_and_exact_or_ambiguous_mappings():
    _, plan = make_plan()
    assert len(plan.landmarks) == 12 and plan.total_tap_bytes < layerwise.MAX_TAP_BYTES
    first_block = plan.landmarks[1]
    assert {"/relu/Relu", "Q", "DQ", "main_branch", "residual_add", first_block.producer} == set(first_block.ancestor_node_names)
    assert first_block.required_initializers == ("scale", "zero")
    assert first_block.cut_inputs == ("images",)
    assert first_block.inspector_status == "unmapped"  # substring is never provenance
    stem = plan.landmarks[0]
    assert stem.quantized_companion_tensors == ("stem_q",)
    assert stem.inspector_status == "fused_or_quantized_region_only"
    assert "quantized_companion_output" in stem.inspector_matches[0].relations
    assert plan.landmarks[-1].inspector_status == "ambiguous_multiple_layers"
    assert len(plan.landmarks[-1].inspector_matches) == 2
    assert plan.required_controls == layerwise.REQUIRED_CONTROLS
    assert plan.engine_bytes_verified is False and plan.prefix_cuts_are_metadata_only


def test_instrumentation_returns_independent_valid_copy_preserving_source_and_logits():
    model, plan = make_plan()
    before = model.SerializeToString()
    copied = layerwise.instrument_onnx_copy(model, plan)
    assert model.SerializeToString() == before
    assert copied is not model and len(copied.graph.output) == 12
    assert copied.graph.output[0].name == "logits"
    assert len({v.name for v in copied.graph.output}) == 12
    assert [n.SerializeToString() for n in copied.graph.node] == [n.SerializeToString() for n in model.graph.node]
    assert [n.SerializeToString() for n in copied.graph.initializer] == [n.SerializeToString() for n in model.graph.initializer]
    onnx.checker.check_model(copied)


def test_instrumentation_rejects_changed_source():
    model, plan = make_plan()
    model.graph.name = "changed"
    with pytest.raises(ValueError, match="exact original graph"):
        layerwise.instrument_onnx_copy(model, plan)


def test_bounded_tensor_summaries_retain_hashes_errors_and_max16_coordinates():
    reference = np.zeros((2, 3, 2, 2), np.float32)
    actual = reference.copy()
    actual[1, 2, 1, 1] = .04
    result = layerwise.compare_tensors(reference, actual)
    assert result["violating_count"] == 1 and result["max_normalized_error"] == pytest.approx(2)
    assert result["worst_coordinates"][0] == [1, 2, 1, 1]
    assert len(result["worst_coordinates"]) == 16
    assert result["reference"]["sha256"] != result["actual"]["sha256"]
    assert result["deployment_accepted"] is False
    with pytest.raises(ValueError):
        layerwise.summarize_tensor(np.zeros(layerwise.MAX_TENSOR_ELEMENTS + 1, np.float32))
    with pytest.raises(ValueError):
        layerwise.summarize_tensor(np.array([np.nan], np.float32))


def test_plan_is_serializable_but_does_not_claim_engine_inspection_or_runtime_execution():
    _, plan = make_plan()
    value = json.loads(json.dumps(asdict(plan)))
    assert value["original_engine_internal_tensors_available"] is False
    assert value["engine_bytes_verified"] is False
    assert value["required_controls"] == list(layerwise.REQUIRED_CONTROLS)
