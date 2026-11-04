"""Real CPU ONNX Q/DQ round trip; no CUDA/TensorRT performance claim."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from fedcore.algorithm.heterogeneous.backend import (
    BackendContractError, BackendSpec, DeploymentArtifact, deployment_size,
    export_onnx_qdq, inspect_qdq, load_deployment, _confirmed_int8_layer, _check_tensorrt_version)
from fedcore.algorithm.heterogeneous.cv import (
    BlockRecipe, HeterogeneousGene, HeterogeneousGenotype, compile_genotype,
    execute_float_workflow, prepare_space)
from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18


def test_backend_probe_failure_keeps_logits_and_existing_tolerances(tmp_path):
    from fedcore.algorithm.heterogeneous.backend import _check_backend_probe
    reference = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    actual = reference.copy()
    actual[0, 1] += .5
    path = tmp_path / "backend-probe.json"
    with pytest.raises(BackendContractError, match="max_abs_error=0.5"):
        _check_backend_probe(reference, actual, BackendSpec(), path)
    evidence = json.loads(path.read_text())
    assert evidence["accepted"] is False
    assert evidence["backend_rtol"] == evidence["backend_atol"] == .02
    assert evidence["max_abs_error"] == .5 and evidence["argmax_agreement"] == 1.
    assert evidence["violating_count"] == 1 and evidence["violating_indices"] == [[0, 1]]
    assert evidence["cpu_qdq_logits"] == reference.tolist()
    assert evidence["tensorrt_logits"] == actual.tolist()
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("actual", [np.ones((1, 3), dtype=np.float32),
                                    np.array([[np.nan]], dtype=np.float32)])
def test_backend_probe_rejects_shape_or_nonfinite_outputs(tmp_path, actual):
    from fedcore.algorithm.heterogeneous.backend import _check_backend_probe
    with pytest.raises(BackendContractError, match="Finite matching"):
        _check_backend_probe(np.ones((1, 1), dtype=np.float32), actual, BackendSpec(),
                             tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_backend_probe_zero_tolerance_keeps_finite_json_diagnostics(tmp_path):
    from fedcore.algorithm.heterogeneous.backend import _check_backend_probe
    path = tmp_path / "zero-tolerance.json"
    with pytest.raises(BackendContractError):
        _check_backend_probe(np.zeros((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32),
            BackendSpec(backend_rtol=0, backend_atol=0), path)
    evidence = json.loads(path.read_text())
    assert evidence["max_normalized_error"] is None and evidence["zero_bound_violations"] == 2


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        model = CifarResNet18().eval()
        calibration = torch.randn(2, 3, 32, 32)
    recipe = BlockRecipe(16, 8, True)
    space = prepare_space(model, calibration, alternatives={"layer1.0": (recipe,), "layer1.1": (BlockRecipe(),)})
    workflow = compile_genotype(HeterogeneousGenotype((HeterogeneousGene("layer1.0", recipe),
                                                      HeterogeneousGene("layer1.1", BlockRecipe()))), space)
    return model, calibration, workflow


def test_real_compact_factor_conv_qdq_export_and_independent_reload(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model, calibration, workflow = fixture()
    source = model.layer1[0].conv1.weight.detach().clone()
    result = execute_float_workflow(model, calibration, workflow)
    assert type(result) is CVWorkflowSuccess
    assert result.model.block_widths[0] == 16 and result.model.conv2_ranks[0] == 8
    artifact = export_onnx_qdq(result.model, calibration, workflow, tmp_path / "deploy",
                               spec=BackendSpec(max_batch_size=8, opt_batch_size=2))
    assert type(artifact) is DeploymentArtifact, artifact
    graph = onnx.load(artifact.model_path)
    assert len(inspect_qdq(graph, artifact.evidence["qdq_verified_nodes"])) == 3
    # TensorRT 10.0.1 rejects INT32 bias DequantizeLinear. Preserve float
    # biases while activations and Conv weights remain explicitly INT8.
    constants = {weight.name: weight for weight in graph.graph.initializer}
    assert not any(node.op_type == "DequantizeLinear" and node.input[0] in constants
                   and constants[node.input[0]].data_type == onnx.TensorProto.INT32
                   for node in graph.graph.node)
    assert artifact.evidence["conv_bias_policy"] == "float32_no_INT32_dequantize"
    assert any(weight.data_type == onnx.TensorProto.INT8 for weight in graph.graph.initializer)
    assert artifact.storage_bytes == deployment_size(artifact.manifest_path)
    loaded = load_deployment(artifact.manifest_path)
    first = loaded(calibration)
    assert first.shape == (2, 10) and torch.isfinite(first).all()
    assert np.array_equal(first.numpy(), artifact.load()(calibration.numpy()))
    assert torch.equal(model.layer1[0].conv1.weight, source)
    assert artifact.evidence["hardware_validation"] == "CPU_ONNX_Runtime_only"
    assert artifact.evidence["representation"]["channel_indices"][0] == list(result.model.channel_indices[0])
    with pytest.raises(BackendContractError, match="CIFAR"):
        loaded(np.zeros((9, 3, 32, 32), dtype=np.float32))
    # The loader binds actual package bytes, not just a pre-existing in-memory session.
    with open(artifact.model_path, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(BackendContractError, match="changed"):
        artifact.load()


def test_export_rejects_stale_calibration_and_missing_float_provenance(tmp_path):
    model, calibration, workflow = fixture()
    failure = export_onnx_qdq(model, calibration, workflow, tmp_path / "missing")
    assert isinstance(failure, StructuralInvalid)
    assert failure.violations[0].code == "StaleCandidate"
    failure = export_onnx_qdq(model, calibration + 1, workflow, tmp_path / "stale")
    assert isinstance(failure, StructuralInvalid)
    assert failure.violations[0].code == "StaleCalibration"
    assert not (tmp_path / "missing").exists() and not (tmp_path / "stale").exists()


def test_declared_int8_conv_cannot_silently_fall_back_to_float():
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_model(helper.make_graph([
        helper.make_node("Conv", ["x", "w"], ["y"], name="claimed_int8")], "float_fallback", [], [], []))
    with pytest.raises(BackendContractError, match="lacks Q/DQ"):
        inspect_qdq(graph, ("claimed_int8",))


def test_tensorrt_coverage_requires_exact_metadata_and_integer_conv_precision():
    layer = {"Name": "arbitrary kernel", "Metadata": "[ONNX Layer: Conv_10]", "LayerType": "CaskConvolution",
             "Inputs": [{"Format/Datatype": "INT8 row-major"}], "Precision": "INT8"}
    assert _confirmed_int8_layer([layer], "Conv_10")
    assert not _confirmed_int8_layer([layer], "Conv_1")
    assert not _confirmed_int8_layer([{**layer, "Metadata": "", "Name": "Conv_10"}], "Conv_10")
    assert not _confirmed_int8_layer([{**layer, "Precision": "Float", "Outputs": [{"Datatype": "INT8"}]}], "Conv_10")
    assert not _confirmed_int8_layer([{**layer, "Precision": "Float", "Weights": {"Datatype": "INT8"},
                                      "Inputs": [{"Datatype": "INT8"}, {"Datatype": "INT8"}]}], "Conv_10")
    assert not _confirmed_int8_layer([{**layer, "LayerType": "Reformat"}], "Conv_10")
    assert not _confirmed_int8_layer([{**layer, "Inputs": [{"Datatype": "Float"}]}], "Conv_10")
    class Unsupported:
        __version__ = "11.0.0"
    with pytest.raises(BackendContractError, match="TensorRT 10"):
        _check_tensorrt_version(Unsupported)
