"""Public CPU export and graph proofs for the opt-in placement policy."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fedcore.algorithm.heterogeneous import (
    BlockRecipe, HeterogeneousGene, HeterogeneousGenotype, PTQPlacement, compile_genotype,
    execute_float_workflow, genotype_from_payload, genotype_payload, prepare_space)
from fedcore.algorithm.heterogeneous.backend import (
    BackendContractError, BackendSpec, DeploymentArtifact, export_onnx_qdq)
from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18


def _source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = CifarResNet18().eval()
        calibration = torch.randn(2, 3, 32, 32)
    return model, calibration


def _genotype(recipe):
    return HeterogeneousGenotype((HeterogeneousGene("layer1.0", recipe),))


@pytest.mark.parametrize("paths", [(), ("layer1.0.conv2.1",), ("layer1.0.conv1", "layer1.0.conv1"),
                                   ("layer1.0.conv2", "layer1.0.conv1")])
def test_placement_requires_canonical_logical_paths(paths):
    with pytest.raises(ValueError, match="logical Conv"):
        PTQPlacement(paths)


def test_placement_is_bound_to_an_enabled_ptq_step_in_its_own_scope():
    model, calibration = _source()
    wrong_scope = PTQPlacement(("layer1.1.conv1",))
    with pytest.raises(ValueError, match="enabled PTQ"):
        BlockRecipe(ptq_placement=wrong_scope)
    result = prepare_space(model, calibration, alternatives={
        "layer1.0": (BlockRecipe(ptq=True, ptq_placement=wrong_scope),)})
    assert type(result) is ValidationFailure
    assert result.violations[0].code == "InvalidPTQPlacement"


@pytest.mark.parametrize(("logical", "physical", "consumer_op"), [
    ("layer1.0.conv1", "layer1.0.conv1", "Relu"),
    ("layer1.0.conv2", "layer1.0.conv2.1", "Add"),
])
def test_placement_public_cpu_export_changes_only_the_proven_pair(
        logical, physical, consumer_op, tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnxruntime import quantization

    model, calibration = _source()
    if consumer_op == "Relu":
        with torch.no_grad():
            model.layer1[0].bn1.bias.copy_(torch.linspace(.125, .375, 64))
    baseline = BlockRecipe(16, 8, True)
    policy = PTQPlacement((logical,))
    recipe = replace(baseline, ptq_placement=policy)
    space = prepare_space(model, calibration, alternatives={"layer1.0": (baseline, recipe)})
    genotype = _genotype(recipe)
    workflow = compile_genotype(genotype, space)
    legacy = compile_genotype(_genotype(baseline), space)
    assert workflow.structural_workflow == legacy.structural_workflow
    assert workflow.fingerprint != legacy.fingerprint
    assert workflow.deferred_ptq[0].ptq_placement == policy
    encoded = genotype_payload(genotype)
    assert genotype_from_payload(json.loads(json.dumps(encoded)), space) == genotype
    assert "ptq_placement" not in genotype_payload(legacy.genotype)["genes"][0]["recipe"]
    broken = deepcopy(encoded)
    broken["genes"][0]["recipe"]["ptq_placement"] = None
    assert type(genotype_from_payload(broken, space)) is ValidationFailure

    built = execute_float_workflow(model, calibration, workflow)
    assert type(built) is CVWorkflowSuccess
    original_quantize = quantization.quantize_static
    captured = {}

    def capture_standard_export(model_input, model_output, reader, **kwargs):
        captured["folded_sha"] = hashlib.sha256(Path(model_input).read_bytes()).hexdigest()
        original_quantize(model_input, model_output, reader, **kwargs)
        captured["standard_sha"] = hashlib.sha256(Path(model_output).read_bytes()).hexdigest()
        captured["standard_graph"] = onnx.load(model_output)

    monkeypatch.setattr(quantization, "quantize_static", capture_standard_export)
    artifact = export_onnx_qdq(built.model, calibration, workflow, tmp_path / "deployment",
                               spec=BackendSpec(max_batch_size=2, opt_batch_size=2, calibration_batch_size=4))
    assert type(artifact) is DeploymentArtifact, artifact
    manifest = artifact.evidence
    assert manifest["schema"] == "heterogeneous_deployment_v2"
    placement = manifest["ptq_placement"]
    assert placement["policy"] == {"omit_output_qdq": [logical]}
    lineage = placement["calibration_lineage"]
    assert lineage["folded_fp32_onnx_sha256"] == captured["folded_sha"]
    assert lineage["standard_qdq_onnx_sha256"] == captured["standard_sha"]
    assert lineage["calibration_fingerprint"] == space.structural_space.calibration.fingerprint
    assert lineage["effective_calibration_batch_size"] == lineage["calibration_rows"] == 2
    assert lineage["onnxruntime_version"] == ort.__version__
    assert lineage["range_use"] == "omitted_output_ranges_unused_no_recalibration"

    boundary, = placement["resolved_boundaries"]
    assert boundary["logical_path"] == logical and boundary["physical_path"] == physical
    assert boundary["consumer_op"] == consumer_op
    assert boundary["conv_node"] == manifest["selected_conv_nodes"][physical]
    standard = captured["standard_graph"]
    final = onnx.load(artifact.model_path)
    before = {node.name: node for node in standard.graph.node}
    after = {node.name: node for node in final.graph.node}
    quantize, dequantize = (before[name] for name in boundary["removed_qdq_nodes"])
    assert (quantize.op_type, dequantize.op_type) == ("QuantizeLinear", "DequantizeLinear")
    assert quantize.input[0] == boundary["conv_output"] and dequantize.input[0] == quantize.output[0]
    assert set(before) - set(after) == {quantize.name, dequantize.name}
    consumer = before[boundary["consumer_node"]]
    slot = boundary["consumer_input_index"]
    assert consumer.input[slot] == dequantize.output[0]
    consumer.input[slot] = boundary["conv_output"]
    assert set(after).issubset(before)
    assert all(node.SerializeToString() == before[name].SerializeToString() for name, node in after.items())
    original_constants = {tensor.name: tensor.SerializeToString() for tensor in standard.graph.initializer}
    final_constants = {tensor.name: tensor.SerializeToString() for tensor in final.graph.initializer}
    assert set(original_constants) - set(final_constants) == set(quantize.input[1:])
    assert all(original_constants[name] == value for name, value in final_constants.items())
    logits = artifact.load()(calibration.numpy())
    assert logits.shape == (2, 10) and np.isfinite(logits).all()

    manifest_path = Path(artifact.manifest_path)
    altered = deepcopy(manifest)
    altered["backend_spec"]["calibration_batch_size"] += 1
    manifest_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(BackendContractError, match="fingerprint"):
        artifact.load()
    altered = deepcopy(manifest)
    altered["schema"] = "heterogeneous_deployment_v1"
    altered.pop("ptq_placement")
    altered.pop("ptq_placement_fingerprint")
    manifest_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(BackendContractError, match="requires a placement manifest"):
        artifact.load()

    # Even a fresh file checksum cannot admit a graph with a shared omitted edge.
    final.graph.node.append(onnx.helper.make_node(
        "Identity", [boundary["conv_output"]], ["extra_output"], name="extra_consumer"))
    onnx.checker.check_model(final)
    onnx.save(final, artifact.model_path)
    altered = deepcopy(manifest)
    altered["model_sha256"] = hashlib.sha256(Path(artifact.model_path).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(BackendContractError, match="one internal consumer input"):
        artifact.load()


@pytest.mark.parametrize("alias", ["conv_output", "quantized", "dequantized", "repeated_consumer_slot"])
def test_proof_rejects_shared_output_edges_before_editing(alias):
    onnx = pytest.importorskip("onnx")
    from fedcore.algorithm.heterogeneous.qdq_placement import plan_output_placement, remove_output_pairs

    helper = onnx.helper
    path = "layer1.0.conv2"
    selected = {path: "conv"}
    conv = helper.make_node("Conv", ["input", "weight", "bias"], ["conv_output"], name="conv")
    consumer = helper.make_node("Add", ["conv_output", "residual"], ["block_output"], name="residual_add")
    float_graph = helper.make_model(helper.make_graph([conv, consumer], "float", [], []))
    plan = plan_output_placement(float_graph, selected, PTQPlacement((path,)))
    quantize = helper.make_node("QuantizeLinear", ["conv_output", "scale", "zero"], ["quantized"], name="Q")
    dequantize = helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["dequantized"], name="DQ")
    consumer.input[0] = "dequantized"
    graph = helper.make_model(helper.make_graph([conv, quantize, dequantize, consumer], "standard", [], [], [
        helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [.25]),
        helper.make_tensor("zero", onnx.TensorProto.INT8, [], [0]),
    ]))
    if alias == "repeated_consumer_slot":
        graph.graph.node[-1].input[1] = "dequantized"
    else:
        graph.graph.output.add().name = alias
    before = graph.SerializeToString()
    with pytest.raises(BackendContractError, match="one internal consumer input"):
        remove_output_pairs(graph, plan)
    assert graph.SerializeToString() == before


def test_placement_export_rejects_tensorrt_before_cuda(tmp_path, monkeypatch):
    from fedcore.experiments import heterogeneous_device

    model, calibration = _source()
    recipe = BlockRecipe(ptq=True, ptq_placement=PTQPlacement(("layer1.0.conv1",)))
    space = prepare_space(model, calibration, alternatives={"layer1.0": (recipe,)})
    workflow = compile_genotype(_genotype(recipe), space)
    monkeypatch.setattr(heterogeneous_device, "require_cuda_binding",
                        lambda *_: pytest.fail("explicit placement reached CUDA binding"))
    destination = tmp_path / "unsupported"
    failure = export_onnx_qdq(model, calibration, workflow, destination, spec=BackendSpec(runtime="tensorrt"))
    assert failure.violations[0].code == "UnsupportedPTQPlacementBackend"
    assert not destination.exists()
