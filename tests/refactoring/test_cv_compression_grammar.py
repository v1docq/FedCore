"""CV boundary, independent numerical reference and metadata grammar laws."""
from dataclasses import replace
import io
import json

import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost, inventory_fingerprint
from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion, RuntimeOptions
from fedcore.algorithm.low_rank.cv_grammar import (
    CVBlockDomain, CVGenotype, CVGrammarSpace, CVScopeGene, ChannelPrune,
    ChannelPruneSVD, ConvSVD, DenseCV, compile_cv_genotype, crossover_cv_genotypes,
    cv_genotype_from_payload, cv_genotype_payload, generate_cv_genotype,
    mutate_cv_genotype, normalize_cv_genotype, validate_cv_workflow)
from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess, execute_cv_workflow, prepare_cv_grammar_space
from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.algorithm.pruning.sparse.cv_channels import (
    CifarResNet18, make_cifar_resnet18, select_internal_channels, validate_cv_model)
from fedcore.tools.registry.checkpoint_manager import CheckpointError, CheckpointManager


CHOICES = (DenseCV(), ChannelPrune(48), ChannelPrune(32), ConvSVD(32), ConvSVD(16),
           ChannelPruneSVD(48, 32), ChannelPruneSVD(48, 16))


def metadata_space():
    return CVGrammarSpace(ModelVersion("cv-test-v1", "a" * 64), CalibrationRef("b" * 64, (1, 3, 32, 32), "float32"),
        tuple(CVBlockDomain(path, CHOICES) for path in ("layer1.0", "layer1.1")), "c" * 64, 44_900_000)


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**20), st.sampled_from(CHOICES))
def test_cv_complete_variation_transport_and_source_binding_laws(seed, choice):
    space = metadata_space()
    genotype = generate_cv_genotype(space, seed=seed)
    assert normalize_cv_genotype(normalize_cv_genotype(genotype, space), space) == genotype
    payload = json.loads(json.dumps(cv_genotype_payload(genotype)))
    assert cv_genotype_from_payload(payload, space) == genotype
    changed = mutate_cv_genotype(genotype, space, seed=seed)
    assert set(gene.path for gene in changed.genes) == set(gene.path for gene in genotype.genes)
    assert sum(a != b for a, b in zip(genotype.genes, changed.genes)) == 1
    first = CVGenotype(tuple(CVScopeGene(domain.path, choice) for domain in space.domains))
    for child in crossover_cv_genotypes(first, changed, space, seed=seed):
        workflow = compile_cv_genotype(child, space)
        assert validate_cv_workflow(workflow) is None
        assert workflow.expected_storage_bytes <= space.source_storage_bytes
        rebound = compile_cv_genotype(child, replace(space, calibration=replace(space.calibration, fingerprint="d" * 64)))
        assert rebound.fingerprint != workflow.fingerprint
        assert rebound.external_artifacts != workflow.external_artifacts


def test_cv_compiler_rejects_forged_nodes_rank_and_insufficient_workspace():
    space = metadata_space()
    genotype = CVGenotype(tuple(CVScopeGene(domain.path, ChannelPruneSVD(48, 16)) for domain in space.domains))
    workflow = compile_cv_genotype(genotype, space)
    assert validate_cv_workflow(replace(workflow, nodes=workflow.nodes[:-1])) is not None
    assert isinstance(compile_cv_genotype(genotype, replace(space, options=RuntimeOptions(max_workspace_bytes=1))), ValidationFailure)
    bad = CVGenotype((CVScopeGene("layer1.0", ConvSVD(True)), CVScopeGene("layer1.1", DenseCV())))
    assert isinstance(compile_cv_genotype(bad, space), ValidationFailure)
    assert isinstance(compile_cv_genotype(genotype, replace(space, options=RuntimeOptions(tensor_byte_budget=1))), ValidationFailure)
    payload = cv_genotype_payload(genotype)
    payload["genes"][0]["choice"]["unknown"] = 1
    assert isinstance(cv_genotype_from_payload(payload, space), ValidationFailure)


@pytest.mark.parametrize("choice", [ConvSVD(None), ConvSVD(True), ConvSVD(0),
                                   ChannelPruneSVD(32, None), ChannelPruneSVD(32, True)])
def test_svd_branch_requires_a_real_positive_integer_rank(choice):
    space = metadata_space()
    assert isinstance(compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),
        CVScopeGene("layer1.1", DenseCV()))), space), ValidationFailure)
    invalid_space = replace(space, domains=(CVBlockDomain("layer1.0", (choice,)),))
    assert isinstance(compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), invalid_space), ValidationFailure)


def test_cv_adapter_uses_actual_fedcore_evolution_and_spea2_without_model_effects(source, calibration):
    from fedcore.algorithm.low_rank.grammar_search import GrammarFitness, GrammarSearchConfig, run_cv_grammar_search
    space = prepare_cv_grammar_space(source, calibration, alternatives={"layer1.0": CHOICES, "layer1.1": CHOICES})
    assert not isinstance(space, ValidationFailure), space
    initial = tuple(CVGenotype(tuple(CVScopeGene(domain.path, choice) for domain in space.domains))
                    for choice in (DenseCV(), ChannelPrune(48), ConvSVD(32), ChannelPruneSVD(48, 32)))
    evaluated = []
    def evaluate(workflow):
        assert validate_cv_workflow(workflow) is None
        assert normalize_cv_genotype(workflow.genotype, space) == workflow.genotype
        evaluated.append(workflow.fingerprint)
        quality = sum((64 - block.internal_width) / 64 + (0 if block.rank is None else (64 - block.rank) / 128)
                      for block in workflow.blocks)
        return GrammarFitness(quality, workflow.expected_storage_bytes / space.source_storage_bytes)
    result = run_cv_grammar_search(space, GrammarSearchConfig(seed=3, population_size=4, generations=3,
        max_evaluations=8, max_proposals=40, max_seconds=30), evaluate,
        fitness_context="synthetic CV graph integration; no accuracy or search-utility claim", initial_genotypes=initial)
    assert result["engine"] == "FedcoreEvoOptimizer"
    assert result["strict_deadline"] is True
    assert 4 <= result["charged_evaluations"] <= 8
    assert result["created_proposals"] <= 40
    assert result["late_evaluation_count"] == 0
    assert len(set(evaluated)) == len(evaluated)
    assert result["finite_archive"]
    assert any(event["event"] == "variation" for event in result["trace"])
    assert any(event["event"] == "population" and event["selection"] == "SPEA2" for event in result["trace"])
    for event in result["trace"]:
        if event["event"] == "evaluated":
            assert event["completed_within_deadline"] is True
            assert not isinstance(cv_genotype_from_payload(event["genotype"], space), ValidationFailure)
    assert all(not isinstance(cv_genotype_from_payload(item["genotype"], space), ValidationFailure)
               for item in result["finite_archive"])


@pytest.fixture(scope="module")
def source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        return make_cifar_resnet18().eval()


@pytest.fixture(scope="module")
def calibration():
    return torch.linspace(-1, 1, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)


def test_dense_constructor_matches_torchvision_cifar_resnet18(source, calibration):
    from torchvision.models import resnet18
    with torch.random.fork_rng(devices=[]):
        reference = resnet18(weights=None, num_classes=10)
        reference.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        reference.maxpool = nn.Identity()
        reference.load_state_dict(source.state_dict(), strict=True)
        reference.eval()
    with torch.no_grad():
        assert torch.equal(reference(calibration), source(calibration))


@pytest.mark.parametrize("choice", [DenseCV(), ChannelPrune(32), ConvSVD(16), ChannelPruneSVD(32, 16)])
def test_owned_cv_execution_real_shapes_bytes_checkpoint_and_exact_outputs(source, calibration, choice):
    before, rng = inventory_fingerprint(capture_inventory(source)), torch.random.get_rng_state().clone()
    space = prepare_cv_grammar_space(source, calibration, alternatives={"layer1.0": (choice,)})
    assert not isinstance(space, ValidationFailure), space
    workflow = compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), space)
    result = execute_cv_workflow(source, calibration, workflow)
    assert type(result) is CVWorkflowSuccess, result
    assert inventory_fingerprint(capture_inventory(source)) == before
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result.model is not source
    actual = inventory_cost(capture_inventory(result.model), "bytes")
    assert actual == workflow.expected_storage_bytes == result.evidence["storage_bytes"]
    assert actual < space.source_storage_bytes if type(choice) is not DenseCV else actual == space.source_storage_bytes
    block = result.model.layer1[0]
    width = choice.keep_channels if type(choice) in (ChannelPrune, ChannelPruneSVD) else 64
    assert block.conv1.weight.shape == (width, 64, 3, 3)
    assert block.bn1.running_var.shape == (width,)
    assert block.bn2.num_features == 64
    if type(choice) in (ConvSVD, ChannelPruneSVD):
        assert type(block.conv2) is nn.Sequential
        assert block.conv2[0].weight.shape == (choice.rank, width, 3, 3)
        assert block.conv2[1].weight.shape == (64, choice.rank, 1, 1)
    else:
        assert block.conv2.weight.shape == (64, width, 3, 3)
    restored = CheckpointManager(".", auto_cleanup=False).deserialize_from_bytes(result.checkpoint_bytes)
    assert type(restored) is CifarResNet18
    assert restored.representation_config() == result.model.representation_config()
    assert inventory_fingerprint(capture_inventory(restored)) == inventory_fingerprint(capture_inventory(result.model))
    with torch.no_grad():
        assert torch.equal(restored(calibration), result.model(calibration))
    source_storage = {value.untyped_storage()._cdata for value in source.state_dict().values()}
    assert not source_storage.intersection(value.untyped_storage()._cdata for value in result.model.state_dict().values())


def test_linked_channel_selection_matches_an_independent_conv_bn_reference(source, calibration):
    choice = ChannelPrune(32)
    space = prepare_cv_grammar_space(source, calibration, alternatives={"layer1.0": (choice,)})
    result = execute_cv_workflow(source, calibration, compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), space))
    assert type(result) is CVWorkflowSuccess, result
    original = source.layer1[0]
    # An independent masked activation reference retains original tensor shapes
    # only for the oracle; the deployed artifact must physically remove them.
    scores = original.conv1.weight.detach().abs().double().sum((1, 2, 3))
    selected = sorted(sorted(range(64), key=lambda index: (-float(scores[index]), index))[:32])
    assert tuple(selected) == select_internal_channels(original, 32)
    probe = calibration.repeat(1, 22, 1, 1)[:, :64].contiguous()
    with torch.no_grad():
        intermediate = original.relu(original.bn1(original.conv1(probe)))
        mask = torch.zeros(64)
        mask[selected] = 1
        reference = original.relu(original.bn2(original.conv2(intermediate * mask.reshape(1, 64, 1, 1))) + probe)
        output = result.model.layer1[0](probe)
    # Different convolution reduction widths may change float32 rounding.
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)


def test_cv_factor_operator_matches_independent_truncated_svd(source, calibration):
    choice = ChannelPruneSVD(32, 16)
    space = prepare_cv_grammar_space(source, calibration, alternatives={"layer1.0": (choice,)})
    result = execute_cv_workflow(source, calibration, compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), space))
    assert type(result) is CVWorkflowSuccess, result
    keep = result.model.channel_indices[0]
    weight = source.layer1[0].conv2.weight.detach()[:, keep].flatten(1).double()
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    reference = (u[:, :16] * singular[:16]) @ vh[:16]
    factors = result.model.layer1[0].conv2
    deployed = factors[1].weight.detach().flatten(1).double() @ factors[0].weight.detach().flatten(1).double()
    torch.testing.assert_close(deployed, reference, atol=2e-7, rtol=2e-5)


def test_stale_model_and_data_are_refused_before_copy_or_solver(source, calibration, monkeypatch):
    from fedcore.algorithm.low_rank import cv_execution
    choice = ChannelPruneSVD(32, 16)
    space = prepare_cv_grammar_space(source, calibration, alternatives={"layer1.0": (choice,)})
    workflow = compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), space)
    def forbidden(*args, **kwargs):
        pytest.fail("No copy or SVD is allowed after a stale/forged request")
    monkeypatch.setattr(cv_execution, "clone_preserving_storages", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    assert type(execute_cv_workflow(source, calibration + 1, workflow)) is StructuralInvalid
    assert type(execute_cv_workflow(source, calibration, replace(workflow, nodes=()))) is StructuralInvalid
    source.layer1[0].bn1.running_mean[0] += 1
    try:
        assert type(execute_cv_workflow(source, calibration, workflow)) is StructuralInvalid
    finally:
        source.layer1[0].bn1.running_mean[0] -= 1


@pytest.mark.parametrize("kind", ["hook", "unknown_child", "sharing", "hidden_tensor", "view"])
def test_cv_guards_reject_unsupported_capabilities_before_source_copy(source, calibration, monkeypatch, kind):
    from copy import deepcopy
    from fedcore.algorithm.low_rank import cv_execution
    model = deepcopy(source)
    if kind == "hook":
        model.layer1[0].register_forward_pre_hook(lambda *args: pytest.fail("hook ran"))
    elif kind == "unknown_child":
        model.layer1[0].unexpected = nn.Identity()
    elif kind == "sharing":
        model.layer1[1].conv1.weight = model.layer1[0].conv1.weight
    elif kind == "hidden_tensor":
        model.hidden = torch.ones(1)
    else:
        tensor = torch.empty(model.conv1.weight.numel() + 1)
        model.conv1.weight = nn.Parameter(tensor[:-1].reshape_as(model.conv1.weight))
    monkeypatch.setattr(cv_execution, "clone_preserving_storages", lambda *args, **kwargs: pytest.fail("copy ran"))
    assert isinstance(prepare_cv_grammar_space(model, calibration, alternatives={"layer1.0": (ChannelPrune(32),)}), ValidationFailure)


def test_cv_checkpoint_rejects_wrong_dtype_shapes_and_aliases(source):
    manager = CheckpointManager(".", auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(source)), weights_only=True)
    payload["state_dict"]["bn1.running_mean"] = payload["state_dict"]["bn1.running_var"]
    with pytest.raises(CheckpointError):
        manager.restore(payload)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(source)), weights_only=True)
    payload["state_dict"]["conv1.weight"] = payload["state_dict"]["conv1.weight"].double()
    with pytest.raises(CheckpointError):
        manager.restore(payload)


def test_low_rank_public_cv_boundary_publishes_success_only(source, calibration, tmp_path, monkeypatch):
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.tools.registry.model_registry import ModelRegistry
    monkeypatch.setenv("FEDCORE_MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    monkeypatch.setattr(ModelRegistry, "_instance", None)
    monkeypatch.setattr(ModelRegistry, "_initialized", False)
    facade = LowRankModel({"device": "cpu", "fedcore_id": "cv-public-boundary-test"})
    choice = ChannelPrune(32)
    space = facade.prepare_cv_grammar(source, calibration, alternatives={"layer1.0": (choice,)})
    workflow = compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), space)
    result = facade.compress_cv_workflow(source, calibration, workflow)
    assert type(result) is CVWorkflowSuccess, result
    assert facade.model_before is source and facade.model_after is result.model
    assert facade.transform_result is result
    failure = facade.compress_cv_workflow(source, calibration + 1, workflow)
    assert type(failure) is StructuralInvalid
    assert facade.model_after is result.model


@pytest.mark.parametrize("kind", ["namespace", "opaque", "frozenset_tensor", "hidden_module_list",
    "parameter_subclass", "buffer_subclass", "module_registry_subclass", "parameter_registry_subclass",
    "parameter_deepcopy_override", "buffer_deepcopy_override", "parameter_hook", "instance_dict_subclass"])
def test_cv_rejects_opaque_instances_and_tensor_subclasses_without_effects(source, calibration, monkeypatch, kind):
    from copy import deepcopy
    from types import SimpleNamespace
    from fedcore.algorithm.low_rank import cv_execution
    model = deepcopy(source)
    def forbidden(*args, **kwargs):
        pytest.fail("A rejected object executed a tensor, iterator, copy or forward effect")
    class Opaque:
        __deepcopy__ = forbidden
        __iter__ = forbidden
    class ParameterSubclass(nn.Parameter):
        __deepcopy__ = forbidden
    class TensorSubclass(torch.Tensor):
        @classmethod
        def __torch_function__(cls, *args, **kwargs):
            return forbidden()
        __deepcopy__ = forbidden
    class RegistrySubclass(dict):
        __iter__ = forbidden
        items = forbidden
        values = forbidden
    if kind == "namespace":
        model.hidden = SimpleNamespace(tensor=torch.ones(1))
    elif kind == "opaque":
        model.hidden = Opaque()
    elif kind == "frozenset_tensor":
        model.hidden = frozenset([torch.ones(1)])
    elif kind == "hidden_module_list":
        model.hidden = [nn.Identity()]
    elif kind == "parameter_subclass":
        model.conv1._parameters["weight"] = ParameterSubclass(model.conv1.weight.detach())
    elif kind == "buffer_subclass":
        model.bn1._buffers["running_mean"] = torch.Tensor._make_subclass(TensorSubclass, torch.ones(64), False)
    elif kind == "module_registry_subclass":
        model._modules = RegistrySubclass(model._modules)
    elif kind == "parameter_registry_subclass":
        model._parameters = RegistrySubclass(model._parameters)
    elif kind == "parameter_deepcopy_override":
        model.conv1.weight.__deepcopy__ = forbidden
    elif kind == "buffer_deepcopy_override":
        model.bn1.running_mean.__deepcopy__ = forbidden
    elif kind == "instance_dict_subclass":
        model.__dict__ = RegistrySubclass(model.__dict__)
    else:
        model.conv1.weight.register_hook(forbidden)
    monkeypatch.setattr(cv_execution, "clone_preserving_storages", forbidden)
    result = prepare_cv_grammar_space(model, calibration, alternatives={"layer1.0": (ChannelPrune(32),)})
    assert isinstance(result, ValidationFailure)


@pytest.mark.parametrize("scope", ["layer1", "layer1.0.conv2", "layer2.0.downsample"])
def test_cv_sequential_execution_order_is_checked_before_copy_or_checkpoint(source, calibration, monkeypatch, scope):
    from copy import deepcopy
    from fedcore.algorithm.low_rank import cv_execution
    if scope == "layer1.0.conv2":
        choice = ConvSVD(16)
        space = prepare_cv_grammar_space(source, calibration, alternatives={"layer1.0": (choice,)})
        result = execute_cv_workflow(source, calibration, compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", choice),)), space))
        assert type(result) is CVWorkflowSuccess, result
        model = result.model
    else:
        model = deepcopy(source)
    sequence = model.get_submodule(scope)
    sequence._modules = dict(reversed(tuple(sequence._modules.items())))
    monkeypatch.setattr(cv_execution, "clone_preserving_storages", lambda *args, **kwargs: pytest.fail("copy ran"))
    assert isinstance(prepare_cv_grammar_space(model, calibration, alternatives={"layer1.0": (ChannelPrune(32),)}), ValidationFailure)
    with pytest.raises(CheckpointError):
        CheckpointManager(".", auto_cleanup=False).serialize_to_bytes(model)


@pytest.mark.parametrize("method", ["describe", "serialize", "restore"])
@pytest.mark.parametrize("kind", ["parameter_override", "buffer_override", "tensor_subclass", "nested"])
def test_public_checkpoint_checks_cv_before_tensor_or_copy_effects(source, monkeypatch, method, kind):
    from copy import deepcopy
    from fedcore.tools.registry import checkpoint_manager
    model = deepcopy(source)
    manager = CheckpointManager(".", auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(source)), weights_only=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected CV checkpoint executed state, topology, tensor or copy effects")
    if kind == "parameter_override":
        model.conv1.weight.__deepcopy__ = forbidden
    elif kind == "buffer_override":
        model.bn1.running_mean.__deepcopy__ = forbidden
    elif kind == "tensor_subclass":
        class TensorSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                return forbidden()
        model.bn1._buffers["running_mean"] = torch.Tensor._make_subclass(TensorSubclass, torch.ones(64), False)
    else:
        model = nn.Sequential(model)
    monkeypatch.setattr(checkpoint_manager, "_validate_remapped_state", forbidden)
    from fedcore.algorithm.low_rank import topology
    monkeypatch.setattr(topology, "inspect_topology", forbidden)
    with pytest.raises(CheckpointError):
        if method == "describe":
            checkpoint_manager.describe_model(model)
        elif method == "serialize":
            manager.serialize_to_bytes(model)
        else:
            manager.restore(payload, model=model)


def test_nested_cv_constructor_payload_is_explicitly_refused_before_allocation(source, monkeypatch):
    from fedcore.tools.registry.checkpoint_manager import build_model, describe_model
    from fedcore.algorithm.pruning.sparse import cv_checkpoint
    nested = {"type": "Sequential", "children": {"0": describe_model(source)}}
    monkeypatch.setattr(cv_checkpoint, "build_cv", lambda *args: pytest.fail("Nested CV constructor allocated"))
    with pytest.raises(CheckpointError):
        build_model(nested)


def test_removed_tensor_hooks_leave_safe_empty_lifecycle_state(source, calibration):
    from copy import deepcopy
    from collections import OrderedDict
    model = deepcopy(source)
    parameter = model.conv1.weight
    backward = parameter.register_hook(lambda gradient: gradient)
    accumulation = parameter.register_post_accumulate_grad_hook(lambda _parameter: None)
    backward.remove()
    accumulation.remove()
    # Tensor deserialization and a removed hook may retain these exact empty
    # registries. They carry no callback and must remain copy/export compatible.
    model.bn1.running_mean._backward_hooks = OrderedDict()
    assert validate_cv_model(model, cpu=True) > 0
    space = prepare_cv_grammar_space(model, calibration, alternatives={"layer1.0": (ChannelPrune(32),)})
    assert not isinstance(space, ValidationFailure), space
    result = execute_cv_workflow(model, calibration, compile_cv_genotype(CVGenotype((CVScopeGene("layer1.0", ChannelPrune(32)),)), space))
    assert type(result) is CVWorkflowSuccess, result
    manager = CheckpointManager(".", auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(result.model))
    assert restored.representation_config() == result.model.representation_config()


def test_opaque_tensor_hook_registry_is_refused_without_user_length_or_iteration(source, calibration):
    from copy import deepcopy
    model = deepcopy(source)
    class OpaqueHooks(dict):
        def __len__(self):
            pytest.fail("Opaque hook registry executed __len__")
        def __iter__(self):
            pytest.fail("Opaque hook registry executed __iter__")
    model.conv1.weight._backward_hooks = OpaqueHooks()
    assert isinstance(prepare_cv_grammar_space(model, calibration, alternatives={"layer1.0": (ChannelPrune(32),)}), ValidationFailure)
