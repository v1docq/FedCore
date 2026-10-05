"""Offline regressions through actual configuration, registry and API imports."""
from copy import deepcopy
from dataclasses import dataclass
from inspect import getattr_static
from pathlib import Path
from typing import Literal, Optional, Union
import importlib
import subprocess
import sys
import os
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedcore.api.api_configs import (
    ConfigTemplate, DeviceConfigTemplate, ComputeConfigTemplate, DistributedConfigTemplate,
    APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate, LearningConfigTemplate,
    TrainingTemplate, MisconfigurationError)
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.main import FedCore
from fedcore.data.data import CompressionInputData
from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError
from fedcore.tools.registry.model_registry import ModelRegistry
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.repository.initializer_industrial_models import (
    FedcoreModels, FEDOT_METHOD_TO_REPLACE)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository


def configuration(model, tmp_path, problem='classification', solver=None):
    metric = 'MulticlassAccuracy__2' if problem == 'classification' else 'MeanSquaredError'
    template = APIConfigTemplate(
        device_config=DeviceConfigTemplate(device='cpu'),
        automl_config=AutoMLConfigTemplate(fedot_config=FedotConfigTemplate(
            problem=problem, initial_assumption=model, metric=[metric])),
        learning_config=LearningConfigTemplate(learning_strategy='checkpoint',
            criterion='cross_entropy' if problem == 'classification' else 'mse',
            peft_strategy_params=TrainingTemplate(epochs=1)),
        compute_config=ComputeConfigTemplate(output_folder=str(tmp_path),
            distributed=DistributedConfigTemplate(threads_per_worker=1)), solver=solver)
    return ConfigFactory.from_template(template)()


def dataset(model, *, shuffle=True, batch_size=6, problem='classification'):
    if problem == 'classification':
        labels = torch.arange(20) % 2
        inputs = torch.nn.functional.one_hot(labels, 2).float()
    else:
        inputs = torch.arange(20).float().view(-1, 1)
        labels = inputs.clone()
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=shuffle,
                        generator=torch.Generator().manual_seed(42))
    return CompressionInputData(model=model, train_dataloader=loader, val_dataloader=loader,
        task=Task(TaskTypesEnum.classification if problem == 'classification' else TaskTypesEnum.regression),
        num_classes=2 if problem == 'classification' else None, input_dim=inputs.shape[-1])


def identity_model(problem='classification'):
    size = 2 if problem == 'classification' else 1
    model = nn.Linear(size, size, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(size))
    return model


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    # Registry is production singleton; isolate its persistent state for every test.
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH', str(tmp_path))
    monkeypatch.setattr(ModelRegistry, '_instance', None)
    monkeypatch.setattr(ModelRegistry, '_initialized', False)
    return ModelRegistry(auto_cleanup=False)


def test_config_defaults_and_nested_values_are_independent():
    C = ConfigFactory.from_template(DeviceConfigTemplate())
    assert C(device='cpu').device == 'cpu'
    assert C().device == 'cuda'
    @dataclass
    class NestedTemplate(ConfigTemplate):
        device: Union[Literal['cpu'], int] = 'cpu'
        parameters: dict = None
    source = NestedTemplate(parameters={'nested': [1]})
    N = ConfigFactory.from_template(source)
    first, second = N(), N()
    first.parameters['nested'].append(2)
    assert second.parameters == source[1]['parameters'] == {'nested': [1]}
    assert N(device=4).device == 4
    with pytest.raises(MisconfigurationError, match='device'):
        N(device='invalid')
    with pytest.raises(MisconfigurationError, match='device'):
        C(device='invalid_device')


def test_union_optional_checks_every_supplied_value():
    @dataclass
    class UnionTemplate(ConfigTemplate):
        value: Optional[Union[Literal['cpu'], int]] = None
    C = ConfigFactory.from_template(UnionTemplate())
    assert C().value is None
    assert C(value=3).value == 3
    assert C(value='cpu').value == 'cpu'
    with pytest.raises(MisconfigurationError, match='value'):
        C(value='gpu')
    with pytest.raises(MisconfigurationError, match='value'):
        C(value=True)


def test_output_task_defaults_are_independent():
    from fedcore.data.data import CompressionOutputData
    first, second = CompressionOutputData(), CompressionOutputData()
    assert first.task is not second.task
    first.task.task_type = TaskTypesEnum.regression
    assert second.task.task_type is TaskTypesEnum.classification


def test_invalid_lora_epochs_rejected_before_resources(tmp_path, monkeypatch):
    import distributed
    from fedcore.api.api_configs import LoraTemplate
    calls = []
    monkeypatch.setattr(distributed, 'LocalCluster', lambda **kw: calls.append(kw))
    config = configuration(identity_model(), tmp_path / 'absent')
    Lora = ConfigFactory.from_template(LoraTemplate(epochs=0))
    config.learning_config.peft_strategy_params = Lora()
    with pytest.raises(MisconfigurationError, match='epochs'):
        FedCore(config)
    assert not calls and not (tmp_path / 'absent').exists()


def test_invalid_config_has_no_runtime_or_filesystem_effects(tmp_path, monkeypatch):
    import distributed
    calls = []
    monkeypatch.setattr(distributed, 'LocalCluster', lambda **kw: calls.append(kw))
    model = identity_model()
    config = configuration(model, tmp_path / 'absent')
    # Deliberately bypass assignment validation to verify public constructor boundary.
    object.__setattr__(config.device_config, 'device', 'invalid_device')
    before = {key: getattr_static(*key) for key in FEDOT_METHOD_TO_REPLACE}
    with pytest.raises(MisconfigurationError, match='device'):
        FedCore(config)
    assert not calls and not (tmp_path / 'absent').exists()
    assert all(getattr_static(*key) is original for key, original in before.items())
    config = configuration(model, tmp_path / 'absent')
    config.compute_config.distributed.n_workers = 0
    with pytest.raises(MisconfigurationError, match='n_workers'):
        FedCore(config)
    assert not calls and not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('factory', [lambda: nn.Linear(2, 3),
    lambda: nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3))])
def test_registry_eviction_restores_predictions_dtype_and_mode(factory, isolated_registry):
    model = factory().double().eval()
    model.requires_grad_(False)
    inputs = torch.randn(5, 2, dtype=torch.float64)
    expected = model(inputs).detach().clone()
    operation = BaseCompressionModel({'device': torch.device('cpu')})
    operation.model_before = model
    operation.model_after = deepcopy(model)
    operation.clear_model_cache()
    assert operation._model_before_cached is operation._model_after_cached is None
    for restored in (operation.model_before, operation.model_after):
        assert isinstance(restored, nn.Module) and not restored.training
        assert next(restored.parameters()).dtype == torch.float64
        assert not next(restored.parameters()).requires_grad
        torch.testing.assert_close(restored(inputs), expected)


def test_checkpoint_validation_preserves_supplied_model_and_registry(tmp_path, isolated_registry):
    manager = isolated_registry.checkpoint_manager
    model = nn.Linear(2, 3)
    data = manager.serialize_to_bytes(model)
    payload = torch.load(__import__('io').BytesIO(data), weights_only=True)
    payload['version'] = 987
    with pytest.raises(CheckpointError, match='version'):
        manager.restore(payload)
    target = nn.Linear(7, 3)
    before = deepcopy(target.state_dict())
    path = tmp_path / 'good.pt'
    manager.save_to_file(data, str(path))
    with pytest.raises(CheckpointError, match='shapes'):
        manager.load_from_file(str(path), model=target)
    for key, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[key])
    unknown = tmp_path / 'bad.pt'
    torch.save(payload, unknown)
    count = len(isolated_registry.storage.load('test'))
    with pytest.raises(CheckpointError):
        isolated_registry.register_model('test', model_path=str(unknown))
    assert len(isolated_registry.storage.load('test')) == count
    legacy = tmp_path / 'legacy.pt'
    torch.save(model.state_dict(), legacy)
    with pytest.raises(CheckpointError, match='Architecture'):
        manager.load_from_file(str(legacy))
    restored = manager.load_from_file(str(legacy), model=nn.Linear(2, 3))
    torch.testing.assert_close(restored.weight, model.weight)


@pytest.mark.parametrize('change', [
    {'version': True}, {'training': {'': 'eval'}},
    {'architecture': {'type': 'Sequential'}},
    {'state_dict': {'weight': {'__fedcore_state__': 'quantized_tensor'}}},
])
def test_malformed_checkpoint_metadata_has_domain_error(change, tmp_path):
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    payload = torch.load(__import__('io').BytesIO(manager.serialize_to_bytes(nn.Linear(2, 3))), weights_only=True)
    payload.update(change)
    with pytest.raises(CheckpointError):
        manager.restore(payload)


def test_compression_initialization_keeps_three_independent_models(isolated_registry):
    source = identity_model()
    operation = BaseCompressionModel({'device': 'cpu'})
    operation._init_model_before_model_after(dataset(source))
    assert len({id(source), id(operation.model_before), id(operation.model_after)}) == 3
    before = source.weight.detach().clone()
    with torch.no_grad():
        operation.model_after.weight.add_(1)
    torch.testing.assert_close(source.weight, before)
    torch.testing.assert_close(operation.model_before.weight, before)


def test_replaced_baseline_checkpoint_survives_cache_eviction(isolated_registry):
    operation = BaseCompressionModel({'device': 'cpu'})
    operation.model_before = identity_model()
    replacement = identity_model()
    with torch.no_grad():
        replacement.weight.mul_(2)
    operation.model_before = replacement
    operation.clear_model_cache()
    torch.testing.assert_close(operation.model_before.weight, replacement.weight)


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('batch_size', [3, 6])
@pytest.mark.parametrize('problem', ['classification', 'regression'])
def test_public_predict_and_report_pairing(shuffle, batch_size, problem, tmp_path):
    model = identity_model(problem)
    api = FedCore(configuration(model, tmp_path, problem))
    data = dataset(model, shuffle=shuffle, batch_size=batch_size, problem=problem)
    before_roles = (data.train_dataloader, data.val_dataloader, data.test_dataloader)
    result = api.predict(data)
    expected = result.target if problem == 'regression' else result.target.long()
    prediction = result.predict if problem == 'regression' else result.predict.argmax(-1)
    torch.testing.assert_close(prediction, expected)
    report = api.get_report(data)
    metric = 'MulticlassAccuracy__2' if problem == 'classification' else 'MeanSquaredError'
    expected_value = 1 if problem == 'classification' else 0
    assert report.loc[metric, (0, 'original')] == expected_value
    assert report.loc[metric, (0, 'fedcore')] == expected_value
    assert report.loc[metric, (0, 'change')] == 0
    assert before_roles == (data.train_dataloader, data.val_dataloader, data.test_dataloader)


def test_report_accepts_single_pass_labeled_stream(tmp_path):
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    data = dataset(model)
    data.val_dataloader = iter(list(data.val_dataloader))
    assert api.get_report(data).loc['MulticlassAccuracy__2', (0, 'change')] == 0


def test_save_load_and_checkpoint_input_public_boundary(tmp_path):
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    data = dataset(model)
    expected = api.predict(data).predict.sort(dim=0).values
    api.get_report(data)
    saved = api.save()
    assert all(path.is_file() for path in saved.values())
    assert {'model', 'metrics', 'prediction'} <= set(saved)
    restored_api = FedCore(configuration(model, tmp_path))
    restored_api.load(saved['model'])
    actual = restored_api.predict(data).predict.sort(dim=0).values
    torch.testing.assert_close(actual, expected)
    checkpoint_data = dataset({'path_to_model': str(saved['model']), 'model': nn.Linear(2, 2, bias=False)})
    second = FedCore(configuration(model, tmp_path))
    torch.testing.assert_close(second.predict(checkpoint_data).predict.sort(dim=0).values, expected)
    checkpoint_data.model = {'path_to_model': str(saved['model']), 'model': nn.Linear(8, 2)}
    with pytest.raises(CheckpointError, match='shapes'):
        second.predict(checkpoint_data)


def test_nested_adaptation_restores_exact_originals_after_exception():
    originals = {key: getattr_static(*key) for key in FEDOT_METHOD_TO_REPLACE}
    repositories = deepcopy(OperationTypesRepository.__repository_dict__)
    initialized = dict(OperationTypesRepository.__initialized_repositories__)
    with pytest.raises(RuntimeError, match='sentinel'):
        with FedcoreModels():
            patched = {key: getattr_static(*key) for key in originals}
            with FedcoreModels():
                pass
            assert all(getattr_static(*key) is value for key, value in patched.items())
            raise RuntimeError('sentinel')
    assert all(getattr_static(*key) is value for key, value in originals.items())
    assert OperationTypesRepository.__repository_dict__ == repositories
    assert OperationTypesRepository.__initialized_repositories__ == initialized


class FakeResource:
    def __init__(self, *args, **kwargs):
        self.closed = 0
    def close(self):
        self.closed += 1


class LocalSolver:
    def __init__(self, failure=None):
        self.failure = failure
        self.model_after = None
    def fit(self, data):
        if self.failure:
            raise self.failure('sentinel')
        self.model_after = deepcopy(data.model)
        return self


@pytest.mark.parametrize('method', ['fit', 'fit_no_evo'])
@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt])
def test_fit_resource_lifecycle_all_exits(method, failure, tmp_path, monkeypatch):
    import distributed
    resources = []
    def create(*args, **kwargs):
        resource = FakeResource()
        resources.append(resource)
        return resource
    monkeypatch.setattr(distributed, 'LocalCluster', create)
    monkeypatch.setattr(distributed, 'Client', create)
    originals = {key: getattr_static(*key) for key in FEDOT_METHOD_TO_REPLACE}
    for _ in range(2):
        model = identity_model()
        api = FedCore(configuration(model, tmp_path, solver=LocalSolver(failure)))
        if failure:
            with pytest.raises(failure, match='sentinel'):
                getattr(api, method)(dataset(model))
        else:
            getattr(api, method)(dataset(model))
            api.predict(dataset(model))
            api.get_report(dataset(model))
        api.shutdown()
    assert len(resources) == 4 and all(resource.closed == 1 for resource in resources)
    assert all(getattr_static(*key) is value for key, value in originals.items())


def test_external_dask_client_is_never_closed(tmp_path):
    external = FakeResource()
    model = identity_model()
    api = FedCore(configuration(model, tmp_path, solver=LocalSolver()), dask_client=external)
    api.fit_no_evo(dataset(model))
    api.shutdown()
    assert external.closed == 0


def test_import_api_does_not_patch_fedot_in_clean_process():
    code = """from fedot.core.operations.operation import Operation
original = Operation.fit
from fedcore.api.main import FedCore
assert Operation.fit is original
"""
    subprocess.run([sys.executable, '-c', code], check=True, timeout=120)


def test_real_training_pipeline_public_fit_no_evo(tmp_path, isolated_registry, monkeypatch):
    import distributed
    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    data = dataset(model)
    original = model.weight.detach().clone()
    api.fit_no_evo(data)
    assert isinstance(api.compressed_model, nn.Module)
    result = api.predict(data)
    assert result.predict.shape == (20, 2)
    report = api.get_report(data)
    assert report.loc['MulticlassAccuracy__2', (0, 'original')] == 1
    torch.testing.assert_close(api.original_model.weight, original)
    assert api.save('model')['model'].is_file()


def test_adaptation_serializes_concurrent_scopes():
    from threading import Event, Thread
    entered, attempted, completed = Event(), Event(), Event()
    failures = []
    originals = {key: getattr_static(*key) for key in FEDOT_METHOD_TO_REPLACE}
    def worker():
        attempted.set()
        try:
            with FedcoreModels():
                entered.set()
        except BaseException as error:
            failures.append(error)
        finally:
            completed.set()
    with FedcoreModels():
        thread = Thread(target=worker)
        thread.start()
        assert attempted.wait(2)
        assert not entered.wait(.05)
    assert completed.wait(5)
    thread.join()
    assert not failures and entered.is_set()
    assert all(getattr_static(*key) is value for key, value in originals.items())


def test_quantized_checkpoint_restores_packed_state_and_metadata(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    sample = torch.randn(5, 3)
    expected = quantized(sample)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    path = tmp_path / 'quantized.pt'
    manager.save_to_file(manager.serialize_to_bytes(quantized), str(path))
    restored = manager.load_from_file(str(path), 'cpu')
    assert restored.state_dict()._metadata == quantized.state_dict()._metadata
    torch.testing.assert_close(restored(sample), expected)


def test_pipeline_quality_uses_actual_shuffled_prediction_targets(tmp_path, isolated_registry, monkeypatch):
    import distributed
    from fedcore.metrics.quality import MetricFactory
    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    api.manager.learning_config.peft_strategy_params.epochs = 0
    data = dataset(model, shuffle=True, batch_size=6)
    api.fit_no_evo(data)
    with FedcoreModels():
        metric = MetricFactory.get_metric('MulticlassAccuracy__2')
        assert metric.get_value(api.manager.solver, api._process_input_data(data)) == 1


def test_actual_fedot_fit_predefined_pipeline(tmp_path, isolated_registry, monkeypatch):
    import distributed
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    with FedcoreModels():
        pipeline = PipelineBuilder().add_node('training_model', params={
            'epochs': 0, 'device': 'cpu', 'criterion': 'cross_entropy'}).build()
    fitted = api.fit(dataset(model), manually_done=True, predefined_model=pipeline)
    assert fitted is not None
    assert api.get_report(dataset(model)).loc['MulticlassAccuracy__2', (0, 'change')] == 0


def test_lora_template_public_operation_preserves_original(tmp_path, isolated_registry, monkeypatch):
    import distributed
    from fedcore.api.api_configs import LoraTemplate
    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    model = nn.Sequential(identity_model())
    config = configuration(model, tmp_path)
    Lora = ConfigFactory.from_template(LoraTemplate(epochs=1, rank=1, target_layers=['0']))
    config.learning_config.peft_strategy_params = Lora()
    api = FedCore(config)
    data = dataset(model)
    original = deepcopy(model.state_dict())
    api.fit_no_evo(data)
    result = api.predict(data)
    assert result.predict.shape == (20, 2)
    assert api.manager.solver.root_node.fitted_operation.history
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[key])
    assert api.get_report(data).loc['MulticlassAccuracy__2', (0, 'change')] == 0


def test_scoped_root_validation_keeps_ordinary_fedot_contract():
    from types import SimpleNamespace
    from fedot.core.operations.model import Model
    import fedot.core.pipelines.verification_rules as rules
    ordinary = SimpleNamespace(root_node=SimpleNamespace(operation=Model('logit')))
    compression = SimpleNamespace(root_node=SimpleNamespace(operation=SimpleNamespace(operation_type='training_model')))
    unknown = SimpleNamespace(root_node=SimpleNamespace(operation=SimpleNamespace(operation_type='unknown')))
    original = rules.has_final_operation_as_model
    original(ordinary)
    with pytest.raises(ValueError, match='not a model'):
        original(compression)
    with FedcoreModels():
        rules.has_final_operation_as_model(ordinary)
        rules.has_final_operation_as_model(compression)
        with pytest.raises(ValueError, match='not a model'):
            rules.has_final_operation_as_model(unknown)
    assert rules.has_final_operation_as_model is original
    with pytest.raises(ValueError, match='not a model'):
        rules.has_final_operation_as_model(compression)


def test_same_generated_name_does_not_rebind_existing_validation():
    @dataclass
    class FirstTemplate(ConfigTemplate):
        value: Literal['cpu'] = 'cpu'
    @dataclass
    class SecondTemplate(ConfigTemplate):
        value: int = 1
    first = ConfigFactory.from_template(FirstTemplate(), name='Repeated')
    second = ConfigFactory.from_template(SecondTemplate(), name='Repeated')
    assert first().value == 'cpu'
    assert second().value == 1
    with pytest.raises(MisconfigurationError, match='value'):
        first(value=1)


def test_setitem_accepts_valid_union_changes_and_rejects_invalid():
    @dataclass
    class OptionalTemplate(ConfigTemplate):
        value: Optional[Union[Literal['cpu'], int]] = None
    C = ConfigFactory.from_template(OptionalTemplate())
    config = C()
    config['value'] = 'cpu'
    config['value'] = 2
    config['value'] = None
    with pytest.raises(MisconfigurationError, match='value'):
        config['value'] = 'bad'


def test_actual_evolutionary_fit_with_small_local_search(tmp_path, isolated_registry, monkeypatch):
    import distributed
    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    settings = api.manager.automl_config.fedot_config
    settings.timeout = 0.05
    settings.pop_size = 2
    settings.n_jobs = 1
    settings.available_operations = ['training_model']
    api.manager.learning_config.peft_strategy_params.epochs = 0
    result = api.fit(dataset(model))
    assert result is not None and isinstance(api.compressed_model, nn.Module)
    assert api.get_report(dataset(model)).loc['MulticlassAccuracy__2', (0, 'change')] == 0
