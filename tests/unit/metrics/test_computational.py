"""CPU computational metrics use real data and make no NVML availability claim."""
import io
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fedcore.tools.ruler import PerformanceEvaluator, MeasurementUnavailable
from fedcore.metrics.quality import MetricFactory


@pytest.fixture
def measured_input():
    model = nn.Linear(5, 3).train()
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(TensorDataset(torch.randn(9, 5, generator=generator), torch.zeros(9)), batch_size=3)
    return model, loader


@pytest.mark.parametrize("name, unit, minimize", [
    ("Latency", "ms/batch", True),
    ("Throughput", "samples/s", False),
    ("ModelSize", "MiB (serialized state)", True),
])
def test_cpu_metric_factory_actual_measurement(measured_input, name, unit, minimize):
    model, loader = measured_input
    state = {k: v.clone() for k, v in model.state_dict().items()}
    metric = MetricFactory.get_metric("CPU" + name)
    value = metric.get_value(model, loader)
    assert value > 0 and torch.isfinite(torch.tensor(value))
    assert metric.unit == unit and metric.need_to_minimize is minimize
    with pytest.raises(NotImplementedError, match="get_value"):
        metric.metric(torch.zeros(3), torch.zeros(3))
    assert model.training
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["Power", "PowerConsumption", "Energy", "EnergyConsumption"])
def test_cpu_energy_has_explicit_unavailable_result(measured_input, name):
    model, loader = measured_input
    with pytest.raises(MeasurementUnavailable, match="NVML"):
        MetricFactory.get_metric("CPU" + name).get_value(model, loader)


def test_measured_batch_limit_and_serialized_size(measured_input):
    model, loader = measured_input
    evaluator = PerformanceEvaluator(model, data=loader, n_batches=1, warmup_batches=0)
    assert len(list(evaluator._generate_example_batch())) == 1
    evaluator.measure_latency()
    assert evaluator.measurement_info["latency"]["measured_batches"] == 1
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    assert evaluator.measure_model_size()[0] == len(buffer.getvalue()) / (1 << 20)
