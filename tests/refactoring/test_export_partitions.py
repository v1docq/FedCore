import copy
import torch
from torch import nn
import pytest
from fedcore.external_runtime.contracts import ContractError, DeviceProfile
from model_exporter.model_analyzer import ModelAnalyzer
from model_exporter.model_splitter import ModelSplitter
from model_exporter.model_logic import ModelManager


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3, bias=False)
        self.fc.weight.data.copy_(torch.eye(3))

    def forward(self, x):
        return self.fc(x) + x


class Repeated(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(self.fc(x))


@pytest.mark.parametrize("support", [[], ["Gemm"], ["Relu"], ["Gemm", "Relu"]])
def test_device_assignment_uses_actual_capabilities(support):
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    profile = DeviceProfile(supported_ops=tuple(support))
    x = torch.randn(2, 3)
    splitter = ModelSplitter(profile)
    info = splitter.get_parts_info(model, x)
    flags = [layer["supported"] for layer in info["model_layers"]]
    assert flags == ["Gemm" in support, "Relu" in support]
    for part in info["parts_info"]:
        assert part["is_npu_part"] == all(flags[part["start_layer"]:part["end_layer"]])
    parts = splitter.split_model(model, info, x)
    actual = x
    for part in parts:
        assert list(actual.shape) == part["layers_info"]["input_shape"]
        actual = part["model"](actual)
        assert list(actual.shape) == part["layers_info"]["output_shape"]
    torch.testing.assert_close(actual, model(x))


def test_residual_and_functional_graphs_are_rejected_before_export(tmp_path):
    model = Residual()
    x = torch.ones(1, 3)
    torch.testing.assert_close(model(x), 2*x)
    with pytest.raises(ContractError) as error:
        ModelSplitter(DeviceProfile(supported_ops=("Gemm", "Add"))).get_parts_info(model, x)
    assert error.value.code == "unsupported_graph"
    assert not list(tmp_path.iterdir())


def test_repeated_layer_execution_and_copy_independence():
    model = Repeated()
    x = torch.randn(2, 3)
    splitter = ModelSplitter(DeviceProfile(supported_ops=("Gemm",)))
    info = splitter.get_parts_info(model, x)
    assert len(info["model_layers"]) == 2
    parts = splitter.split_model(model, info, x)
    torch.testing.assert_close(parts[0]["model"](x), model(x))
    baseline = copy.deepcopy(model.state_dict())
    parts[0]["model"][0].weight.data.zero_()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, baseline[key])


def test_unsupported_overrides_supported_and_tampering_is_rejected():
    splitter = ModelSplitter(DeviceProfile(supported_ops=("Gemm",), unsupported_ops=("Gemm",)))
    model = nn.Linear(3, 2)
    info = splitter.get_parts_info(model, torch.ones(1, 3))
    assert not info["parts_info"][0]["is_npu_part"]
    info["parts_info"][0]["is_npu_part"] = True
    with pytest.raises(ContractError, match="support flags"):
        splitter.split_model(model, info, torch.ones(1, 3))


def test_profile_a_b_a_and_intermediate_export(tmp_path):
    model = nn.Sequential(nn.Conv1d(2, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(24, 2))
    x = torch.randn(1, 2, 8)
    manager = ModelManager()
    a = DeviceProfile("A", ("Conv", "Relu"), cpu_framework="torchscript", npu_framework="torchscript")
    b = DeviceProfile("B", ("Gemm",), cpu_framework="torchscript", npu_framework="torchscript")
    first = manager.analyze_model(model, x, a)
    second = manager.analyze_model(model, x, b)
    again = manager.analyze_model(model, x, a)
    assert first == again
    assert first != second
    result = manager.export_parts(model, tmp_path, example_input=x, profile=a)
    assert "error" not in result
    value = x
    for file in result["exported_files"]:
        value = torch.jit.load(file)(value)
    torch.testing.assert_close(value, model(x))
