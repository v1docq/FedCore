import json
import torch
from torch import nn
import pytest
from fedcore.tools.export import ExportError, export_model, normalize_framework
from model_exporter.fedcore_ops import run_operation


class LambdaLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
        self.identity = lambda x: x

    def forward(self, x):
        return self.identity(self.fc(x))


def test_torchscript_lambda_is_real_loadable_format(tmp_path):
    model = LambdaLinear().train()
    x = torch.randn(1, 8)
    path = export_model(model, "torchscript", tmp_path / "model.pt", x)
    loaded = torch.jit.load(str(path)).eval()
    torch.testing.assert_close(loaded(x), model(x))
    assert model.training  # export does not change the caller's mode
    metadata = json.loads(path.with_suffix(".pt.json").read_text())
    assert metadata["format"] == "torchscript"
    assert metadata["input_spec"]["inputs"][0]["shape"] == [1, 8]


def test_both_compilers_fail_without_pickle_or_success(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional compiler failure")
    monkeypatch.setattr(torch.jit, "script", fail)
    monkeypatch.setattr(torch.jit, "trace", fail)
    with pytest.raises(ExportError) as error:
        export_model(nn.Linear(8, 2), "torchscript", tmp_path / "model.pt", torch.zeros(1, 8))
    assert error.value.code == "compilation_failed"
    assert len(error.value.causes) == 2
    assert not (tmp_path / "model.pt").exists()
    assert not list(tmp_path.glob("*.json"))


def test_wrong_or_missing_input_cannot_be_a_success(tmp_path):
    with pytest.raises(ExportError, match="InputSpec"):
        export_model(nn.Linear(8, 2), "torchscript", tmp_path / "x.pt", torch.zeros(1, 3, 224, 224))
    with pytest.raises(ExportError) as error:
        export_model(nn.Linear(8, 2), "torchscript", tmp_path / "x.pt")
    assert error.value.code == "missing_input_spec"
    assert not (tmp_path / "x.pt").exists()


def test_inplace_export_does_not_modify_the_callers_example(tmp_path):
    model = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(8, 2))
    x = -torch.ones(1, 8)
    before = x.clone()
    path = export_model(model, "torchscript", tmp_path / "inplace.pt", x)
    assert path.exists()
    torch.testing.assert_close(x, before)


@pytest.mark.parametrize("name", ["rknn", "openvino", "tflite", "", None, "export_onnx"])
def test_unknown_backends_are_rejected(name):
    with pytest.raises(ExportError) as error:
        normalize_framework(name)
    assert error.value.code == "unsupported_backend"


@pytest.mark.parametrize("name, expected", [("pt", "torchscript"), ("pytorch", "torchscript"), ("trt", "tensorrt"), ("ONNX", "onnx")])
def test_known_aliases(name, expected):
    assert normalize_framework(name) == expected


def test_export_prefix_cannot_bypass_operation_allowlist(tmp_path):
    with pytest.raises(PermissionError):
        run_operation("export_rknn", tmp_path / "missing.pt")


def test_onnx_declared_loader_and_output(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = nn.Linear(8, 2).eval()
    x = torch.randn(2, 8)
    path = export_model(model, "onnx", tmp_path / "x.pt", x)
    assert path.suffix == ".onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])
    torch.testing.assert_close(actual, model(x))
