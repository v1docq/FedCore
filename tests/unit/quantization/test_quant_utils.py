import pytest
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from fedcore.algorithm.quantization.utils import (
    uninplace, get_flattened_qconfig_dict,
    QDQWrapper, QDQWrapping
)
from fedcore.algorithm.low_rank.reassembly.core_reassemblers import ParentalReassembler


def test_uninplace_recursively():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.seq = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2,2))
    m = M()
    assert m.relu.inplace and m.seq[0].inplace
    uninplace(m)
    assert not m.relu.inplace and not m.seq[0].inplace

def test_get_flattened_qconfig_dict_defaults():
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.qconfig import default_qconfig
    qm = QConfigMapping().set_global(default_qconfig)
    qm.set_object_type(nn.Conv2d, default_qconfig)
    flat = get_flattened_qconfig_dict(qm)
    assert "" in flat and nn.Conv2d in flat
    assert flat[""] is default_qconfig and flat[nn.Conv2d] is default_qconfig

def test_parental_reassembler_embedding_and_basicblock():
    emb = nn.Embedding(10, 4)
    seq = nn.Sequential(emb)
    model = ParentalReassembler.reassemble(deepcopy(seq))
    out_emb = list(model.children())[0]
    assert isinstance(out_emb, nn.Embedding)
    assert torch.allclose(out_emb.weight, emb.weight)
    from torchvision.models.resnet import BasicBlock
    block = BasicBlock(3, 3)
    seq2 = nn.Sequential(block)
    model2 = ParentalReassembler.reassemble(deepcopy(seq2))
    wrapped = list(model2.children())[0]
    assert type(wrapped) is BasicBlock
    seq2.eval(); model2.eval()
    x = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(model2(x), seq2(x))
    assert wrapped.conv1.weight.data_ptr() != block.conv1.weight.data_ptr()

def test_is_leaf_quantizable_linear():
    lin = nn.Linear(5, 3)
    from torch.ao.quantization import default_dynamic_qconfig
    lin.qconfig = default_dynamic_qconfig
    inp = torch.randn(2,5)
    res = QDQWrapper.is_leaf_quantizable(lin, (inp,), mode='dynamic')
    assert isinstance(res, bool)

def test_add_quant_entry_exit_inserts_wrappers():
    seq = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,2)).eval()
    from torch.ao.quantization import default_qconfig, prepare, convert
    for m in seq: m.qconfig = default_qconfig
    inp = torch.randn(2,4)
    m2 = QDQWrapper.add_quant_entry_exit(deepcopy(seq), inp, allow={nn.Linear}, mode='static')
    assert isinstance(m2[0], QDQWrapping) and isinstance(m2[-1], QDQWrapping)
    prepare(m2, inplace=True)
    m2(inp)
    convert(m2, inplace=True)
    output = m2(inp)
    assert isinstance(m2[0].base, nn.quantized.Linear)
    assert isinstance(m2[-1].base, nn.quantized.Linear)
    assert output.shape == (2, 2)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

@pytest.mark.parametrize("mode", ["dynamic", "static", "qat"])
def test_public_quantizer_completes_actual_conversion(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    from fedcore.algorithm.quantization.quantizers import BaseQuantizer
    from fedcore.tools.registry.model_registry import ModelRegistry
    monkeypatch.setenv("FEDCORE_MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    ModelRegistry._instance = None
    ModelRegistry._initialized = False
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 6), nn.ReLU(), nn.Linear(6, 2))
    loader = DataLoader(TensorDataset(torch.randn(6, 3, 2, 2), torch.tensor([0, 1] * 3)), batch_size=3)
    data = SimpleNamespace(model=model, target=model, train_dataloader=loader, calibration_dataloader=loader)
    state = deepcopy(model.state_dict())
    operation = BaseQuantizer({"quant_type": mode, "qat_params": {"epochs": 1}})
    converted = operation.fit(data)
    assert operation.quantization_result.status == "completed"
    assert operation.quantization_result.training_steps == (2 if mode == "qat" else 0)
    assert any(type(layer).__module__.startswith("torch.ao.nn.quantized") for layer in converted.modules())
    output = converted(next(iter(loader))[0])
    assert output.shape == (3, 2) and torch.isfinite(output).all()
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)
