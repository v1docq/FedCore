import pytest
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from fedcore.algorithm.quantization.utils import (
    uninplace, get_flattened_qconfig_dict,
    QDQWrapper, QDQWrapping
)
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.algorithm.low_rank.reassembly.core_reassemblers import ParentalReassembler
from fedcore.algorithm.quantization.quantizers import BaseQuantizer
from fedcore.data.data import CompressionInputData
from fedot.core.repository.tasks import Task, TaskTypesEnum


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
    from torch.nn.quantized import FloatFunctional
    assert hasattr(wrapped, 'skip_add') and isinstance(wrapped.skip_add, FloatFunctional)

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

def simple_dl():
    x = torch.randn(10, 3, 8, 8)
    y = torch.randint(0, 2, (10,))
    return DataLoader(TensorDataset(x, y), batch_size=5)


def _state_dict_size_mb(model):
    return sum(
        t.numel() * t.element_size()
        for t in model.state_dict().values()
        if torch.is_tensor(t)
    ) / (1 << 20)


def _cifar_like_resnet(num_classes=10):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.eval()


def _compression_input_data(model, dataloader):
    example_input = next(iter(dataloader))[0]
    return CompressionInputData(
        features=example_input,
        model=model,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        test_dataloader=dataloader,
        task=Task(TaskTypesEnum.classification),
        input_dim=example_input.size(-1),
    )


@pytest.mark.parametrize('quant_type', ['dynamic', 'static', 'qat'])
def test_base_quantizer_fit_predict(quant_type):
    """BaseQuantizer quantizes a CIFAR-like ResNet18 as a pipeline node."""
    torch.manual_seed(0)
    model = _cifar_like_resnet()
    dataloader = simple_dl()
    input_data = _compression_input_data(model, dataloader)

    quantizer = BaseQuantizer({'quant_type': quant_type,
                               'allow_emb': False,
                               'allow_conv': True,
                               'qat_params': {'epochs': 1}})
    quant_model = quantizer.fit(input_data)

    assert quant_model is not None
    assert getattr(quant_model, '_is_quantized', False)
    # the original model must stay untouched and float
    assert not getattr(model, '_is_quantized', False)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # quantized models are CPU-only and keep the output shape
    with torch.no_grad():
        out = quant_model(next(iter(dataloader))[0])
    assert out.shape == (5, 10)

    if quant_type in ('static', 'qat'):
        # weights are quantized to int8 -> ~4x smaller footprint
        assert _state_dict_size_mb(quant_model) < 0.5 * _state_dict_size_mb(model)

    prediction = quantizer.predict(input_data, output_mode='fedcore')
    assert prediction is not None
    original_prediction = quantizer.predict(input_data, output_mode='original')
    assert original_prediction is not None


def test_base_quantizer_dynamic_only_quantizes_allowed_modules():
    """allow_conv/allow_emb flags restrict the quantized module set."""
    torch.manual_seed(0)
    model = _cifar_like_resnet()
    dataloader = simple_dl()
    input_data = _compression_input_data(model, dataloader)

    quantizer = BaseQuantizer({'quant_type': 'dynamic', 'allow_conv': True})
    quant_model = quantizer.fit(input_data)
    # dynamic quantization covers Linear layers but not convolutions
    assert isinstance(quant_model.fc, nn.quantized.dynamic.Linear)
