import pytest
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from fedcore.algorithm.quantization.hooks import (
    DynamicQuantizationHook, StaticQuantizationHook, QATHook
)
from fedcore.algorithm.quantization.utils import (
    ParentalReassembler, uninplace, get_flattened_qconfig_dict,
    QDQWrapper, QDQWrapping
)
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel


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
    seq = nn.Sequential(nn.Linear(4,4), nn.Linear(4,2))
    from torch.ao.quantization import default_qconfig
    for m in seq: m.qconfig = default_qconfig
    inp = torch.randn(2,4)
    m2 = QDQWrapper.add_quant_entry_exit(deepcopy(seq), inp, allow={nn.Linear}, mode='static')
    assert isinstance(m2[0], QDQWrapping) and isinstance(m2[-1], QDQWrapping)

@pytest.fixture
def simple_dl():
    x = torch.randn(10, 3, 8, 8)
    y = torch.randint(0,2,(10,))
    return DataLoader(TensorDataset(x,y), batch_size=5)

def test_dynamic_quant_hook_does_not_crash(simple_dl):
    hook = DynamicQuantizationHook(1, torch.qint8, set(), -1, "fbgemm")
    hook.model = nn.Linear(3*8*8,2)
    assert hook.trigger(1, dict())
    hook.action(1, {})

def test_static_quant_hook_runs_validation(simple_dl):
    dummy = type('ID', (), {})()
    dummy.features = type('F', (), {})()
    dummy.features.val_dataloader = simple_dl
    hook = StaticQuantizationHook(1, torch.qint8, set(), -1, "fbgemm")
    hookable_trainer = BaseNeuralModel(nn.Conv2d(3,3,3), {}, [])
    hook.link_to_trainer(hookable_trainer)
    assert hook.trigger(1, {})
    hook.action(1, {"val_loader": simple_dl})


def test_qat_hook_train_loop(simple_dl):
    dummy = type('ID', (), {})()
    dummy.features = type('F', (), {})()
    dummy.features.train_dataloader = simple_dl
    params = {
        'input_data': dummy,
        'epochs':1,
        'optimizer': torch.optim.SGD,
        'criterion': nn.CrossEntropyLoss(),
        'lr': 0.01,
        'device': torch.device('cpu')
    }
    hook = QATHook(-1, torch.qint8, set(), 1, "fbgemm")
    hookable_trainer = BaseNeuralModel(nn.Sequential(nn.Flatten(), nn.Linear(3*8*8,2)), {"epoch": 20}, [hook])
    hook.link_to_trainer(hookable_trainer)
    #hookable_trainer.fit(dummy) TODO correct test with BaseQuantizer
    assert hook.trigger(1, {})
    hook.action(1, {})