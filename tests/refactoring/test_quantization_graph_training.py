from copy import deepcopy
import pytest
import torch
from torch import nn
from fedcore.algorithm.quantization.training import *
from fedcore.algorithm.quantization.resources import *
from fedcore.algorithm.quantization.portable_layers import PortableQuantizedFactorLinear
from fedcore.algorithm.quantization.tensor_formats import QuantizedTensorSpec


@pytest.mark.parametrize('mode',['static','qat'])
def test_regional_base_quantizer_conversion_real_updates_backend_storage_and_load(mode):
    torch.manual_seed(41)
    model=nn.Sequential(nn.Linear(4,5),nn.ReLU(),nn.Linear(5,2))
    before=deepcopy(model.state_dict())
    x=torch.randn(8,4);target=torch.randn(8,2)
    result=execute_backend_stage(model,BackendStageSpec(mode,('0',),training_steps=2),
        calibration_batches=((x,target),),training_batches=((x,target),))
    assert isinstance(result,ConvertedInference),result
    assert result.model(x).shape==(8,2)
    assert result.storage_bytes==sum(v.untyped_storage().nbytes() for v in result.model.state_dict().values())
    for name,value in before.items():assert torch.equal(model.state_dict()[name],value)
    if mode=='qat':
        assert result.evidence['training']['training_steps']==2
        assert result.evidence['training']['parameter_delta_squared']>0
        assert result.evidence['training']['changed_parameters']
        # A separate float optimizer is an actual control, not a second QAT copy.
        control=deepcopy(model);optimizer=torch.optim.SGD(control.parameters(),lr=.01)
        initial=control(x).detach().clone()
        for _ in range(2):
            optimizer.zero_grad();loss=(control(x)-target).square().mean();loss.backward();optimizer.step()
        assert not torch.equal(initial,control(x))
    assert type(result.model.operators['operator_1']) is nn.Linear or any(type(m) is nn.Linear for m in result.model.modules())
    repeated=execute_backend_stage(result.model,BackendStageSpec('qat',('operators.operator_0',)),
        calibration_batches=((x,target),),training_batches=((x,target),))
    assert isinstance(repeated,StructuralInvalid)


def test_qat_training_budget_failure_charged():
    model=nn.Sequential(nn.Linear(3,2))
    x,y=torch.randn(4,3),torch.randn(4,2)
    ledger=ResourceLedger(StageBudgets(max_training_steps=1))
    result=execute_backend_stage(model,BackendStageSpec('qat',('0',),training_steps=3),
        calibration_batches=((x,y),),training_batches=((x,y),),ledger=ledger)
    assert isinstance(result,ResourceFailure),result
    assert ledger.training_steps==2
    assert ledger.events[-1]['status']=='refused'


def test_existing_factor_recovery_then_encode_ready_factors():
    torch.manual_seed(4)
    layer=PortableQuantizedFactorLinear.from_factors(torch.randn(4,2,dtype=torch.float64),torch.randn(2,3,dtype=torch.float64),
        left_spec=QuantizedTensorSpec(8),right_spec=QuantizedTensorSpec(8))
    model=nn.Sequential(layer)
    x=torch.randn(10,3,dtype=torch.float64);target=torch.randn(10,4,dtype=torch.float64)
    before={key:value.clone() for key,value in model.state_dict().items()}
    result=recover_quantized_factors(model,('0',),training_batches=((x,target),),steps=2,learning_rate=.1)
    assert isinstance(result,ConvertedInference),result
    assert all(stage['changed_parameters'] for stage in result.evidence['stages'])
    assert not torch.equal(result.model(x),model(x))
    for key,value in before.items():assert torch.equal(model.state_dict()[key],value)
