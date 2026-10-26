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
    assert ledger.actual_training_steps==1
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


def test_mixed_float32_factor_recovery_and_alias_closure():
    torch.manual_seed(18)
    packed=PortableQuantizedFactorLinear.from_factors(torch.randn(4,2),torch.randn(2,4))
    model=nn.Sequential(nn.Linear(4,4),packed,nn.Linear(4,4),packed)
    x,y=torch.randn(6,4),torch.randn(6,4)
    result=recover_quantized_factors(model,('1',),training_batches=((x,y),),steps=1,learning_rate=.02)
    assert isinstance(result,ConvertedInference),result
    assert result.model[1] is result.model[3]
    assert result.model(x).dtype==torch.float32
    assert result.evidence['resources']['actual_training_steps']==2


def test_unified_program_qat_and_recovery_are_checked_stages():
    from fedcore.algorithm.quantization.program import QuantizationProgram,RecoveryStageSpec,ProgramArtifact,quantize,training_identity
    from fedcore.algorithm.quantization.graph_specs import QuantizationGraphRequest,OperatorRecipe,TransformSpec,Method
    torch.manual_seed(6)
    model=nn.Sequential(nn.Linear(4,3))
    x,y=torch.randn(7,4),torch.randn(7,3);batches=((x,y),)
    dense=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DENSE)),))
    program=QuantizationProgram(dense,backend=BackendStageSpec('qat',('0',),training_steps=1),training_fingerprint=training_identity(batches))
    result=quantize(model,program,calibration=x,training_batches=batches)
    assert isinstance(result,ProgramArtifact),result
    assert 'regional_prepare' in result.plan.stages and 'convert' in result.plan.stages
    assert result.evidence['resources']['actual_training_steps']==1
    factors=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DSVD,rank=2)),))
    recovery=QuantizationProgram(factors,recovery=RecoveryStageSpec(('0',),steps=1),training_fingerprint=training_identity(batches))
    recovered=quantize(model,recovery,calibration=x,training_batches=batches)
    assert isinstance(recovered,ProgramArtifact),recovered
    assert 'encode_ready_factors' in recovered.plan.stages
    stale=quantize(model,recovery,calibration=x,training_batches=((x,y+1),))
    assert isinstance(stale,StructuralInvalid)


def test_cross_entropy_qat_accepts_integer_labels_and_hashes_their_order():
    from fedcore.algorithm.quantization.program import QuantizationProgram,ProgramArtifact,quantize,training_identity
    from fedcore.algorithm.quantization.graph_specs import QuantizationGraphRequest,OperatorRecipe,TransformSpec,Method
    model=nn.Sequential(nn.Linear(3,3));x=torch.randn(5,3);labels=torch.tensor([0,1,2,0,1])
    batches=((x,labels),)
    request=QuantizationProgram(QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DENSE)),)),
        backend=BackendStageSpec('qat',('0',),training_steps=1,criterion='cross_entropy'),training_fingerprint=training_identity(batches))
    result=quantize(model,request,calibration=x,training_batches=batches)
    assert isinstance(result,ProgramArtifact),result
    assert result.evidence['resources']['actual_training_steps']==1
    assert training_identity(((x,labels.flip(0)),))!=training_identity(batches)


@pytest.mark.parametrize('stage',['recovery','qat'])
def test_seeded_program_is_reproducible_and_preserves_external_rng(stage):
    from fedcore.algorithm.quantization.program import QuantizationProgram,RecoveryStageSpec,ProgramArtifact,quantize,training_identity
    from fedcore.algorithm.quantization.graph_specs import QuantizationGraphRequest,OperatorRecipe,TransformSpec,Method
    torch.manual_seed(25)
    model=nn.Sequential(nn.Linear(3,3));x,y=torch.randn(6,3),torch.randn(6,3);batches=((x,y),)
    compression=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DSVD,rank=2) if stage=='recovery' else TransformSpec(Method.DENSE)),))
    program=QuantizationProgram(compression,recovery=RecoveryStageSpec(('0',),steps=1,seed=37) if stage=='recovery' else None,
        backend=BackendStageSpec('qat',('0',),training_steps=1,seed=37) if stage=='qat' else None,training_fingerprint=training_identity(batches))
    before=torch.random.get_rng_state().clone()
    first=quantize(model,program,calibration=x,training_batches=batches)
    assert isinstance(first,ProgramArtifact),first
    assert torch.equal(torch.random.get_rng_state(),before)
    torch.randn(17)  # Candidate order cannot change a seeded training artifact.
    second_before=torch.random.get_rng_state().clone()
    second=quantize(model,program,calibration=x,training_batches=batches)
    assert isinstance(second,ProgramArtifact),second
    assert torch.equal(torch.random.get_rng_state(),second_before)
    assert torch.equal(first.model(x),second.model(x))
    assert first.checkpoint_bytes==second.checkpoint_bytes


def test_reused_base_quantizer_clears_completed_steps_before_new_failure():
    from types import SimpleNamespace
    from fedcore.algorithm.quantization.quantizers import BaseQuantizer,QuantizationError
    model=nn.Sequential(nn.Linear(3,2));x,y=torch.randn(4,3),torch.randn(4,2)
    data=SimpleNamespace(model=model,calibration_dataloader=((x,y),),train_dataloader=((x,y),))
    quantizer=BaseQuantizer({'quant_type':'qat','qat_params':{'epochs':1,'max_steps':1,'criterion':'mse'}})
    quantizer.fit(data)
    assert quantizer.quantization_result.training_steps==1
    quantizer.qconfig_mapping='invalid'
    with pytest.raises(QuantizationError):quantizer.fit(data)
    assert quantizer.quantization_result.training_steps==0
    assert quantizer.training_evidence=={}
