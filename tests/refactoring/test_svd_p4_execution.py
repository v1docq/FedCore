"""Public P4 boundaries: actual training, provenance, compact export and refusal."""
from copy import deepcopy
import json
import math

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.method_specs import ARS,DobiSVD,RankDyna,method_payload,parse_method
from fedcore.algorithm.low_rank.method_execution import plan_method,transform_method
from fedcore.algorithm.low_rank.learned_execution import LearnedProfileError
from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
from fedcore.experiments.protocol import TensorSplit,ProtocolError
from fedcore.experiments.runner import train_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager,build_model,describe_model,CheckpointError
from fedcore.tools.export import export_model


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = nn.Sequential(nn.Linear(6,6),nn.Tanh(),nn.Linear(6,3)).double().eval()
        x = torch.randn(12,6,dtype=torch.double)
    split = TensorSplit(x,model(x).detach()+.2,tuple(f'train-{i}' for i in range(len(x))))
    return model,x,split


def executor(split):
    def run(model,stage,updates,objective):
        result = train_model(model,split,'regression',epochs=math.ceil(updates/3),
            batch_size=4,learning_rate=.005,seed=23,batch_objective=objective,max_updates=updates)
        return dict(result,updates=result['training_steps'],stage=stage)
    return run


SPECS = (ARS(updates=3,input_size=4,hidden_size=4,mask_bias=0),
         DobiSVD(updates=3,max_activation_rows=32,max_calibration_batches=8),
         RankDyna(updates=3))


@pytest.mark.parametrize('spec',SPECS)
def test_public_training_is_reproducible_private_and_checkpoint_exportable(spec,tmp_path):
    source,x,split = case()
    snapshot = deepcopy(source.state_dict())
    rng = torch.random.get_rng_state().clone()
    payload = json.loads(json.dumps(method_payload(spec)))
    assert parse_method(payload['method'],payload['options'],version=payload['version']) == spec
    first = transform_method(source,x,spec,target_paths=('0',),batch_size=4,recovery_executor=executor(split))
    assert torch.equal(rng,torch.random.get_rng_state())
    second = transform_method(source,x,spec,target_paths=('0',),batch_size=4,recovery_executor=executor(split))
    for name,value in source.state_dict().items():
        assert torch.equal(value,snapshot[name])
    for name,value in first.model.state_dict().items():
        torch.testing.assert_close(value,second.model.state_dict()[name],rtol=0,atol=0)
    for name,value in source[2].state_dict().items():
        assert torch.equal(value,first.model[2].state_dict()[name])
    assert first.evidence['training']['training_steps'] == spec.updates
    assert first.evidence['structure_training']['observed_optimizer_steps'] == spec.updates
    assert first.evidence['parameters_after'] == sum(p.numel() for p in first.model.parameters())
    json.dumps(first.evidence,allow_nan=False)
    manager = CheckpointManager(str(tmp_path/'registry'),auto_cleanup=False)
    path = tmp_path/'model.pt'
    manager.save_to_file(manager.serialize_to_bytes(first.model),str(path))
    restored = manager.load_from_file(str(path)).eval()
    torch.testing.assert_close(restored(x),first.model.eval()(x))
    assert restored._fedcore_method_evidence == first.evidence
    artifact = export_model(restored,'torchscript',tmp_path/'export.pt',x[:1])
    with artifact.open('rb') as stream:
        deployed = torch.jit.load(stream)
    torch.testing.assert_close(deployed(x[:1]),restored(x[:1]))


@pytest.mark.parametrize('spec',SPECS)
def test_plan_is_read_only_and_fixed_rank_and_unexecuted_training_refuse(spec,monkeypatch):
    source,x,split = case()
    snapshot = deepcopy(source.state_dict())
    def forbidden(*args,**kwargs):
        raise AssertionError('Planning must not run SVD or a model forward')
    monkeypatch.setattr(torch.linalg,'svd',forbidden)
    handle = source.register_forward_pre_hook(forbidden)
    try:
        plan = plan_method(source,spec,target_paths=('0',))
        assert plan.request.objective.endswith('_structure_training')
        with pytest.raises(LearnedProfileError,match='fixed rank'):
            plan_method(source,spec,rank=2,target_paths=('0',))
        with pytest.raises(LearnedProfileError,match='explicit fixed-update'):
            transform_method(source,x,spec,target_paths=('0',))
        with pytest.raises(Exception,match='ResourceLimitExceeded'):
            transform_method(source,x,spec,target_paths=('0',),max_workspace_bytes=16,
                             recovery_executor=executor(split))
    finally:
        handle.remove()
    assert all(torch.equal(value,snapshot[name]) for name,value in source.state_dict().items())


def test_reported_updates_without_real_training_are_rejected():
    source,x,_ = case()
    with pytest.raises(LearnedProfileError,match='exact context-managed'):
        transform_method(source,x,RankDyna(),target_paths=('0',),
            recovery_executor=lambda model,stage,updates,objective:{'updates':updates})


def test_dobi_row_limit_precedes_output_allocation(monkeypatch):
    from fedcore.algorithm.low_rank.dobi_profiles import prepare_dobi
    source,_,_ = case()
    objective = prepare_dobi(source,('0',),DobiSVD(max_activation_rows=2))
    def forbidden(*args,**kwargs):
        raise AssertionError('Row limit must precede linear output allocation')
    monkeypatch.setattr(torch.nn.functional,'linear',forbidden)
    with pytest.raises(Exception,match='activation rows'):
        objective.model(torch.ones(3,6,dtype=torch.double))


@pytest.mark.parametrize('bias',[True,False])
def test_zero_rank_has_no_dense_matrix_and_portable_geometry(bias,tmp_path):
    layer = nn.Linear(5,3,bias=bias).double()
    zero = ZeroLinear.from_linear(layer)
    x = torch.ones(2,4,5,dtype=torch.double)
    expected = x.new_zeros(2,4,3)
    if bias:
        expected += layer.bias
    torch.testing.assert_close(zero(x),expected)
    assert sum(p.numel() for p in zero.parameters()) == (3 if bias else 0)
    assert all(value.numel() <= 3 for value in zero.state_dict().values())
    restored = build_model(describe_model(zero))
    restored.load_state_dict(zero.state_dict())
    torch.testing.assert_close(restored(x),zero(x))
    invalid = describe_model(zero)
    invalid['config']['bias'] = 'false'
    with pytest.raises(CheckpointError):
        build_model(invalid)


def test_training_boundary_detects_optimizer_left_with_replaced_parameter():
    source,_,split = case()
    class BadTransition:
        evidence = {}
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def __call__(self,model,x,y): return (model(x)-y).square().mean()
        def after_step(self,model,optimizer):
            model[0].weight = nn.Parameter(model[0].weight.detach().clone())
    with pytest.raises(ProtocolError,match='stale or duplicate'):
        train_model(source,split,'regression',epochs=1,batch_size=4,learning_rate=.01,
                    batch_objective=BadTransition(),max_updates=1)


@pytest.mark.parametrize('spec',SPECS)
def test_petra_named_candidate_uses_only_declared_training_stage(spec):
    from fedcore.experiments.protocol import ExperimentBundle,ExperimentProtocol,CandidateSpec
    from fedcore.experiments.runner import apply_candidate
    source,x,train = case()
    splits = {role:TensorSplit(x,train.y,tuple(f'{role}-{i}' for i in range(len(x))))
              for role in ('calibration','validation','test')}
    bundle = ExperimentBundle(train=train,original_model=source,task='regression',**splits)
    protocol = ExperimentProtocol(baseline_epochs=0,finetune_epochs=0,batch_size=4,
                                  learning_rate=.005,measurement_repeats=2,warmup=1)
    payload = method_payload(spec)
    candidate = CandidateSpec(payload['method'],{'method_options':payload['options'],
                              'target_paths':['0'],'finetune_epochs':0})
    result,evidence = apply_candidate(source,bundle,protocol,candidate)
    assert result is not source
    assert evidence['training_steps'] == spec.updates
    training = evidence['method_transform']['training']
    assert training['training_data']['ids'] == list(train.ids)
    assert training['optimizer_state_bytes_peak'] >= training['optimizer_state_bytes']
    assert len(training['step_resource_history']) == spec.updates
