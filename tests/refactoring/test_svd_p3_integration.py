"""Shared runtime boundaries for P3: fixed recovery, metadata and capabilities."""
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.method_specs import (
    ESPACE, PELA, LASER, CTRAFM, CTRMLPAFM, FlatValueOutput, MoDeValueOutput,
    FlatMLPChannels, MoDeMLPChannels, FlatQueryKey, MoDeQueryKey,
    method_payload, parse_method,
)
from fedcore.experiments.protocol import ProtocolError, TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.external_runtime.contracts import ContractError, MethodOptions


def test_all_method_descriptions_remain_metadata_only_without_importing_torch():
    code = ('import sys; from fedcore.algorithm.low_rank.method_specs import ESPACE, FlatQueryKey; '
            'assert "torch" not in sys.modules; assert ESPACE().criterion == "mse"')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('spec', [ESPACE(), PELA(block_map=(("0","0"),)), LASER(),
    CTRAFM((("user",2),)), CTRMLPAFM(), FlatValueOutput(), MoDeValueOutput(), FlatMLPChannels((0,2)),
    MoDeMLPChannels((1,3)), FlatQueryKey((0,1)), MoDeQueryKey((2,3))])
def test_p3_metadata_wire_roundtrip_and_external_v3_explicit_refusal(spec):
    payload = json.loads(json.dumps(method_payload(spec), allow_nan=False))
    assert parse_method(payload['method'], payload['options'], version=payload['version']) == spec
    with pytest.raises(ContractError, match='Unsupported external'):
        MethodOptions(spec)


class _Objective:
    def __init__(self, model, *, fail=False):
        self.model, self.fail = model, fail
        self.entered = self.exited = False
        self.handle = None
        self.observations = 0

    def __enter__(self):
        self.entered = True
        self.handle = self.model.register_forward_hook(self._observe)
        return self

    def _observe(self, _module, _args, _output):
        self.observations += 1

    def __exit__(self, *args):
        self.handle.remove()
        self.exited = True

    def __call__(self, model, x, y):
        residual = model(x) - y
        return residual.square().mean() * (float('nan') if self.fail else 1.)

    @property
    def evidence(self):
        return {'objective': 'independent_mse_reference', 'observations': self.observations}


def _training_case():
    model = nn.Linear(2, 1, dtype=torch.float64)
    x = torch.arange(24, dtype=torch.float64).reshape(12,2) / 24
    y = x[:, :1] * 2 + 1
    return model, TensorSplit(x, y, tuple(f'train-{i}' for i in range(12)))


def test_shared_training_executes_exact_fixed_updates_and_owns_objective_cleanup():
    model, split = _training_case()
    before = model.weight.detach().clone()
    objective = _Objective(model)
    evidence = train_model(model, split, 'regression', epochs=2, batch_size=4,
        learning_rate=.01, batch_objective=objective, max_updates=5)
    assert evidence['training_steps'] == objective.observations == 5
    assert objective.entered and objective.exited and not model._forward_hooks
    assert not torch.equal(model.weight, before)
    assert evidence['training_role'] == 'train'
    assert evidence['optimizer_state_bytes'] > 2*sum(p.numel()*p.element_size() for p in model.parameters())
    assert evidence['recovery_objective']['observations'] == 5
    json.dumps(evidence, allow_nan=False)


def test_shared_training_removes_recovery_hooks_when_loss_fails():
    model, split = _training_case()
    objective = _Objective(model, fail=True)
    before = model.weight.detach().clone()
    with pytest.raises(ProtocolError, match='finite scalar'):
        train_model(model, split, 'regression', epochs=1, batch_size=4,
            learning_rate=.01, batch_objective=objective, max_updates=1)
    assert objective.exited and not model._forward_hooks
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize('updates', [True, 0, -1, 4])
def test_impossible_fixed_budget_refuses_before_optimizer_creation(monkeypatch, updates):
    model, split = _training_case()
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid recovery budget must be checked before effects')
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    with pytest.raises(ProtocolError, match='max_updates'):
        train_model(model, split, 'regression', epochs=1, batch_size=4,
            learning_rate=.01, max_updates=updates)


@pytest.mark.parametrize('spec', [FlatValueOutput(), MoDeValueOutput(),
    FlatMLPChannels(), MoDeMLPChannels(), FlatQueryKey(), MoDeQueryKey()])
def test_public_coupled_profiles_roundtrip_and_export_physical_compact_graph(tmp_path, monkeypatch, spec):
    from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    from fedcore.tools.export import export_model
    from fedcore.experiments.measurement import tensor_state_bytes
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH', str(tmp_path / 'registry'))
    torch.set_num_threads(4)
    is_mlp = isinstance(spec, (FlatMLPChannels,MoDeMLPChannels))
    model = (ReferenceMLP(8,12,dtype=torch.float64) if is_mlp else
             ReferenceAttention(8,2,1,4,dtype=torch.float64)).eval()
    x = torch.randn(8,3,8,dtype=torch.float64)
    before = {name:value.detach().clone() for name,value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    facade = LowRankModel({'device':'cpu'})
    result = facade.compress_profile(model,x,spec,rank=2,target_paths=('',),batch_size=4)
    assert result.model is facade.model_after and facade.model_before is model
    assert result.evidence['tensor_bytes_after'] == tensor_state_bytes(result.model)
    assert result.evidence['parameters_after'] < result.evidence['parameters_before']
    assert torch.equal(torch.get_rng_state(),rng)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert not model._forward_pre_hooks and all(p.grad is None for p in model.parameters())
    manager = CheckpointManager(str(tmp_path/'checkpoints'),auto_cleanup=False)
    checkpoint = tmp_path/'compressed.ckpt'
    manager.save_to_file(manager.serialize_to_bytes(result.model),str(checkpoint))
    restored = manager.load_from_file(str(checkpoint)).eval()
    holdout = torch.randn(2,3,8,dtype=torch.float64)
    torch.testing.assert_close(restored(holdout),result.model.eval()(holdout))
    assert restored._fedcore_method_evidence == json.loads(json.dumps(result.evidence))
    artifact = export_model(restored,'torchscript',tmp_path/'deployed.pt',holdout)
    with artifact.open('rb') as stream:
        deployed = torch.jit.load(stream)
    torch.testing.assert_close(deployed(holdout),restored(holdout))


@pytest.mark.parametrize('method,options', [
    ('espace',{'criterion':'nl_first_order_bound','updates':2}),
    ('pela',{'block_map':[['0','0'],['2','2']],'updates':2,'kappa':1.5}),
    ('ctr_mlp_afm',{'nonlinear_variant':'relu_between_factors_v1','updates':2}),
])
def test_petra_fixed_recovery_uses_train_only_and_exact_steps(monkeypatch,method,options):
    from fedcore.experiments import runner
    from fedcore.experiments.protocol import CandidateSpec,ExperimentBundle,ExperimentProtocol
    torch.set_num_threads(4)
    model = nn.Sequential(nn.Linear(6,6),nn.Tanh(),nn.Linear(6,2)).double().eval()
    generator = torch.Generator().manual_seed(392)
    splits = []
    for index,role in enumerate(('train','validation','calibration','test')):
        x = torch.randn(12,6,generator=generator,dtype=torch.float64)+index
        splits.append(TensorSplit(x,model(x).detach()+.1,tuple(f'{role}-{i}' for i in range(12))))
    bundle = ExperimentBundle(*splits,'regression',model)
    protocol = ExperimentProtocol(batch_size=4,baseline_epochs=0,finetune_epochs=0,
                                  learning_rate=.01,threads=4)
    calls = []
    original = runner.train_model
    def tracked(current,split,task,**kwargs):
        assert split is bundle.train
        assert kwargs['max_updates'] == 2
        result = original(current,split,task,**kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(runner,'train_model',tracked)
    parameters = {'method_options':options,'target_paths':['0','2']}
    if method in ('espace','ctr_mlp_afm'):
        parameters['rank'] = 2
    snapshot = {name:value.detach().clone() for name,value in model.state_dict().items()}
    result,evidence = runner.apply_candidate(model,bundle,protocol,CandidateSpec(method,parameters))
    assert len(calls) == 1 and calls[0]['training_steps'] == evidence['training_steps'] == 2
    assert calls[0]['optimizer_state_bytes'] > 0
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in result.modules())
    assert torch.isfinite(result(bundle.validation.x)).all()
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,snapshot[name],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
    json.dumps(evidence,allow_nan=False)
