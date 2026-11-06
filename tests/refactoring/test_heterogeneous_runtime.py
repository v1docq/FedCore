"""Small CPU reference cases for the new evaluator and bounded host transport."""
import json
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fedcore.experiments.heterogeneous_runtime import (
    AttemptLedger, RecoveryConfig, evaluate_loaded, recover_owned, recovery_tape)
from fedcore.experiments.heterogeneous_host import _run_bounded


def test_tape_conserves_rows_and_does_not_depend_on_search_rng():
    before = random.getstate()
    config = RecoveryConfig(updates=3,batch_size=2)
    first = recovery_tape((8,9,10,11,12),model_seed=42,config=config)
    assert first == recovery_tape((8,9,10,11,12),model_seed=42,config=config)
    assert len(first) == 3 and all(set(b.indices) <= {8,9,10,11,12} for b in first)
    assert len(set(i for b in first[:2] for i in b.indices)) == 4
    assert random.getstate() == before


def test_partial_recovery_is_recorded_and_global_rng_restored(tmp_path):
    config = RecoveryConfig(updates=3,batch_size=2)
    tape = recovery_tape((0,1,2,3),model_seed=1,config=config)
    model = torch.nn.Linear(2,2)
    ledger = AttemptLedger(tmp_path/'attempt','workflow',2,'heldout-source')
    torch_state, py_state = torch.get_rng_state().clone(), random.getstate()
    np_state = np.random.get_state()
    calls = []
    def batch(indices, seed):
        calls.append(seed)
        if len(calls) == 2: raise ValueError('simulated second-batch failure')
        return torch.ones(2,2),torch.tensor([0,1])
    with pytest.raises(ValueError,match='second-batch'):
        recover_owned(model,tape,batch,config=config,deadline=time.perf_counter()+10,progress=ledger.progress)
    ledger.close('failed',failure='simulated')
    receipt = json.loads((tmp_path/'attempt/receipt.json').read_text())
    assert receipt['sgd_attempted'] == receipt['sgd_completed'] == 1
    assert not model.training
    assert torch.equal(torch.get_rng_state(),torch_state) and random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1],np_state[1])
    events = [json.loads(s)['event'] for s in (tmp_path/'attempt/progress.jsonl').read_text().splitlines()]
    assert events.index('sgd_attempted') < events.index('sgd_completed') < events.index('finished')


def test_deployed_quality_requires_complete_heldout_rows():
    batches = [(torch.tensor([[2.,0.],[0.,2.]]),torch.tensor([0,1]))]
    actual = evaluate_loaded(lambda x:x,batches,expected_rows=2,deadline=time.perf_counter()+5)
    assert actual['rows'] == 2 and actual['accuracy'] == 1.
    assert actual['cross_entropy'] == pytest.approx(np.log1p(np.exp(-2)))
    with pytest.raises(ValueError,match='incomplete'):
        evaluate_loaded(lambda x:x,batches,expected_rows=3,deadline=time.perf_counter()+5)


def test_bounded_transport_real_cpu_child_and_output_limit():
    command = (sys.executable,'-c','import sys; print(sys.stdin.read())')
    assert _run_bounded(command,'explicit-request',2).strip() == 'explicit-request'
    with pytest.raises(RuntimeError,match='limit'):
        _run_bounded((sys.executable,'-c','print("x"*100000)'),'',2)
    with pytest.raises(TimeoutError,match='timeout'):
        _run_bounded((sys.executable,'-c','import time; time.sleep(5)'),'',.05)


def test_cuda_resident_entry_rejects_cpu_input_without_touching_cuda():
    from fedcore.algorithm.heterogeneous.backend import TensorRTDeployment, BackendContractError
    runtime = object.__new__(TensorRTDeployment)
    runtime.manifest = {'backend_spec':{'max_batch_size':128}}
    with pytest.raises(BackendContractError,match='CUDA'):
        runtime.infer_device(torch.zeros(1,3,32,32))


def test_candidate_rejects_cpu_only_ptq_placement_before_cuda_or_recovery(tmp_path, monkeypatch):
    from fedcore.algorithm.heterogeneous import (
        BlockRecipe, HeterogeneousGene, HeterogeneousGenotype, PTQPlacement, compile_genotype, prepare_space)
    from fedcore.algorithm.heterogeneous.backend import BackendSpec
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
    from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18
    from fedcore.experiments import heterogeneous_device, heterogeneous_runtime

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        source = CifarResNet18().eval()
        calibration = torch.randn(1, 3, 32, 32)
    path = 'layer1.0'
    recipe = BlockRecipe(ptq=True, ptq_placement=PTQPlacement(omit_output_qdq=(path + '.conv1',)))
    space = prepare_space(source, calibration, alternatives={path: (recipe,)})
    workflow = compile_genotype(HeterogeneousGenotype((HeterogeneousGene(path, recipe),)), space)

    def unexpected(*args, **kwargs):
        pytest.fail('unsupported placement reached CUDA or recovery')

    monkeypatch.setattr(heterogeneous_device, 'require_cuda_binding', unexpected)
    monkeypatch.setattr(heterogeneous_runtime, 'recover_owned', unexpected)
    evaluator = heterogeneous_runtime.CandidateEvaluator(
        source, calibration, (), unexpected, unexpected, tmp_path, time.perf_counter() + 30,
        BackendSpec(runtime='tensorrt'), 'heldout-source', device='cuda:0')
    result = evaluator(workflow)
    assert type(result) is StructuralInvalid
    assert 'ONNX Runtime' in result.violations[0].message
    receipt = json.loads((tmp_path / workflow.fingerprint / 'receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['sgd_attempted'] == receipt['sgd_completed'] == 0
    assert receipt['failure']['violations'][0]['code'] == result.violations[0].code


def test_gpu_telemetry_is_scoped_to_the_explicit_uuid(monkeypatch):
    from fedcore.experiments import heterogeneous_host as host
    calls = []
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='1001, 20\n')
    monkeypatch.setattr(host.subprocess,'run',run)
    uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
    assert host.query_device_gpu_rows(uuid) == ('1001, 20',)
    assert '--id='+uuid in calls[0]
    with pytest.raises(ValueError,match='UUID'): host.query_device_gpu_rows('3')
