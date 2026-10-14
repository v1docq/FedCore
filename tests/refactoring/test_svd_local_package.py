"""Protocol counts, independent moment reference and continuous-Adam recovery."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from fedcore.experiments import runner
from fedcore.experiments.protocol import ExperimentBundle, ProtocolError, TensorSplit, stable_hash, tensor_hash
from fedcore.experiments.svd_local_package import (
    RecoveryObjective, initial_transform_plan, nested_calibration, planned_batches,
    ridge_from_inputs, select_validation_policy, timing_schedule,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def split(count=19):
    generator = torch.Generator().manual_seed(41)
    return TensorSplit(torch.randn(count,3,generator=generator),torch.arange(count)%2,tuple(f"sample{i}" for i in range(count)))


def test_schedule_is_fixed_balanced_and_complete():
    plan = timing_schedule()
    assert plan == timing_schedule() and plan != timing_schedule(43)
    assert len(plan) == len({row.session_id for row in plan}) == 132
    artifacts = [row for row in plan if row.kind == "artifact"]
    assert len(artifacts)==96
    first = [row.side for row in artifacts if row.order==0]
    assert first.count("baseline")==first.count("candidate")==24
    for device in ("cpu","cuda:0"):
        for batch in (1,64):
            micro = [row for row in plan if row.kind=="conv2" and row.device==device and row.batch==batch]
            assert len(micro)==9
            for subject in ("original_dense","factor_r8","materialized_r8"):
                assert {row.order for row in micro if row.subject==subject}=={0,1,2}
    initial = initial_transform_plan()
    assert len(initial)==12 and len({row.case_id for row in initial})==12


def test_selection_ignores_development_test_and_failure_rows():
    def row(identifier,accuracy,ce,n=180,alpha=0,test_accuracy=1):
        return {"status":"succeeded","spec":{"case_id":identifier,"calibration_n":n,"alpha":alpha},
                "quality":{"validation":{"value":accuracy,"cross_entropy":ce},"test":{"value":test_accuracy}}}
    rows = [row("high_test",.8,.2,test_accuracy=1),row("chosen",.9,.4,test_accuracy=0),
            {"status":"unsupported","reason":"rank overflow"}]
    assert select_validation_policy(rows)["spec"]["case_id"]=="chosen"
    ties = [row("b",.9,.4,90,1e-4),row("a",.9,.4,90,1e-4),row("c",.9,.4,180,0)]
    assert select_validation_policy(ties)["spec"]["case_id"]=="a"
    rows[1]["quality"]["validation"]["cross_entropy"]=float("nan")
    with pytest.raises(ProtocolError,match="finite"):
        select_validation_policy(rows)


def test_nested_roles_preserve_historical_pool_order_and_hashes():
    base = split(180)
    independent = [replace(split(),ids=tuple(f"{role}-{i}" for i in range(19))) for role in ("train","validation","test")]
    bundle = ExperimentBundle(independent[0],independent[1],base,independent[2],"classification",nn.Linear(3,2))
    # Role identity validation is the runner's responsibility; this helper only
    # slices the explicitly selected calibration pool.
    short = nested_calibration(bundle,32)
    medium = nested_calibration(bundle,90)
    assert short.calibration.ids==medium.calibration.ids[:32]==base.ids[:32]
    assert torch.equal(short.calibration.x,base.x[:32])
    assert short.train is bundle.train and short.test is bundle.test
    with pytest.raises(ProtocolError):
        nested_calibration(bundle,33)


def test_ridge_conv_moment_matches_explicit_zero_padded_windows():
    layer = nn.Conv2d(1,2,3,padding=1,bias=True).double()
    inputs = torch.arange(18,dtype=torch.float64).reshape(2,1,3,3)
    value,evidence = ridge_from_inputs(layer,inputs,1e-4)
    padded = nn.functional.pad(inputs,(1,1,1,1))
    rows = torch.stack([padded[n,:,i:i+3,j:j+3].flatten()
                        for n in range(2) for i in range(3) for j in range(3)])
    covariance = rows.T@rows/len(rows)
    assert value==pytest.approx(1e-4*torch.trace(covariance).item()/9,abs=1e-14)
    assert evidence["images"]==2 and evidence["correlated_windows"]==18
    assert evidence["window_independence_claim"] is False


def test_continuous_adam_checkpoints_batch_order_and_teacher_are_preserved():
    data=split(19)
    source=nn.Linear(3,2)
    teacher=deepcopy(source).eval().requires_grad_(False)
    initial=runner.model_state_hash(teacher)
    indices=planned_batches(data,updates=100,batch_size=4)
    expected=[stable_hash({"x":tensor_hash(data.x[i]),"y":tensor_hash(data.y[i])}) for i in indices]
    observed=[]
    objective=RecoveryObjective(teacher,lambda model,optimizer,step:observed.append((step,id(optimizer),
                    [int(state["step"].item()) for state in optimizer.state.values()])),expected_batch_hashes=expected)
    result=runner.train_model(source,data,"classification",epochs=25,batch_size=4,learning_rate=1e-4,
                              seed=42,batch_objective=objective,max_updates=100,drop_last=True)
    assert result["training_steps"]==objective.updates==100
    assert observed[0][0]==100 and set(observed[0][2])=={100}
    assert objective.evidence["visited_samples"]==400
    assert objective.evidence["teacher_forwards"]==100 and objective.evidence["teacher_unchanged"]
    assert runner.model_state_hash(teacher)==initial and all(p.grad is None for p in teacher.parameters())
    assert objective.evidence["batch_hashes"]==expected
    json.dumps(result,allow_nan=False)


def test_custom_objective_rejects_wrong_batch_order_before_student_forward():
    model=nn.Linear(3,2)
    data=split(4)
    objective=RecoveryObjective(expected_batch_hashes=("not-real",))
    with objective,pytest.raises(ProtocolError,match="order"):
        objective(model,data.x,data.y)
    assert objective.updates==0 and objective.samples==0


def test_runner_drop_last_budget_validation_retains_default_behavior():
    data=split(5)
    model=nn.Linear(3,2)
    with pytest.raises(ProtocolError,match="epoch budget"):
        runner.train_model(model,data,"classification",epochs=1,batch_size=4,learning_rate=.01,
                           drop_last=True,max_updates=2)
    result=runner.train_model(model,data,"classification",epochs=1,batch_size=4,learning_rate=.01,max_updates=2)
    assert result["training_steps"]==2
