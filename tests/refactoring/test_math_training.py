from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from fedcore.algorithm.distillation.distilator import BaseDistilator
from fedcore.algorithm.quantization.quantizers import BaseQuantizer,QuantizationError,validate_quantization_request

@pytest.fixture
def training_data(tmp_path,monkeypatch):
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH',str(tmp_path/'registry'))
    from fedcore.tools.registry.model_registry import ModelRegistry
    ModelRegistry._instance=None;ModelRegistry._initialized=False
    torch.manual_seed(19)
    model=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,3))
    loader=DataLoader(TensorDataset(torch.randn(8,4),torch.randint(0,3,(8,))),batch_size=4)
    return SimpleNamespace(model=model,target=model,train_dataloader=loader,val_dataloader=loader,
                           calibration_dataloader=loader)


def test_actual_distillation_student_step_teacher_unchanged(training_data):
    teacher=training_data.model
    student=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,3))
    before_student=deepcopy(student.state_dict());before_teacher=deepcopy(teacher.state_dict())
    op=BaseDistilator({'epochs':1,'student_model':student,'device':'cpu'})
    after=op.fit(training_data)
    assert len(op.history)==2 and all(torch.isfinite(torch.tensor(op.history)))
    assert any(not torch.equal(before_student[k],v) for k,v in after.state_dict().items())
    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,before_teacher[k],rtol=0,atol=0)
    assert all(p.grad is None for p in op.base_model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in after.parameters())

@pytest.mark.parametrize('mode',['dynamic','static','qat'])
def test_actual_quantization_stages_and_baseline(training_data,mode):
    source=training_data.model;source.train()
    x=next(iter(training_data.train_dataloader))[0]
    weights=deepcopy(source.state_dict());before=source(x).detach().clone()
    op=BaseQuantizer({'quant_type':mode,'qat_params':{'epochs':1,'lr':.01}})
    after=op.fit(training_data)
    assert op.quantization_result.status=='completed'
    assert op.quantization_result.training_steps==(2 if mode=='qat' else 0)
    assert after(x).shape==(4,3)
    assert source.training
    torch.testing.assert_close(source(x),before)
    for k,v in source.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=0,atol=0)
    assert any(type(m).__module__.startswith('torch.ao.nn.quantized') for m in after.modules())
    if mode=='qat':assert len(op.history['train_loss'])==2


def test_quantization_rejects_invalid_composition(training_data):
    op=BaseQuantizer({'quant_type':'dynamic'})
    quantized=op.fit(training_data)
    before=deepcopy(quantized.state_dict())
    bad=SimpleNamespace(model=quantized,train_dataloader=training_data.train_dataloader)
    with pytest.raises(QuantizationError,match='Already quantized'):BaseQuantizer().fit(bad)
    with pytest.raises(ValueError,match='floating tensor'):
        validate_quantization_request(training_data.model,torch.quantize_per_tensor(torch.ones(2,4),.1,0,torch.quint8),'qat','fbgemm',torch.qint8)
    with pytest.raises(QuantizationError):BaseQuantizer({'quant_type':'qat','qat_params':{'epochs':0}}).fit(training_data)


def test_actual_lora_operation_adapter_only(training_data):
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    source=training_data.model
    weights=deepcopy(source.state_dict())
    op=BaseLoRA({'epochs':1,'lora_r':2,'lr':.01,'device':'cpu'})
    result=op.fit(training_data)
    assert len(op.history)==2
    assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in result.named_parameters() if 'lora_B' in name)
    assert all(p.grad is None for name,p in result.named_parameters() if 'base_layer' in name)
    for k,v in source.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=0,atol=0)


def test_qat_observed_real_stages_before_convert(training_data,monkeypatch):
    from fedcore.algorithm.quantization import quantizers
    events=[]
    originals={name:getattr(quantizers,name) for name in ("prepare_qat_fx","convert_fx")}
    step=torch.optim.SGD.step
    backward=torch.Tensor.backward
    handles=[]
    def prepare(*args,**kwargs):
        model=originals["prepare_qat_fx"](*args,**kwargs)
        events.append("prepare")
        assert any("fake_quant" in name for name,_ in model.named_modules())
        handles.append(model.register_forward_hook(lambda *a:events.append("forward")))
        return model
    def observed_backward(tensor,*args,**kwargs):
        assert tensor.ndim==0 and torch.isfinite(tensor)
        events.append("backward")
        return backward(tensor,*args,**kwargs)
    def observed_step(optimizer,*args,**kwargs):
        gradients=[p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g)>0 for g in gradients)
        events.append("step")
        return step(optimizer,*args,**kwargs)
    def convert(*args,**kwargs):
        events.append("convert")
        assert events[:7]==["prepare","forward","backward","step","forward","backward","step"]
        for handle in handles:handle.remove()
        return originals["convert_fx"](*args,**kwargs)
    monkeypatch.setattr(quantizers,"prepare_qat_fx",prepare)
    monkeypatch.setattr(quantizers,"convert_fx",convert)
    monkeypatch.setattr(torch.Tensor,"backward",observed_backward)
    monkeypatch.setattr(torch.optim.SGD,"step",observed_step)
    BaseQuantizer({"quant_type":"qat","qat_params":{"epochs":1,"optimizer":"sgd","lr":.01}}).fit(training_data)
    assert events==["prepare","forward","backward","step","forward","backward","step","convert"]


def test_actual_magnitude_pruning_preserves_baseline(training_data):
    from fedcore.algorithm.pruning.pruners import BasePruner
    from fedcore.data.data import CompressionInputData
    from fedot.core.repository.tasks import Task,TaskTypesEnum
    source=training_data.model
    before=deepcopy(source.state_dict())
    features=next(iter(training_data.train_dataloader))[0]
    expected=source(features).detach().clone()
    data=CompressionInputData(model=source,train_dataloader=training_data.train_dataloader,
        val_dataloader=training_data.val_dataloader,num_classes=3,input_dim=4,
        task=Task(TaskTypesEnum.classification))
    operation=BasePruner({"device":"cpu","importance":"magnitude","epochs":1,
        "prune_each":-1,"pruning_ratio":.5,"pruning_iterations":1,"criterion":"cross_entropy",
        "log_each":0,"save_each":0})
    result=operation.fit(data)
    assert result[0].out_features==4 and result[2].in_features==4
    assert result(features).shape==(4,3)
    assert operation.model_before is not source and operation.model_before is not result
    assert operation.trainer.device=="cpu" and operation.data_batch_for_calib.device.type=="cpu"
    for k,v in source.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    for k,v in operation.model_before.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    torch.testing.assert_close(source(features),expected,rtol=0,atol=0)


def test_actual_low_rank_onetime_operation(training_data):
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    from fedcore.data.data import CompressionInputData
    from fedot.core.repository.tasks import Task,TaskTypesEnum
    source=nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,3))
    with torch.no_grad():source[0].weight.copy_(torch.diag(torch.tensor([100.]+[1.]*15)))
    state=deepcopy(source.state_dict())
    loader=DataLoader(TensorDataset(torch.randn(8,16),torch.tensor([0,1,2,0]*2)),batch_size=4)
    data=CompressionInputData(model=source,train_dataloader=loader,val_dataloader=loader,
        num_classes=3,input_dim=16,task=Task(TaskTypesEnum.classification))
    operation=LowRankModel({"device":"cpu","epochs":1,"learning_rate":0.,"rank_prune_each":-1,
        "non_adaptive_threshold":.9,"strategy":"explained_variance","log_each":0,"save_each":0,
        "criterion":"cross_entropy"})
    result=operation.fit(data)
    assert isinstance(result[0],IDecomposed)
    assert result[0].rank_pruning_info["rank"]==4
    assert result[0]._representation=="two_layers"
    assert result[0].U.numel()+result[0].Vh.numel()+result[0].bias.numel()<source[0].weight.numel()+source[0].bias.numel()
    assert result(torch.ones(2,16)).shape==(2,3)
    assert operation.model_before is not source
    for k,v in source.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)


class LocalFeatureModel(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(4,3)
    def forward(self,pixel_values,labels,**kwargs):
        logits=self.linear(pixel_values)
        return SimpleNamespace(logits=logits,loss=nn.functional.cross_entropy(logits,labels),
            attentions=(logits.unsqueeze(1),),hidden_states=(logits,))


def test_distillation_fit_uses_user_loss_and_intermediate_weights(training_data):
    from fedcore.losses.distilation_loss import KLLossSoft
    teacher=LocalFeatureModel();student=LocalFeatureModel()
    features,target=next(iter(training_data.train_dataloader))
    with torch.no_grad():
        t=teacher(pixel_values=features,labels=target)
        p=student(pixel_values=features,labels=target)
        expected=(3*p.loss + .2*KLLossSoft()(p.logits,t.logits) +
                  2*nn.functional.mse_loss(p.hidden_states[0],t.hidden_states[0]) +
                  .4*nn.functional.mse_loss(p.attentions[0],t.attentions[0]))
    batches=[{"pixel_values":features,"labels":target}]
    data=SimpleNamespace(model=teacher,train_dataloader=batches)
    operation=BaseDistilator({"epochs":1,"device":"cpu","student_model":student,"lr":.01,
        "loss_weight":3,"last_layer_loss_weight":.2,
        "distilation_params":{"intermediate_feat_layers_weights":[2],"intermediate_attn_layers_weights":[.4]}})
    original=deepcopy(teacher.state_dict())
    operation.fit(data)
    assert operation.history==pytest.approx([float(expected)])
    assert operation.distilation_params.loss_weight==3 and operation.distilation_params.last_layer_loss_weight==.2
    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match="Intermediate"):
        BaseDistilator({"epochs":1,"device":"cpu","intermediate_feat_layers_weights":[1]}).fit(training_data)
    with pytest.raises(ValueError,match="nonnegative"):
        BaseDistilator({"loss_weight":float("nan")})
