import copy
import io
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from fedcore.losses.losses_impl import alpha_divergence,f_divergence
from fedcore.losses.distilation_loss import KLLossSoft,ReverseKLLossSoft,JSDLossSoft,CrossEntropyLossSoft,AlphaDivergenceLossSoft
from fedcore.models.network_modules.layers.lora import Linear,Embedding,Conv2d,apply_lora
from fedcore.tools.index_resolver import IndexResolvingParameter,wrap_parameters_with_resolver
from fedcore.metrics.pareto import ParetoMetrics
from fedcore.metrics.cv_metrics import Accuracy,Precision,F1,MASE,MAPE,R2
from fedcore.metrics.quality import MetricFactory,COMPUTATIONAL_METRICS
from fedcore.tools.ruler import PerformanceEvaluator,MeasurementUnavailable
from external.flatllmcore.core.absorption import ActivationCollector,AbsorptionCompressor

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('alpha',[0.,.25,.5,.75,1.])
@pytest.mark.parametrize('reduction',['none','mean','sum','batchmean'])
def test_distillation_reference_and_gradients(dtype,alpha,reduction):
    teacher=torch.tensor([[1000.,-1000.],[2.,-1.]],dtype=dtype,requires_grad=True)
    student=torch.tensor([[-1000.,1000.],[.1,2.]],dtype=dtype,requires_grad=True)
    loss=alpha_divergence(teacher,student,alpha,reduction)
    assert torch.isfinite(loss).all()
    loss.sum().backward();assert torch.isfinite(student.grad).all() and student.grad.abs().sum()>0
    assert teacher.grad is None
    identical=alpha_divergence(teacher,teacher,alpha,reduction)
    torch.testing.assert_close(identical,torch.zeros_like(identical),atol=2e-6,rtol=0)
    if alpha==0:
        q=teacher.detach().log_softmax(-1);p=student.detach().log_softmax(-1)
        reference=(q.exp()*(q-p)).sum(-1)
        if reduction in ('mean','batchmean'):reference=reference.mean()
        elif reduction=='sum':reference=reference.sum()
        torch.testing.assert_close(loss.detach(),reference)


def test_module_losses_and_fdiv_student_direction():
    teacher=torch.randn(3,4,requires_grad=True);student=torch.randn(3,4,requires_grad=True)
    for cls in (KLLossSoft,ReverseKLLossSoft,JSDLossSoft):
        value=cls()(student,teacher);assert value.ndim==0 and torch.isfinite(value)
        value.backward(retain_graph=True)
    _,loss=f_divergence(teacher,student,.5);loss.sum().backward()
    assert student.grad.abs().sum()>0 and teacher.grad is None
    assert KLLossSoft()(teacher,teacher).abs()<1e-6
    AlphaDivergenceLossSoft()(student,teacher,.5).backward()

@pytest.mark.parametrize('layer,cls,input',[
    (nn.Linear(4,3),Linear,torch.randn(2,4)),
    (nn.Embedding(6,3,padding_idx=1),Embedding,torch.tensor([0,1,2])),
    (nn.Conv2d(4,6,(3,2),padding=(1,1),dilation=(1,2),groups=2,padding_mode='reflect'),Conv2d,torch.randn(2,4,8,8))])
def test_lora_creation_independence_and_merge(layer,cls,input):
    original=copy.deepcopy(layer)
    adapter=cls(layer,'one',r=2)
    torch.testing.assert_close(adapter(input),layer(input))
    with torch.no_grad():
        if isinstance(layer,nn.Embedding):adapter.lora_embedding_B['one'].fill_(.1)
        else:adapter.lora_B['one'].weight.fill_(.1)
    adapter.eval();expected=adapter(input)
    adapter.merge(safe_merge=True);torch.testing.assert_close(adapter(input),expected)
    adapter.merge();torch.testing.assert_close(adapter(input),expected)
    adapter.unmerge();torch.testing.assert_close(adapter(input),expected)
    torch.testing.assert_close(adapter.base_layer.weight,original.weight,rtol=0,atol=0)
    torch.testing.assert_close(layer(input),original(input))
    assert layer.weight.data_ptr()!=adapter.base_layer.weight.data_ptr()


def test_apply_lora_and_resolver_state_roundtrip():
    source=nn.Sequential(nn.Linear(4,4));adapted=apply_lora(source,rank=2)
    assert isinstance(adapted[0],Linear) and isinstance(source[0],nn.Linear)
    p=IndexResolvingParameter(nn.Parameter(torch.randn(3,2),requires_grad=False),aggregation_mode='intersect')
    assert not p.requires_grad and not hasattr(p,'unknown_attribute')
    for indices in ([0],[1],[2]):p.resolve_indices(indices)
    assert p.current_to_original=={}
    cloned=copy.deepcopy(p);assert cloned.resolver_state()==p.resolver_state() and not cloned.requires_grad
    buffer=io.BytesIO();torch.save(p,buffer);buffer.seek(0);loaded=torch.load(buffer,weights_only=False)
    assert isinstance(loaded,IndexResolvingParameter) and loaded.resolver_state()==p.resolver_state()
    wrapped=wrap_parameters_with_resolver(source,inplace=False)
    assert isinstance(wrapped[0].weight,IndexResolvingParameter) and type(source[0].weight)==nn.Parameter
    assert wrap_parameters_with_resolver(wrapped)[0].weight is wrapped[0].weight


def test_pareto_against_independent_reference():
    torch.manual_seed(9)
    for directions in (True,False,[True,False,True]):
        points=torch.randint(0,4,(20,3))
        direction=[directions]*3 if isinstance(directions,bool) else directions
        expected=[]
        for a in points.tolist():
            dominated=False
            for b in points.tolist():
                no_worse=all(y>=x if d else y<=x for x,y,d in zip(a,b,direction))
                better=any(y>x if d else y<x for x,y,d in zip(a,b,direction))
                dominated|=no_worse and better
            expected.append(not dominated)
        actual=ParetoMetrics().pareto_metric_list(points,directions)
        assert actual.tolist()==expected
        permutation=torch.randperm(len(points))
        assert torch.equal(actual[permutation],ParetoMetrics().pareto_metric_list(points[permutation],directions))
    assert ParetoMetrics().pareto_metric_list([[1,0],[0,1]]).all()
    assert ParetoMetrics().pareto_metric_list([]).numel()==0
    with pytest.raises(ValueError):ParetoMetrics().pareto_metric_list([[float('nan'),1]])


def test_legacy_metrics_and_training_scale():
    t=torch.tensor([0,1,2]);p=torch.tensor([0,1,0])
    assert F1.metric(t,t)==1 and F1.metric(t,p)<1
    assert Precision.metric(t,p)<1 and Accuracy.metric(t,p)==pytest.approx(2/3)
    assert F1.metric(t*3+7,p*3+7)==F1.metric(t,p)
    scale=MASE.training_scale(torch.tensor([1.,3.,5.]))
    assert MASE.metric(torch.tensor([4.,4.]),torch.tensor([2.,2.]),training_scale=scale)==1
    with pytest.raises(ValueError):MASE.metric(torch.ones(2),torch.zeros(2))
    assert MAPE.metric(torch.zeros(2),torch.zeros(2))==0
    assert MAPE.metric(torch.zeros(2),torch.ones(2))==float('inf')
    assert R2.metric(torch.ones(2),torch.ones(2))==1
    assert R2.metric(torch.ones(2),torch.zeros(2))==0


def test_metrics_registry_batch_limits_and_energy():
    loader=DataLoader(TensorDataset(torch.ones(5,2),torch.ones(5)),batch_size=1)
    pe=PerformanceEvaluator(nn.Linear(2,1),data=loader,n_batches=1,warmup_batches=0)
    assert len(list(pe._generate_example_batch(1)))==1
    assert len(list(pe._generate_example_batch(2)))==2
    pe.measure_latency();assert pe.measurement_info['latency']['measured_batches']==1
    with pytest.raises(MeasurementUnavailable):pe.measure_energy()
    times=iter([10.,12.]);pe._clock=lambda:next(times);pe._power_reader=lambda:5.
    energy,elapsed,power=pe._eval_single_power(torch.ones(1,2),torch.device('cpu'))
    assert energy==10 and elapsed==2 and power==5
    for name,(executor,unit,minimize) in COMPUTATIONAL_METRICS.items():
        metric=MetricFactory.get_metric(name)
        assert metric.executor==executor and metric.unit==unit and metric.need_to_minimize==minimize
        assert hasattr(PerformanceEvaluator,executor)


class ToyMLP(nn.Module):
    def __init__(self,d=2):
        super().__init__();self.up_proj=nn.Linear(1,d,bias=False);self.gate_proj=nn.Linear(1,d,bias=False);self.down_proj=nn.Linear(d,1,bias=False)
    def forward(self,x):return self.down_proj(torch.sigmoid(self.gate_proj(x))*self.up_proj(x))


def toy_compressor(d=2,tolerance=.99):
    layer=nn.Module();layer.mlp=ToyMLP(d)
    model=nn.Module();model.model=nn.Module();model.model.layers=nn.ModuleList([layer])
    return AbsorptionCompressor(model,tolerance=tolerance,device='cpu')


def test_nystrom_duplicates_actual_down_restoration():
    compressor=toy_compressor()
    mlp=compressor.model.model.layers[0].mlp
    with torch.no_grad():
        mlp.up_proj.weight.fill_(1);mlp.gate_proj.weight.fill_(1);mlp.down_proj.weight.fill_(1)
    x=torch.tensor([[1.],[2.]],dtype=torch.float32)
    expected=mlp(x).detach()
    z=(torch.sigmoid(mlp.gate_proj(x))*mlp.up_proj(x)).detach()
    collector=ActivationCollector(mlp.up_proj,1,.99,'cpu');collector.add_activation(z);collector.compute_eigenvectors()
    compressor.collectors['layer_0.mlp.up_proj']=collector
    result=compressor.apply_absorption_mlp(0,.5)
    torch.testing.assert_close(mlp(x),expected)
    assert result['rank']==1 and result['retained_calibration_energy']==pytest.approx(1)
    assert float(mlp.down_proj.weight[0,0].detach())==pytest.approx(2)


def test_nystrom_tolerance_rejects_identity_without_mutation():
    compressor=toy_compressor(4)
    mlp=compressor.model.model.layers[0].mlp
    col=ActivationCollector(mlp.up_proj,1,.99,'cpu');col.add_activation(torch.eye(4));col.compute_eigenvectors()
    compressor.collectors['layer_0.mlp.up_proj']=col
    original=mlp.down_proj.weight.clone()
    with pytest.raises(ValueError,match='Unattainable'):compressor.apply_absorption_mlp(0,.5)
    torch.testing.assert_close(original,mlp.down_proj.weight)
    zero=ActivationCollector(mlp.up_proj,1,.99,'cpu');zero.add_activation(torch.zeros(2,4));assert zero.compute_eigenvectors()==1


def test_pareto_public_entries_and_legacy_discovery():
    from fedcore.metrics.quality import ParetoMetrics as QualityPareto
    from fedcore.metrics.cv_metrics import ParetoMetrics as LegacyPareto
    assert QualityPareto is LegacyPareto
    assert QualityPareto().pareto_metric_list([[1,0],[0,1]]).tolist()==[True,True]
    assert MetricFactory.get_metric('LegacyF1') is F1
    assert MetricFactory.get_metric('LegacyMASE') is MASE


def test_pareto_unsigned_and_large_integer_objectives():
    for points in (torch.tensor([[0,255],[255,0],[0,0]],dtype=torch.uint8),
                   torch.tensor([[-2**63,2**63-1],[2**63-1,-2**63]],dtype=torch.int64)):
        result=ParetoMetrics().pareto_metric_list(points,[False,True])
        assert result.tolist()==[True]+[False]*(len(points)-1)


def test_resolver_union_empty_duplicates_and_metadata_roundtrip():
    p=IndexResolvingParameter(nn.Parameter(torch.ones(4,2)),aggregation_mode="union")
    p.resolve_indices([])
    p.resolve_indices([2,2])
    p.resolve_indices([0])
    assert p.current_to_original=={0:0,1:2}
    cloned=IndexResolvingParameter(torch.zeros(4,2),aggregation_mode="union")
    cloned.load_resolver_state(p.resolver_state())
    assert cloned.resolver_state()==p.resolver_state()
    cloned.resolve_indices([3])
    assert p.current_to_original=={0:0,1:2}
    with pytest.raises(ValueError):p.resolve_indices([9])
