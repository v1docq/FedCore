"""Independent quantization, byte accounting and public export contracts."""
from copy import deepcopy
import io
import json
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.dobi_specs import DobiSVD
from fedcore.algorithm.low_rank.dobi_profiles import prepare_dobi,estimate_dobi_bytes
from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear,remapped_tensor_bytes
from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
from fedcore.algorithm.low_rank.method_specs import method_payload,parse_method
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
from fedcore.tools.registry.checkpoint_manager import CheckpointManager,CheckpointError,describe_model,build_model
from fedcore.tools.export import export_model
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def sample(out_features=7,in_features=4,*,dtype=torch.float64,bias=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(721)
        layer = nn.Linear(in_features,out_features,bias=bias,dtype=dtype)
        x = torch.randn(12,in_features,dtype=dtype)
    return layer,x


def oracle(weight,rank,tail_dtype,scale_dtype,compute_dtype):
    """Reference the specified scalar rounding rule, independent of decoder."""
    u,s,vh = torch.linalg.svd(weight.double(),full_matrices=False)
    m,n = weight.shape
    if m >= n:
        a,b = u[:,:rank]*s[:rank],vh[:rank]
        tail = a[n:].to(tail_dtype).to(compute_dtype)
        a = a[:n]
    else:
        a,b = u[:,:rank],s[:rank,None]*vh[:rank]
        tail = b[:,m:].to(tail_dtype).to(compute_dtype)
        b = b[:,:m]
    decoded = []
    for block,axis in ((a,0),(b,1)):
        scales = []
        codes = torch.empty_like(block,dtype=torch.int8)
        for component in range(rank):
            values = block[:,component] if axis==0 else block[component]
            maximum = float(values.abs().max())
            scale = torch.tensor(maximum/127 if maximum else 1.,dtype=scale_dtype)
            q = torch.round(values/scale.double()).clamp(-127,127).to(torch.int8)
            if axis==0:
                codes[:,component] = q
            else:
                codes[component] = q
            scales.append(scale)
        scale = torch.stack(scales).to(compute_dtype)
        decoded.append(codes.to(compute_dtype)*scale.unsqueeze(axis))
    a,b = decoded
    return (torch.cat((a,tail),0),b) if m>=n else (a,torch.cat((b,tail),1))


@pytest.mark.parametrize('shape',[(7,4),(4,7),(4,4)])
@pytest.mark.parametrize('tail_dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('rank',[1,4])
def test_codes_decode_and_forward_match_independent_scalar_oracle(shape,tail_dtype,rank):
    layer,x = sample(*shape)
    remap = RemappedLinear.from_weight(layer.weight,rank,layer.bias,
        tail_dtype=str(tail_dtype)[6:],scale_dtype='float64',compute_dtype='float64')
    a,b = oracle(layer.weight,rank,tail_dtype,torch.float64,torch.float64)
    left,right = remap.decoded_factors()
    torch.testing.assert_close(left@right,a@b,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(remap(x),F.linear(F.linear(x,b),a,layer.bias),rtol=1e-12,atol=1e-12)
    m,n = shape
    expected = 2*min(m,n)*rank + 2*abs(m-n)*rank + 16*rank + 8*m
    assert sum(v.numel()*v.element_size() for v in remap.state_dict().values()) == expected
    assert remapped_tensor_bytes(n,m,rank,tail_dtype=str(tail_dtype)[6:],
        scale_dtype='float64',bias_dtype='float64') == expected
    assert sum(p.numel() for p in remap.parameters()) == m
    assert all(v.dtype==torch.int8 for v in (remap.left_codes,remap.right_codes))
    assert not remap._modules


def test_zero_component_quantization_and_unbiased_input_gradients():
    w = torch.zeros(5,3,dtype=torch.float64)
    remap = RemappedLinear.from_weight(w,2,compute_dtype='float64')
    x = torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    assert torch.equal(remap(x),torch.zeros(2,5,dtype=x.dtype))
    remap(x).sum().backward()
    assert torch.equal(x.grad,torch.zeros_like(x))
    layer,x = sample()
    remap = RemappedLinear.from_weight(layer.weight,3,layer.bias,compute_dtype='float64')
    x.requires_grad_(True)
    gradient = torch.randn(12,7,dtype=torch.float64)
    remap(x).backward(gradient)
    a,b = remap.decoded_factors()
    torch.testing.assert_close(x.grad,gradient@a@b)


@pytest.mark.parametrize('option,value',[('remapping','fp16'),('remap_tail_dtype','int8'),
    ('remap_scale_dtype','float16'),('remap_compute_dtype','bfloat16'),('remap_tail_dtype',True)])
def test_precision_config_refuses_unsupported_options(option,value):
    with pytest.raises(MethodSpecError):
        DobiSVD(**{option:value})


def test_precision_config_roundtrip_and_backward_compatible_defaults():
    spec = DobiSVD(remapping='symmetric_int8_v1',remap_tail_dtype='bfloat16',
                   remap_scale_dtype='float64',remap_compute_dtype='float64')
    p = json.loads(json.dumps(method_payload(spec)))
    assert parse_method(p['method'],p['options'],version=p['version']) == spec
    assert parse_method('dobi_svd',{'remapping':'none'}) == DobiSVD()
    layer,_ = sample()
    costs = estimate_dobi_bytes(nn.Sequential(layer),['0'],spec)
    assert costs['remap_resvd_and_reconstruction_lower_bound_bytes'] >= 8*layer.weight.numel()
    assert all(type(value) is int and value>=0 for value in costs.values())


@pytest.mark.parametrize('weight,options',[
    (torch.tensor([[float('nan')]],dtype=torch.float64),{}),
    (torch.tensor([[float('inf')]],dtype=torch.float64),{}),
    (torch.tensor([[0.],[1e10]],dtype=torch.float64),{'tail_dtype':'float16'}),
    (torch.tensor([[1e-50]],dtype=torch.float64),{'scale_dtype':'float32'}),
    (torch.tensor([[1e40]],dtype=torch.float64),{'compute_dtype':'float32'}),
])
def test_nonfinite_underflow_and_overflow_refused_before_artifact(weight,options):
    with pytest.raises((NumericalDomainViolation,ValueError),match='finite|overflow|underflow'):
        RemappedLinear.from_weight(weight,1,**options)


@pytest.mark.parametrize('rank',[0,2,4])
@pytest.mark.parametrize('shape',[(7,4),(4,7),(4,4)])
def test_remapped_endpoints_calibrated_errors_and_real_bytes(rank,shape):
    layer,x = sample(*shape)
    source = deepcopy(layer)
    model = nn.Sequential(layer)
    objective = prepare_dobi(model,['0'],DobiSVD(initial_fraction=rank/4,remapping='symmetric_int8_v1',
        remap_compute_dtype='float64',remap_scale_dtype='float64'))
    record = objective.finalize(model,x,4)
    result = objective.model[0]
    assert type(result) is (ZeroLinear if rank==0 else RemappedLinear)
    local = record['layers']['0']['calibrated_output_error']
    mse = float((source(x)-result(x)).square().mean())
    assert local['total_mse'] == pytest.approx(mse,rel=1e-10,abs=1e-16)
    assert record['end_to_end_calibrated_output_error']['mse'] == pytest.approx(mse,rel=1e-10,abs=1e-16)
    if rank==4:
        assert local['truncation_mse'] == pytest.approx(0,abs=1e-25)
        assert local['remapping_mse'] > 0
    assert record['tensor_bytes'] == sum(v.numel()*v.element_size() for v in result.state_dict().values())
    assert not layer._forward_pre_hooks and not objective.layers['0']._forward_pre_hooks
    json.dumps(record,allow_nan=False)


def test_output_error_separates_exact_hard_projector_from_remap(monkeypatch):
    import fedcore.algorithm.low_rank.dobi_profiles as profiles
    layer,x = sample(4,4)
    exact = []
    original = profiles.fixed_output_factors
    def capture(weight,basis):
        pair = original(weight,basis)
        exact.append(pair.left@pair.right)
        return pair
    monkeypatch.setattr(profiles,'fixed_output_factors',capture)
    objective = prepare_dobi(nn.Sequential(layer),['0'],DobiSVD(remapping='symmetric_int8_v1',remap_compute_dtype='float64'))
    record = objective.finalize(objective.model,x,4)
    hard = F.linear(x,exact[0],layer.bias)
    decoded = objective.model(x)
    error = record['layers']['0']['calibrated_output_error']
    assert error['remapping_mse'] == pytest.approx(float((hard-decoded).square().mean()),rel=1e-11)
    assert error['truncation_mse'] == pytest.approx(float((layer(x)-hard).square().mean()),rel=1e-11)


def test_byte_rebudget_counts_scales_bias_and_untouched_model_before_svd(monkeypatch):
    layer,x = sample(7,4)
    source = nn.Sequential(layer,nn.Tanh(),nn.Linear(7,2,dtype=torch.float64))
    untouched = sum(v.numel()*v.element_size() for v in source[2].state_dict().values())
    target = untouched+remapped_tensor_bytes(4,7,1,bias_dtype='float64')
    objective = prepare_dobi(source,['0'],DobiSVD(initial_fraction=1.,remapping='symmetric_int8_v1'))
    record = objective.finalize(objective.model,x,4,tensor_byte_budget=target)
    assert record['tensor_bytes'] == target
    assert record['layers']['0']['rank'] == 1
    assert len(record['rank_reductions']) == 3
    assert record['unchanged_tensor_bytes'] == untouched
    minimum = untouched+7*8
    objective = prepare_dobi(deepcopy(nn.Sequential(layer,nn.Tanh(),nn.Linear(7,2,dtype=torch.float64))),
        ['0'],DobiSVD(initial_fraction=1.,remapping='symmetric_int8_v1'))
    def forbidden(*args,**kwargs):
        raise AssertionError('Infeasible byte floor must refuse before any SVD')
    monkeypatch.setattr(torch.linalg,'svd',forbidden)
    with pytest.raises(NumericalDomainViolation,match='unchanged tensors and biases'):
        objective.finalize(objective.model,x,4,tensor_byte_budget=minimum-1)


@pytest.mark.parametrize('tail_dtype',['float16','bfloat16'])
def test_strict_checkpoint_roundtrip_and_script_dynamic_batch(tail_dtype,tmp_path,monkeypatch):
    layer,x = sample()
    remap = RemappedLinear.from_weight(layer.weight,3,layer.bias,tail_dtype=tail_dtype)
    before = deepcopy(remap.state_dict())
    manager = CheckpointManager(str(tmp_path/'registry'),auto_cleanup=False)
    blob = manager.serialize_to_bytes(remap)
    payload = torch.load(io.BytesIO(blob),weights_only=True)
    restored = manager.restore(payload)
    assert restored.representation_config() == remap.representation_config()
    assert all(torch.equal(before[name],value) and before[name].dtype==value.dtype
               for name,value in restored.state_dict().items())
    assert torch.equal(restored(x),remap(x))
    def forbidden(*args,**kwargs):
        raise AssertionError('Deployment must never run SVD')
    monkeypatch.setattr(torch.linalg,'svd',forbidden)
    path = export_model(restored,'torchscript',tmp_path/'remap.pt',x[:2])
    with path.open('rb') as stream:
        compiled = torch.jit.load(stream)
    torch.testing.assert_close(compiled(x[:5]),restored(x[:5]),rtol=0,atol=0)


@pytest.mark.parametrize('corruption',['dtype','negative_scale','nan_tail','bad_code','scale_overflow','missing','extra'])
def test_checkpoint_state_corruption_rejected_without_mutating_supplied_model(corruption,tmp_path):
    layer,_ = sample()
    remap = RemappedLinear.from_weight(layer.weight,2,layer.bias)
    manager = CheckpointManager(str(tmp_path/'registry'),auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(remap)),weights_only=True)
    state = payload['state_dict']
    if corruption=='dtype': state['tail'] = state['tail'].float()
    if corruption=='negative_scale': state['left_scale'][0] = -1
    if corruption=='nan_tail': state['tail'][0,0] = float('nan')
    if corruption=='bad_code': state['left_codes'][0,0] = -128
    if corruption=='scale_overflow': state['left_scale'][0] = torch.finfo(torch.float32).max
    if corruption=='missing': del state['right_scale']
    if corruption=='extra': state['unknown'] = torch.tensor(1)
    before = deepcopy(remap.state_dict())
    with pytest.raises(CheckpointError): manager.restore(payload,model=remap)
    assert all(torch.equal(value,before[name]) for name,value in remap.state_dict().items())


@pytest.mark.parametrize('change',[lambda m:m.half(),lambda m:m.double(),lambda m:m.to(dtype=torch.float32)])
def test_generic_dtype_conversion_refuses_atomically_but_device_transfer_works(change):
    layer,_ = sample()
    remap = RemappedLinear.from_weight(layer.weight,2,layer.bias)
    before = deepcopy(remap.state_dict())
    with pytest.raises(ValueError,match='precision is fixed'): change(remap)
    assert all(torch.equal(value,before[name]) and value.dtype==before[name].dtype
               for name,value in remap.state_dict().items())
    assert remap.to('cpu') is remap


def test_checkpoint_refuses_modified_operator_and_unknown_descriptor():
    layer,_ = sample()
    remap = RemappedLinear.from_weight(layer.weight,2,layer.bias)
    description = describe_model(remap)
    description['config']['left_dtype']='uint8'
    with pytest.raises(CheckpointError): build_model(description)
    handle = remap.register_forward_hook(lambda *args:None)
    with pytest.raises(CheckpointError,match='unmodified'): describe_model(remap)
    handle.remove()
    remap.tail = remap.tail.float()
    with pytest.raises(CheckpointError,match='state'): describe_model(remap)


def test_public_training_remap_rebudgets_and_preserves_source(tmp_path):
    layer,x = sample(7,4)
    source = nn.Sequential(layer)
    snapshot = deepcopy(source.state_dict())
    split = TensorSplit(x,source(x).detach()+.1,tuple('train-'+str(i) for i in range(len(x))))
    def executor(model,stage,updates,objective):
        result = train_model(model,split,'regression',epochs=1,batch_size=4,learning_rate=.002,
            seed=71,batch_objective=objective,max_updates=updates)
        return dict(result,updates=result['training_steps'],stage=stage)
    budget = remapped_tensor_bytes(4,7,1,bias_dtype='float64')
    spec = DobiSVD(initial_fraction=1.,remapping='symmetric_int8_v1',remap_tail_dtype='bfloat16')
    result = transform_method(source,x,spec,target_paths=('0',),batch_size=4,
        recovery_executor=executor,tensor_byte_budget=budget)
    assert result.model[0].rank==1
    assert result.evidence['tensor_bytes_after']==budget
    assert result.evidence['parameters_after']==7
    assert result.evidence['buffer_bytes_after']==budget-7*8
    assert result.evidence['structure_training']['budget_denominator'].startswith('selected_original_weight_tensor_bytes')
    assert all(torch.equal(value,snapshot[name]) for name,value in source.state_dict().items())
    json.dumps(result.evidence,allow_nan=False)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_device_transfer_forward_and_input_backward():
    layer,x = sample()
    remap = RemappedLinear.from_weight(layer.weight,3,layer.bias,tail_dtype='bfloat16',compute_dtype='float32')
    expected = remap(x)
    gpu = deepcopy(remap).to('cuda')
    x_gpu = x.cuda().requires_grad_(True)
    y = gpu(x_gpu)
    torch.testing.assert_close(y.cpu(),expected,rtol=1e-5,atol=1e-6)
    y.square().sum().backward()
    assert bool(torch.isfinite(x_gpu.grad).all())
    assert gpu.tail.dtype==torch.bfloat16 and gpu.left_scale.dtype==torch.float32
    assert all(v.device.type=='cuda' for v in gpu.state_dict().values())


@pytest.mark.parametrize('shape',[(7,4),(4,7),(4,4)])
@pytest.mark.parametrize('rank',[1,4])
def test_float64_tail_owns_exact_storage_even_for_contiguous_and_empty_slices(shape,rank):
    layer,x = sample(*shape)
    remap = RemappedLinear.from_weight(layer.weight,rank,layer.bias,tail_dtype='float64',compute_dtype='float64')
    assert remap.tail.storage_offset()==0
    assert remap.tail.untyped_storage().nbytes()==remap.tail.numel()*8
    actual = sum(v.untyped_storage().nbytes() for v in remap.state_dict().values())
    expected = remapped_tensor_bytes(shape[1],shape[0],rank,tail_dtype='float64',bias_dtype='float64')
    assert actual==expected
    objective = prepare_dobi(nn.Sequential(layer),['0'],DobiSVD(initial_fraction=rank/4,
        remapping='symmetric_int8_v1',remap_tail_dtype='float64',remap_compute_dtype='float64'))
    record = objective.finalize(objective.model,x,4)
    assert record['tensor_bytes']==expected


@pytest.mark.parametrize('name',['decoded_factors','representation_config','validate_state'])
def test_checkpoint_rejects_instance_replacement_of_representation_methods(name):
    layer,_ = sample(4,4)
    remap = RemappedLinear.from_weight(layer.weight,4,layer.bias)
    setattr(remap,name,lambda *args,**kwargs:None)
    with pytest.raises(CheckpointError,match='unmodified'):
        describe_model(remap)


@pytest.mark.parametrize('kind',['within_layer','between_layers','unselected_buffer','padded_storage'])
def test_checkpoint_rejects_live_packed_storage_aliases_and_padding(kind):
    layer,_ = sample(4,4)
    first = RemappedLinear.from_weight(layer.weight,4,layer.bias)
    second = RemappedLinear.from_weight(layer.weight,4,layer.bias)
    model = nn.Sequential(first,second)
    if kind=='within_layer':
        first.right_codes = first.left_codes
    elif kind=='between_layers':
        second.left_codes = first.left_codes
    elif kind=='unselected_buffer':
        model.register_buffer('unselected',first.left_scale)
    else:
        first.left_codes = torch.zeros(5,4,dtype=torch.int8)[:4]
    with pytest.raises(CheckpointError,match='RemappedLinear'):
        describe_model(model)


@pytest.mark.parametrize('kind',['within_layer','between_layers','unselected_parameter','forged_topology'])
def test_checkpoint_rejects_forged_aliases_before_assign(kind,tmp_path):
    layer,_ = sample(4,4,dtype=torch.float32)
    first = RemappedLinear.from_weight(layer.weight,4,layer.bias)
    second = RemappedLinear.from_weight(layer.weight,4,layer.bias)
    model = nn.Sequential(first,second,nn.Linear(4,4))
    manager = CheckpointManager(str(tmp_path/'registry'),auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(model)),weights_only=True)
    state = payload['state_dict']
    if kind=='within_layer':
        state['0.right_codes'] = state['0.left_codes']
    elif kind=='between_layers':
        state['1.left_codes'] = state['0.left_codes']
    elif kind=='unselected_parameter':
        state['2.bias'] = state['0.left_scale']
    else:
        payload['topology']['buffer_aliases'] = [['0.left_scale','1.left_scale']]
    before = deepcopy(model.state_dict())
    with pytest.raises(CheckpointError,match='RemappedLinear'):
        manager.restore(payload,model=model)
    assert all(torch.equal(value,before[name]) for name,value in model.state_dict().items())


def test_local_error_sum_overflow_refuses_before_publication_and_restores_hooks():
    layer = nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad(): layer.weight.fill_(1e154)
    model = nn.Sequential(layer,nn.Tanh()).train()
    objective = prepare_dobi(model,['0'],DobiSVD(initial_fraction=0,remapping='symmetric_int8_v1'))
    with pytest.raises(NumericalDomainViolation,match='local calibration error'):
        objective.finalize(objective.model,torch.ones(16,1,dtype=torch.float64),16)
    assert not objective.finalized
    assert objective.model[0] is objective.layers['0']
    assert not objective.layers['0']._forward_pre_hooks
    assert not objective.layers['0'].export_mode
    assert model.training and model[1].training


@pytest.mark.parametrize('kwargs',[{'rank':-1},{'rank':5},{'in_features':False},
    {'out_features':0},{'tail_dtype':'int8'},{'scale_dtype':'float16'},
    {'bias_dtype':'bfloat16'},{'bias':'false'}])
def test_public_byte_cost_validates_dimensions_precision_and_bias(kwargs):
    options = {'in_features':4,'out_features':4,'rank':2}
    options.update(kwargs)
    with pytest.raises(ValueError): remapped_tensor_bytes(**options)


def test_public_byte_cost_zero_rank_and_no_bias():
    assert remapped_tensor_bytes(3,5,0,bias=False)==0
    assert remapped_tensor_bytes(3,5,0,bias_dtype='float64')==40
