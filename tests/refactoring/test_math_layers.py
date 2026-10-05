import copy
import pytest
import torch
from torch import nn
from fedcore.models.network_impl.decomposed_layers import DecomposedLinear,DecomposedConv1d,DecomposedConv2d,DecomposedEmbedding
from fedcore.algorithm.low_rank.rank_pruning import _apply_S_strategy,rank_threshold_pruning_in_place
from fedcore.algorithm.low_rank.svd_tools import decompose_module
from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
from fedcore.losses.low_rank_loss import OrthogonalLoss,HoyerLoss
from fedcore.losses.regularization_losses import AdaptiveRegularizationLoss,LaiMSE,LaiMAE
from fedcore.algorithm.pruning.sfp_tools import energy_filter_zeroing

@pytest.mark.parametrize('strategy',['explained_variance','absolute_sum','energy','quantile'])
@pytest.mark.parametrize('values',[[1.]*8,[100.,1.,1.,0.],[0.],[1.],[1e30,1e30]])
def test_rank_monotonic_and_boundaries(strategy,values):
    s=torch.tensor(values)
    ranks=[len(_apply_S_strategy(s,strategy,t,4)) for t in [.25,.75,.9,1.]]
    assert ranks==sorted(ranks)
    assert ranks[-1]>=torch.count_nonzero(s)
    if strategy=='explained_variance':
        for t,r in zip([.25,.75,.9,1.],ranks):
            masses=s.double().square()
            assert masses[:r].sum()>=t*masses.sum()

@pytest.mark.parametrize('mode',['one_layer','two_layers','three_layers',None])
@pytest.mark.parametrize('groups',[1,2,4])
@pytest.mark.parametrize('spatial',[False,True])
@pytest.mark.parametrize('padding_mode',['zeros','reflect'])
def test_conv2d_actual_backend_parity(mode,groups,spatial,padding_mode):
    torch.manual_seed(7)
    base=nn.Conv2d(4,8,(3,2),stride=(2,1),padding=(1,1),groups=groups,padding_mode=padding_mode).double()
    x=torch.randn(2,4,10,11,dtype=torch.double)
    layer=DecomposedConv2d(base,'spatial' if spatial else 'channel',compose_mode=mode)
    torch.testing.assert_close(layer(x),base(x),atol=1e-10,rtol=1e-10)
    assert OrthogonalLoss()(layer)<1e-20
    layer.compose_weight_for_inference()
    torch.testing.assert_close(layer(x),base(x),atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(to_standard_module(layer)(x),base(x),atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize('groups',[1,2,4])
def test_conv1d_path(groups):
    base=nn.Conv1d(4,8,3,padding=1,padding_mode='replicate',groups=groups,bias=False).double()
    model=nn.Sequential(copy.deepcopy(base));x=torch.randn(2,4,12,dtype=torch.double)
    decompose_module(model,decomposer_params={})
    torch.testing.assert_close(model(x),base(x))
    model[0].compose_mode='two_layers';model[0].compose_weight_for_inference()
    torch.testing.assert_close(model(x),base(x))


def test_bias_cost_independence_and_atomic_rejection():
    base=nn.Linear(8,8,bias=False).double();layer=DecomposedLinear(base)
    assert layer.bias is None and layer._evaluate_compose_mode()=='one_layer'
    with torch.no_grad():layer.U.add_(1)
    assert not torch.equal(layer._get_composed_weight(),base.weight)
    assert to_standard_module(layer).bias is None
    model=nn.Sequential(nn.Linear(8,8),nn.Embedding(4,3,sparse=True))
    original=model[0]
    with pytest.raises(ValueError,match='Sparse'):decompose_module(model)
    assert model[0] is original


def test_trained_sign_scale_operator_energy():
    l=DecomposedLinear(nn.Linear(8,8,bias=False).double())
    u=torch.eye(8,dtype=torch.double);s=torch.tensor([-10.,1,1,1,1,1,1,1],dtype=torch.double)
    l.set_U_S_Vh(u,s,u)
    other=copy.deepcopy(l);other.set_U_S_Vh(u*3,s/3,u)
    before=l.factor_matrix().detach().clone()
    for module in (l,other):rank_threshold_pruning_in_place(module,.9,round_to_times=1)
    assert l.S.numel()==1
    torch.testing.assert_close(l.factor_matrix(),other.factor_matrix())
    assert l.factor_matrix().square().sum()>=.9*before.square().sum()


def test_embedding_settings_max_norm_and_padding_gradient():
    base=nn.Embedding(6,4,padding_idx=2,max_norm=.8,scale_grad_by_freq=True).double()
    l=DecomposedEmbedding(base);x=torch.tensor([0,2,3,3])
    torch.testing.assert_close(l(x),base(x))
    torch.testing.assert_close(l(x),base(x))
    l(x).sum().backward();assert torch.isfinite(l.U.grad).all()
    assert torch.count_nonzero(l.U.grad[2])==0


def test_zero_regularizers_and_frozen():
    l=DecomposedLinear(nn.Linear(4,4))
    with torch.no_grad():l.S.zero_()
    loss=HoyerLoss()(l);assert loss==0;loss.backward();assert torch.isfinite(l.S.grad).all()
    l.U.requires_grad_(False);main=l(torch.randn(2,4)).square().mean()
    assert torch.isfinite(AdaptiveRegularizationLoss()(l,main))
    assert AdaptiveRegularizationLoss(0)(l,torch.tensor(2.))==0
    for cls in (LaiMSE,LaiMAE):
        with pytest.raises(ValueError):cls(0)

@pytest.mark.parametrize('threshold',[.25,.75,1.])
def test_filter_energy_contract(threshold):
    c=nn.Conv2d(1,4,1,bias=False)
    with torch.no_grad():c.weight.fill_(1)
    energy_filter_zeroing(c,threshold)
    assert c.weight.square().sum()>=4*threshold
    if threshold==1:assert torch.count_nonzero(c.weight)==4
    with torch.no_grad():c.weight.zero_()
    energy_filter_zeroing(c,threshold);assert torch.isfinite(c.weight).all()

@pytest.mark.parametrize('threshold',[float('nan'),float('inf'),0,-1,1.1])
def test_rank_invalid_threshold_is_atomic(threshold):
    layer=DecomposedLinear(nn.Linear(4,4));original=layer.factor_matrix().detach().clone()
    with pytest.raises(ValueError):rank_threshold_pruning_in_place(layer,threshold)
    torch.testing.assert_close(layer.factor_matrix(),original)

@pytest.mark.parametrize('backend',['rsvd','cur'])
def test_backend_canonicalization_contract(backend):
    layer=DecomposedLinear(nn.Linear(8,8).double(),decomposer=backend,decomposer_params={'rank':4})
    original=layer.factor_matrix().detach().clone()
    rank_threshold_pruning_in_place(layer,.75,round_to_times=1)
    assert layer.factor_matrix().square().sum()>=.75*original.square().sum()-1e-12


def test_frozen_training_modes_dtype_and_regularizer_order():
    base=nn.Conv2d(4,4,(3,2),groups=2,padding=(1,1)).double().eval();base.weight.requires_grad_(False)
    layer=DecomposedConv2d(base,'spatial');assert not layer.training
    assert all(not p.requires_grad for name,p in layer.named_parameters() if name!='bias')
    recreated=to_standard_module(layer);assert not recreated.training and not recreated.weight.requires_grad
    assert recreated.weight.dtype==torch.float64
    other=DecomposedLinear(nn.Linear(3,4).double())
    with torch.no_grad():other.U.add_(.1)
    torch.testing.assert_close(OrthogonalLoss()(nn.Sequential(layer,other)),OrthogonalLoss()(nn.Sequential(other,layer)))


def test_filter_threshold_one_preserves_tiny_nonzero_filters():
    layer=nn.Conv2d(1,2,1,bias=False).double()
    with torch.no_grad():layer.weight[:,0,0,0]=torch.tensor([1.,1e-150],dtype=torch.double)
    before=layer.weight.detach().clone()
    energy_filter_zeroing(layer,1.)
    torch.testing.assert_close(layer.weight,before,rtol=0,atol=0)


def test_rank_exact_first_reaching_independent_reference():
    generator=torch.Generator().manual_seed(73)
    for n in (1,2,8,21):
        spectrum=torch.rand(n,generator=generator,dtype=torch.double).sort(descending=True).values
        for strategy,power in (("absolute_sum",1),("explained_variance",2)):
            masses=[float(x)**power for x in spectrum]
            for threshold in (.25,.75,.9,1.):
                cumulative=0.
                expected=n
                for i,mass in enumerate(masses):
                    cumulative+=mass
                    if cumulative>=threshold*sum(masses):
                        expected=i+1;break
                assert len(_apply_S_strategy(spectrum,strategy,threshold,1))==expected


def test_root_layer_decomposition_returns_replacement():
    source=nn.Linear(4,3,bias=False).double()
    x=torch.randn(2,4,dtype=torch.double)
    result=decompose_module(source)
    assert isinstance(result,DecomposedLinear)
    assert type(source) is nn.Linear
    torch.testing.assert_close(result(x),source(x))
