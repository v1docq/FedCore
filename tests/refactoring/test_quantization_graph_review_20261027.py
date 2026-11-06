"""Independent regressions from the 2026-10-27 PR166 review."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure, StructuralInvalid
from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig
from fedcore.algorithm.low_rank.compression_grammar import MutationKind
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair
from fedcore.algorithm.quantization.graph_specs import Method, TransformSpec, OperatorRecipe, QuantizationGraphRequest, recipe_graph
from fedcore.algorithm.quantization.graph_execution import ExportVerifiedArtifact
from fedcore.algorithm.quantization.grammar import (prepare_quantization_space, generate_genotype, normalize_genotype,
    crossover_genotypes, mutate_genotype, compile_genotype, QuantizationGenotype, QuantizationGene, run_quantization_search)
from fedcore.algorithm.quantization.program import (RecoveryStageSpec, QuantizationProgram, ProgramArtifact,
    quantize, training_identity, check_quantization_program)
from fedcore.algorithm.quantization.training import BackendStageSpec
from fedcore.algorithm.quantization.catalogue import (prepare_quantization_catalogue, QuantizationCatalogue,
    allocate_quantization_catalogue, assess_quantization_selection, AssessedArtifact)
from fedcore.algorithm.quantization.resources import ResourceLedger, StageBudgets
from fedcore.algorithm.quantization.checkpoint import validate_descriptor_budget, build_quantized
from fedcore.algorithm.quantization.portable_layers import PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear
from fedcore.algorithm.quantization.matrix_transforms import scalar_ldl
from fedcore.algorithm.quantization.tensor_formats import QuantizedTensorSpec
from fedcore.algorithm.quantization.tender import TenderLinear


def training_space():
    model=nn.Sequential(nn.Linear(3,3),nn.Linear(3,2))
    x,y=torch.randn(4,3),torch.randn(4,2)
    dense,factors=TransformSpec(Method.DENSE),TransformSpec(Method.DSVD,rank=1)
    backend=BackendStageSpec('qat',('0','1'),training_steps=1)
    recovery=RecoveryStageSpec(('0','1'),steps=1)
    space=prepare_quantization_space(model,{'0':(dense,factors),'1':(dense,factors)},calibration=x,
        evaluation_context='review-training-closure',training_choices=(backend,recovery),training_batches=((x,y),))
    parents=tuple(QuantizationGenotype(tuple(QuantizationGene(d.aliases,spec,recipe_graph(spec)) for d in space.domains),training=stage)
                  for spec,stage in ((dense,backend),(factors,recovery)))
    return space,parents


@settings(max_examples=12,deadline=None)
@given(st.integers(0,2**32-1))
def test_global_training_constraints_remain_closed_under_all_operators(seed):
    space,parents=training_space()
    assert all(normalize_genotype(p,space)==p for p in parents)
    children=crossover_genotypes(*parents,space,seed=seed)
    assert children==parents  # A one-region exchange cannot satisfy either global stage.
    candidates=(*children,generate_genotype(space,seed=seed),
        mutate_genotype(parents[0],space,kind=MutationKind.CHOICE,seed=seed))
    for candidate in candidates:
        assert normalize_genotype(candidate,space)==candidate
        methods={gene.transform.method for gene in candidate.genes}
        assert methods==({Method.DENSE} if candidate.training==parents[0].training else {Method.DSVD})


@pytest.mark.parametrize('changes',[{'learning_rate':float('nan')},{'learning_rate':-1},{'learning_rate':'bad'},
    {'training_id':''},{'paths':('0','0')}])
def test_invalid_recovery_configuration_is_rejected_before_compression(changes):
    model=nn.Sequential(nn.Linear(3,2));x,y=torch.randn(4,3),torch.randn(4,2)
    stage=replace(RecoveryStageSpec(('0',),steps=1),**changes)
    request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DSVD,rank=1)),))
    program=QuantizationProgram(request,recovery=stage,training_fingerprint=training_identity(((x,y),)))
    result=quantize(model,program,calibration=x,training_batches=((x,y),))
    assert isinstance(result,StructuralInvalid),result
    space=prepare_quantization_space(model,{'0':(request.recipes[0].transform,)},calibration=x,
        evaluation_context='review-malformed-stage',training_choices=(stage,),training_batches=((x,y),))
    assert isinstance(space,ValidationFailure),space


@pytest.mark.parametrize('stage',[RecoveryStageSpec(('0',),steps=1),BackendStageSpec('qat',('0',),training_steps=1)])
def test_training_grammar_rejects_missing_ordered_training_data(stage):
    model=nn.Sequential(nn.Linear(3,2))
    spec=TransformSpec(Method.DSVD,rank=1) if type(stage) is RecoveryStageSpec else TransformSpec(Method.DENSE)
    space=prepare_quantization_space(model,{'0':(spec,)},calibration=torch.randn(4,3),
        evaluation_context='review-missing-training',training_choices=(stage,))
    assert isinstance(space,ValidationFailure),space


@pytest.mark.parametrize('calibration',[None,{'0':torch.ones(4,3)}])
def test_backend_program_and_grammar_require_root_model_calibration(calibration):
    model=nn.Sequential(nn.Linear(3,2))
    spec=TransformSpec(Method.DENSE)
    stage=BackendStageSpec('static',('0',))
    program=QuantizationProgram(QuantizationGraphRequest((OperatorRecipe('0',spec),)),backend=stage)
    result=quantize(model,program,calibration=calibration)
    assert isinstance(result,StructuralInvalid),result
    assert any(v.code=='BackendCalibrationCoordinates' for v in result.violations)
    space=prepare_quantization_space(model,{'0':(spec,)},calibration=calibration,
        evaluation_context='review-backend-calibration',training_choices=(stage,))
    assert isinstance(space,ValidationFailure),space


def test_root_calibration_and_operator_rows_have_distinct_coordinate_identities():
    from collections import OrderedDict
    from fedcore.algorithm.quantization.graph_execution import calibration_identity,check_quantization_graph,execute_quantization_graph
    inputs=torch.ones(2,2)
    rows={'model_input':inputs}
    assert calibration_identity(inputs).fingerprint!=calibration_identity(rows).fingerprint
    model=nn.Sequential(OrderedDict((('model_input',nn.Linear(2,2)),)))
    request=QuantizationGraphRequest((OperatorRecipe('model_input',TransformSpec(Method.PTQ)),))
    checked=check_quantization_graph(model,request,calibration=inputs)
    result=execute_quantization_graph(model,checked,calibration=rows)
    assert isinstance(result,StructuralInvalid) and result.violations[0].code=='StalePlan',result


def test_alias_recovery_uses_the_same_declared_region_after_canonicalization():
    layer=nn.Linear(3,3);model=nn.Sequential(layer,layer)
    x,y=torch.randn(4,3),torch.randn(4,3);batches=((x,y),)
    stage=RecoveryStageSpec(('1',),steps=1)
    space=prepare_quantization_space(model,{'1':(TransformSpec(Method.DSVD,rank=1),)},calibration=x,
        evaluation_context='review-alias-stage',training_choices=(stage,),training_batches=batches)
    genotype=generate_genotype(space)
    workflow=compile_genotype(genotype,space)
    assert not isinstance(check_quantization_program(model,workflow.program,calibration=x,training_batches=batches),ValidationFailure)
    result=quantize(model,workflow.program,calibration=x,training_batches=batches)
    assert isinstance(result,ProgramArtifact),result
    assert result.model[0] is result.model[1]


def test_ready_factor_catalogue_without_calibration_uses_original_factor_product():
    left=torch.tensor([[1.,.3],[.1,-.2],[.4,.7]],dtype=torch.float64)
    right=torch.tensor([[.2,-.8],[.5,.1]],dtype=torch.float64)
    source=FactorizedLinear.from_factors(FactorPair(left,right,{}),None)
    model=nn.Sequential(source)
    spec=TransformSpec(Method.FACTORS)
    catalogue=prepare_quantization_catalogue(model,{'0':(spec,)})
    assert isinstance(catalogue,QuantizationCatalogue),catalogue
    packed=PortableQuantizedFactorLinear.from_factors(left,right)
    expected=float((left@right-packed.left.decode()@packed.right.decode()).square().sum())
    assert catalogue.entries[0].local_error==expected
    allocated=allocate_quantization_catalogue(catalogue,10000)
    result=assess_quantization_selection(model,catalogue,allocated,evaluate=lambda _:0.,evaluation_context='review-ready-factors')
    assert isinstance(result,AssessedArtifact),result


@pytest.mark.parametrize('explicit_limit',[None,512*1024**2,4*1024**3])
def test_program_rss_limit_remains_active_after_compression(monkeypatch,explicit_limit):
    import fedcore.algorithm.quantization.program as program_module
    import fedcore.algorithm.quantization.resources as resources_module
    request_limit=1024**3
    rss=[100*1024**2]
    monkeypatch.setattr(resources_module.psutil,'Process',lambda *a,**k:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss[0])))
    original=program_module.compress_quantization_graph
    def compressed(*args,**kwargs):
        result=original(*args,**kwargs)
        assert isinstance(result,ExportVerifiedArtifact),result
        rss[0]=2*1024**3
        return result
    monkeypatch.setattr(program_module,'compress_quantization_graph',compressed)
    ledger=ResourceLedger(StageBudgets(max_rss_bytes=explicit_limit)) if explicit_limit else None
    model=nn.Sequential(nn.Linear(2,2))
    program=QuantizationProgram(QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DENSE)),),max_peak_bytes=request_limit))
    result=quantize(model,program,ledger=ledger)
    assert isinstance(result,ResourceFailure) and result.code==('RSSBudget' if explicit_limit is not None and explicit_limit<request_limit else 'ResourceLimitExceeded'),result
    if ledger is not None:assert ledger.budgets.max_rss_bytes==explicit_limit


def backend_description(constants=None,operators=None):
    return {'type':'QuantizationGraph','schema':'quantization_graph_v1','kind':'RegisteredBackendGraph',
        'config':{'instructions':[{'name':'x','op':'placeholder','target':'x','args':[]},
            {'name':'out','op':'output','target':'output','args':[{'node':'x'}]}],
            'operators':operators or {},'constants':constants or {}}}


def test_backend_descriptor_budget_sums_constants_and_float_operators_before_allocation(monkeypatch):
    description=backend_description({'one':{'shape':[2],'dtype':'float64'},'two':{'shape':[2],'dtype':'int64'}},
        {'float':{'type':'Linear','args':{'in_features':2,'out_features':2,'bias':True}}})
    validate_descriptor_budget(description,max_bytes=56)
    with pytest.raises(ValueError,match='allocation budget'):validate_descriptor_budget(description,max_bytes=55)
    huge=backend_description({'c'+str(i):{'shape':[2**20],'dtype':'float64'} for i in range(17)})
    monkeypatch.setattr(torch,'zeros',lambda *a,**k:pytest.fail('Allocated before aggregate descriptor admission'))
    with pytest.raises(ValueError,match='allocation budget'):build_quantized(huge)


@pytest.mark.parametrize('power,input_power,bias,expected',[(30,0,1.,1.),(80,60,0.,0.)])
def test_sum_operator_accumulates_branches_and_one_bias_before_output_cast(power,input_power,bias,expected):
    base=PortableQuantizedLinear.from_weight(torch.tensor([[2.**power]],dtype=torch.float64),torch.tensor([bias],dtype=torch.float64))
    correction=PortableQuantizedFactorLinear.from_factors(torch.tensor([[-2.**power]],dtype=torch.float64),torch.ones(1,1,dtype=torch.float64))
    layer=QuantizedSumLinear(base,correction)
    assert torch.equal(layer(torch.tensor([[2.**input_power]],dtype=torch.float32)),torch.tensor([[expected]],dtype=torch.float32))


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_ldl_retains_small_coordinate_pivots_of_a_supported_psd_direction(scale):
    vector=torch.tensor([1e-6,1.],dtype=torch.float64)
    moment=torch.outer(vector,vector)*scale
    l,d=scalar_ldl(moment,1e-10)
    assert d[0]>0 and float(l[1,0])==pytest.approx(1e6)
    torch.testing.assert_close((l*d.sqrt())@(l*d.sqrt()).T,moment,rtol=1e-14,atol=scale*1e-30)
    with pytest.raises(ValueError,match='PSD'):scalar_ldl(torch.tensor([[1.,2.],[2.,1.]],dtype=torch.float64)*scale,1e-10)


@pytest.mark.parametrize('role',['source','calibration','training'])
@pytest.mark.parametrize('event_name',['evaluated','stopped'])
def test_real_optimizer_invalidates_all_fitness_after_live_context_changes(role,event_name):
    model=nn.Sequential(nn.Linear(3,2));x,y=torch.randn(4,3),torch.randn(4,2);batches=((x,y),)
    space=prepare_quantization_space(model,{'0':(TransformSpec(Method.DENSE),)},calibration=x,
        evaluation_context='review-live-context',training_batches=batches)
    initial=generate_genotype(space)
    changed=[]
    def trace(event):
        if event['event']==event_name and not changed:
            with torch.no_grad():
                {'source':model[0].weight,'calibration':x,'training':y}[role].add_(1)
            changed.append(True)
    result=run_quantization_search(model,space,GrammarSearchConfig(population_size=2,generations=1,max_evaluations=2,max_seconds=30,max_proposals=6),
        calibration=x,training_batches=batches,evaluate=lambda _:0.,initial_genotypes=(initial,initial),trace_callback=trace)
    assert changed
    assert result['status']=='failed' and result['stop_reason']=='context_invalidated',result
    assert result['best_genotypes']==[] and result['finite_archive']==[]
    assert result['context_failure']['violations'][0]['code']=={'source':'StaleSource','calibration':'StaleCalibration','training':'StaleTraining'}[role]


@pytest.mark.parametrize('mode',['static','qat'])
@pytest.mark.parametrize('regions',[('2',),('0','2')])
def test_regional_backend_resolves_shared_module_aliases_and_preserves_source(mode,regions):
    layer=nn.Linear(3,3);model=nn.Sequential(layer,nn.ReLU(),layer)
    before={name:value.clone() for name,value in model.state_dict().items()}
    x,y=torch.randn(6,3),torch.randn(6,3);batches=((x,y),)
    request=QuantizationGraphRequest((OperatorRecipe('2',TransformSpec(Method.DENSE)),))
    program=QuantizationProgram(request,backend=BackendStageSpec(mode,regions,training_steps=1),
        training_fingerprint=training_identity(batches))
    result=quantize(model,program,calibration=x,training_batches=batches)
    assert isinstance(result,ProgramArtifact),result
    assert result.evidence['training_stages'][0]['regions']==regions
    assert result.evidence['training_stages'][0]['canonical_regions']==('0',)
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
    assert model[0] is model[2]


@pytest.mark.parametrize('mode',['static','qat'])
def test_root_dense_regional_backend_is_an_explicit_owned_linear_boundary(mode):
    model=nn.Linear(2,2)
    before={name:value.clone() for name,value in model.state_dict().items()}
    x,y=torch.randn(6,2),torch.randn(6,2);batches=((x,y),)
    request=QuantizationGraphRequest((OperatorRecipe('',TransformSpec(Method.DENSE)),))
    stage=BackendStageSpec(mode,('',),training_steps=1)
    space=prepare_quantization_space(model,{'':(TransformSpec(Method.DENSE),TransformSpec(Method.SVD,rank=1))},calibration=x,
        evaluation_context='review-root-dense-backend',training_choices=(stage,),training_batches=batches)
    workflow=compile_genotype(generate_genotype(space),space)
    result=quantize(model,workflow.program,calibration=x,training_batches=batches)
    assert isinstance(result,ProgramArtifact),result
    assert result.evidence['training_stages'][0]['regions']==('',)
    assert result.evidence['training_stages'][0]['execution_regions']==('0',)
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())


@pytest.mark.parametrize('region,expected_method',[('0',Method.DENSE),('0.left',Method.SVD),('0.right',Method.SVD)])
def test_backend_generation_matches_explicit_dense_or_factor_boundary(region,expected_method):
    model=nn.Sequential(nn.Linear(3,2));x=torch.randn(4,3)
    stage=BackendStageSpec('static',(region,))
    space=prepare_quantization_space(model,{'0':(TransformSpec(Method.DENSE),TransformSpec(Method.SVD,rank=1))},
        calibration=x,evaluation_context='review-backend-boundary',training_choices=(stage,))
    genotype=generate_genotype(space,seed=2)
    assert genotype.genes[0].transform.method is expected_method
    workflow=compile_genotype(genotype,space)
    assert not isinstance(check_quantization_program(model,workflow.program,calibration=x),ValidationFailure)
    result=quantize(model,workflow.program,calibration=x)
    assert isinstance(result,ProgramArtifact),result


def test_root_svd_explicit_factor_boundary_and_direct_refusal():
    model=nn.Linear(3,2);x=torch.randn(4,3)
    request=QuantizationGraphRequest((OperatorRecipe('',TransformSpec(Method.SVD,rank=1)),))
    rejected=check_quantization_program(model,QuantizationProgram(request,backend=BackendStageSpec('static',('',))),calibration=x)
    assert isinstance(rejected,ValidationFailure)
    stage=BackendStageSpec('static',('left','right'))
    space=prepare_quantization_space(model,{'':(TransformSpec(Method.DENSE),request.recipes[0].transform)},calibration=x,
        evaluation_context='review-root-factors',training_choices=(stage,))
    workflow=compile_genotype(generate_genotype(space),space)
    assert workflow.genotype.genes[0].transform.method is Method.SVD
    assert not isinstance(check_quantization_program(model,workflow.program,calibration=x),ValidationFailure)
    result=quantize(model,QuantizationProgram(request,backend=stage),calibration=x)
    assert isinstance(result,ProgramArtifact),result


def test_program_checks_rss_after_its_final_checkpoint_roundtrip(monkeypatch):
    import fedcore.algorithm.quantization.resources as resources_module
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    rss=[100*1024**2]
    monkeypatch.setattr(resources_module.psutil,'Process',lambda *a,**k:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss[0])))
    original=CheckpointManager.deserialize_from_bytes
    reads=[]
    def loaded(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        reads.append(True)
        if len(reads)==2:rss[0]=2*1024**3
        return result
    monkeypatch.setattr(CheckpointManager,'deserialize_from_bytes',loaded)
    request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DENSE)),),max_peak_bytes=1024**3)
    result=quantize(nn.Sequential(nn.Linear(2,2)),QuantizationProgram(request))
    assert len(reads)==2
    assert isinstance(result,ResourceFailure) and result.code=='ResourceLimitExceeded',result


@pytest.mark.parametrize('maximum',[49.,98.,1e-309,1e-310])
def test_tender_anchor_channel_and_subnormal_scale_remain_in_first_group(maximum):
    weight=torch.ones(1,1,dtype=torch.float64)
    rows=torch.tensor([[-maximum],[maximum]],dtype=torch.float64)
    layer=TenderLinear.calibrate(weight,None,rows)
    assert layer.exponents.tolist()==[0]
    assert layer.activation_codes(rows).flatten().tolist()==[-127,127]
    torch.testing.assert_close(layer(rows),rows,rtol=1e-11,atol=0)


def test_tender_half_maximum_and_its_adjacent_values_use_exact_threshold():
    half=torch.tensor(24.5,dtype=torch.float64)
    smaller=torch.nextafter(half,torch.tensor(0.,dtype=torch.float64))
    larger=torch.nextafter(half,torch.tensor(float('inf'),dtype=torch.float64))
    ranges=torch.stack((torch.tensor(49.,dtype=torch.float64),half,smaller,larger))
    layer=TenderLinear.calibrate(torch.ones(1,4,dtype=torch.float64),None,torch.stack((-ranges,ranges)))
    assert layer.exponents.tolist()==[1,0]
    assert layer.activation_indices.tolist()==[0,3,1,2]
    assert layer.group_ptr.tolist()==[0,2,4]


@pytest.mark.parametrize('weight_value,input_value',[(1e-308,1e308),(1e308,1e-308),(1.,1e308)])
def test_tender_three_factor_scale_product_preserves_representable_result(weight_value,input_value):
    weight=torch.tensor([[weight_value]],dtype=torch.float64)
    rows=torch.tensor([[-input_value],[input_value]],dtype=torch.float64)
    layer=TenderLinear.calibrate(weight,None,rows)
    expected=rows*weight_value
    assert bool(torch.isfinite(layer(rows)).all())
    torch.testing.assert_close(layer(rows),expected,rtol=1e-11,atol=0)


def test_tender_scaled_integer_product_retains_subnormal_result():
    layer=TenderLinear(2,1,2,bias=False)
    from fedcore.algorithm.quantization.tensor_formats import pack_codes
    layer.weight.packed=pack_codes(torch.tensor([[128,254]],dtype=torch.int64),8)
    layer.weight.scale.fill_(1e-170)
    layer.activation_scale.fill_(1e-170)
    layer.exponents.copy_(torch.tensor([60,0]))
    layer.validate_state(layer.state_dict())
    inputs=torch.tensor([[1e-170*2**60,0.]],dtype=torch.float64)
    from decimal import Decimal,localcontext
    with localcontext() as context:
        context.prec=100
        expected=float(Decimal(2**60)*Decimal.from_float(1e-170)*Decimal.from_float(1e-170))
    assert expected>0
    assert float(layer(inputs)[0,0])==expected


def test_second_moment_normalizes_before_gram_to_avoid_representable_overflow():
    from fedcore.algorithm.quantization.matrix_transforms import input_second_moment
    rows=torch.tensor([[1e154,1e154],[1e154,-1e154]],dtype=torch.float64)
    moment=input_second_moment(rows,2)
    assert bool(torch.isfinite(moment).all())
    torch.testing.assert_close(moment,torch.eye(2,dtype=torch.float64)*1e308,atol=1e293,rtol=1e-14)
    model=nn.Sequential(nn.Linear(2,2,bias=False).double())
    model[0].weight.data.copy_(torch.tensor([[1e-154,.3e-154],[.2e-154,-.8e-154]],dtype=torch.float64))
    from fedcore.algorithm.quantization.graph_execution import compress_quantization_graph
    result=compress_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.CALDERA,rank=1,
        outer_iterations=1,inner_iterations=1)),)),calibration=rows)
    assert isinstance(result,ExportVerifiedArtifact),result


def test_second_moment_preserves_subnormal_entries_beside_overflowing_entries():
    from fedcore.algorithm.quantization.matrix_transforms import input_second_moment
    tiny=torch.full((4,1),2e-162,dtype=torch.float64)
    expected=(tiny.T@tiny)/len(tiny)
    assert float(expected[0,0])>0
    assert torch.equal(input_second_moment(tiny,1),expected)
    mixed=torch.cat((torch.full_like(tiny,1e154),tiny),dim=1)
    result=input_second_moment(mixed,2)
    assert bool(torch.isfinite(result).all())
    assert result[0,0]==pytest.approx(1e308)
    assert result[1,1]==expected[0,0]
    assert result[0,1]==(mixed.T@mixed)[0,1]/len(mixed)


def test_small_retained_psd_direction_changes_ldlq_feedback_and_public_caldera():
    from fedcore.algorithm.quantization.matrix_transforms import scalar_ldlq
    vector=torch.tensor([1e-12,1.],dtype=torch.float64)
    weight=torch.tensor([[4e11,.5]],dtype=torch.float64)
    encoded,_=scalar_ldlq(weight,torch.outer(vector,vector),QuantizedTensorSpec(2,bound=1e12),generator=torch.Generator().manual_seed(0))
    assert encoded.decode().tolist()==[[1e12,0.]]
    assert float(((weight-encoded.decode())@vector).square().sum())==pytest.approx(.01)
    from fedcore.algorithm.quantization.graph_execution import compress_quantization_graph
    model=nn.Sequential(nn.Linear(2,2).double())
    rows=torch.tensor([[-1e-6,-1.],[1e-6,1.]],dtype=torch.float64)
    result=compress_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.CALDERA,rank=1,
        outer_iterations=1,inner_iterations=1)),)),calibration=rows)
    assert isinstance(result,ExportVerifiedArtifact),result
    assert result.evidence['operators'][0]['metric_support']==1


@pytest.mark.parametrize('guard_result',[None,'invalid',ResourceFailure('ChangedContext','Review guard stopped')])
def test_optional_common_context_guard_preserves_default_and_types_failures(guard_result):
    from fedcore.algorithm.low_rank.grammar_search import run_grammar_search,GrammarFitness
    from fedcore.algorithm.quantization.grammar import QUANTIZATION_GRAMMAR_ADAPTER
    model=nn.Sequential(nn.Linear(2,2))
    space=prepare_quantization_space(model,{'0':(TransformSpec(Method.DENSE),)},evaluation_context='review-optional-guard')
    initial=generate_genotype(space)
    config=GrammarSearchConfig(population_size=2,generations=1,max_evaluations=2,max_seconds=30,max_proposals=5)
    adapter=replace(QUANTIZATION_GRAMMAR_ADAPTER,context_guard=None if guard_result is None else lambda:guard_result)
    result=run_grammar_search(space,config,lambda _:GrammarFitness(1.,2.),fitness_context=space.evaluation_context,
        initial_genotypes=(initial,initial),grammar_adapter=adapter)
    if guard_result is None:
        assert result['status']=='succeeded' and result['charged_evaluations']==1
        assert result['context_failure'] is None and result['finite_archive']
    else:
        assert result['status']=='failed' and not result['finite_archive']
        assert result['context_failure']['code']==('ChangedContext' if type(guard_result) is ResourceFailure else 'InvalidContextGuardResult')
    with pytest.raises(ValueError,match='context_guard'):
        run_grammar_search(space,config,lambda _:GrammarFitness(1.,2.),fitness_context=space.evaluation_context,
            grammar_adapter=replace(QUANTIZATION_GRAMMAR_ADAPTER,context_guard=7))


def test_context_invalidation_latches_even_if_trace_restores_source_at_stop():
    model=nn.Sequential(nn.Linear(2,2));original=model[0].weight.detach().clone()
    space=prepare_quantization_space(model,{'0':(TransformSpec(Method.DENSE),)},evaluation_context='review-latched-context')
    initial=generate_genotype(space)
    def trace(event):
        with torch.no_grad():
            if event['event']=='evaluated':model[0].weight.add_(1)
            if event['event']=='stopped':model[0].weight.copy_(original)
    result=run_quantization_search(model,space,GrammarSearchConfig(population_size=1,generations=1,max_evaluations=1,max_seconds=30,max_proposals=3),
        calibration=None,evaluate=lambda _:0.,initial_genotypes=(initial,),trace_callback=trace)
    assert torch.equal(model[0].weight,original)
    assert result['status']=='failed' and not result['best_genotypes'] and not result['finite_archive']
