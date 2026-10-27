from dataclasses import replace
from collections import OrderedDict
import pytest
import torch
from torch import nn
from hypothesis import given,settings,strategies as st
from fedcore.algorithm.quantization.grammar import *
from fedcore.algorithm.quantization.graph_specs import *
from fedcore.algorithm.quantization.graph_execution import *
from fedcore.algorithm.quantization.tensor_formats import Grid,Rounding,PackedTensor
from fedcore.algorithm.quantization.cache import *
from fedcore.algorithm.quantization.process import compression_nodes
from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig
from fedcore.algorithm.low_rank.compression_grammar import MutationKind


def fixture():
    torch.manual_seed(8)
    model=nn.Sequential(nn.Linear(5,6),nn.ReLU(),nn.Linear(6,3))
    x=torch.randn(12,5)
    q=QuantizedTensorSpec(4,rounding=Rounding.STOCHASTIC,grid=Grid.ENDPOINT)
    specs=(TransformSpec(Method.LPLR,rank=2,left=q,right=q),
           TransformSpec(Method.CALDERA,rank=1,outer_iterations=1,inner_iterations=1,weight=QuantizedTensorSpec(4)),
           TransformSpec(Method.TENDER))
    space=prepare_quantization_space(model,{'0':(TransformSpec(Method.DSVD,rank=2),TransformSpec(Method.PTQ,weight=QuantizedTensorSpec(4))),
        '2':specs},calibration=x,evaluation_context='toy-heldout-fixed-v1')
    return model,x,space


@given(st.integers(min_value=0,max_value=2**32-1))
@settings(max_examples=15,deadline=None)
def test_generated_mutated_and_crossed_dags_obey_independent_constraints(seed):
    model,x,space=fixture()
    first,second=generate_genotype(space,seed=seed),generate_genotype(space,seed=seed+1)
    children=crossover_genotypes(first,second,space,seed=seed)
    for genotype in (first,second,*children,mutate_genotype(first,space,kind=MutationKind.SCOPE,seed=seed)):
        checked=normalize_genotype(genotype,space)
        assert not isinstance(checked,ValidationFailure),checked
        assert normalize_genotype(checked,space)==checked
        assert {g.aliases for g in checked.genes}=={d.aliases for d in space.domains}
        for gene in checked.genes:
            names=set()
            for node in gene.graph.nodes:
                assert all(parent in names for parent in node.inputs)
                names.add(node.name)
            assert gene.graph.output in names
            if gene.transform.method in (Method.LPLR,Method.LPLR_SVD):
                ops=[n.operation for n in gene.graph.nodes]
                assert ops.index('QuantizeLeft')<ops.index('LeastSquaresFit')<ops.index('QuantizeRight')


def test_prefix_rewrite_executes_noncommuting_operations_and_common_ports():
    model,x,space=fixture()
    spec=TransformSpec(Method.DSVD,rank=2,left=QuantizedTensorSpec(4),right=QuantizedTensorSpec(4))
    plain=compress_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe('0',spec),)),calibration=x)
    rewritten=compress_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe('0',replace(spec,prequantize=QuantizedTensorSpec(2))),)),calibration=x)
    assert type(plain) is ExportVerifiedArtifact and type(rewritten) is ExportVerifiedArtifact
    assert not torch.equal(plain.model(x),rewritten.model(x))
    ops=[entry['operation'] for entry in rewritten.evidence['operators'][0]['process_trace']]
    assert ops[:4]==['ExtractMatrix','PreQuantizeTensor','DecodeTensor','SVDTruncate']
    graph=recipe_graph(spec)
    renamed=ProcessGraphSpec(tuple(ProcessNode('v'+n.name,n.operation,tuple('v'+p for p in n.inputs)) for n in graph.nodes),'v'+graph.output)
    assert canonical_graph(renamed)==graph
    common=compression_nodes(graph,scope=('0',),model_version='source-v1')
    assert all(port.artifact in common[i].reads for i,node in enumerate(common) for port in node.inputs)
    assert plain.plan.fingerprint!=rewritten.plan.fingerprint


def test_source_architecture_and_calibration_stale_plan():
    model,x,space=fixture()
    request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.PTQ)),))
    plan=check_quantization_graph(model,request,calibration=x)
    model[1]=nn.Sigmoid()
    assert isinstance(execute_quantization_graph(model,plan,calibration=x),StructuralInvalid)
    assert source_identity(model)!=space.source
    assert isinstance(execute_quantization_graph(model,check_quantization_graph(model,request,calibration=x),calibration=x+1),StructuralInvalid)


def test_dependency_cache_state_model_ownership_invalidation_and_capacity():
    cache=DependencyCache(128)
    keys=[CacheIdentity(kind,'M0',pred,'data','coords','op','policy') for kind,pred in [('statistics','M0'),('metric','M1'),('solution','M1')]]
    source=torch.arange(4,dtype=torch.float64)
    assert cache.put(keys[0],(source,),dependencies=('source-v0',))
    assert cache.put(keys[1],(source,),dependencies=(keys[0].digest,))
    assert cache.put(keys[2],(source,),dependencies=(keys[1].digest,))
    source.zero_();read=cache.get(keys[0]);assert read[0].sum()==6
    read[0].zero_();assert cache.get(keys[0])[0].sum()==6
    assert not cache.put(replace(keys[2],policy='large'),(torch.zeros(100),))
    removed=cache.invalidate('source-v0')
    assert removed==frozenset(k.digest for k in keys)
    assert cache.bytes==0 and cache.get(keys[2]) is None


def test_cache_used_and_current_predecessor_changes_dependent_statistics():
    model,x,space=fixture();cache=DependencyCache()
    recipes=(OperatorRecipe('0',TransformSpec(Method.PTQ,weight=QuantizedTensorSpec(2))),
             OperatorRecipe('2',TransformSpec(Method.CALDERA,rank=1,outer_iterations=1,inner_iterations=1)))
    request=QuantizationGraphRequest(recipes,source_mode=SourceMode.CURRENT_STUDENT)
    first=compress_quantization_graph(model,request,calibration=x,cache=cache)
    assert type(first) is ExportVerifiedArtifact,first
    repeated=compress_quantization_graph(model,request,calibration=x,cache=cache)
    assert type(repeated) is ExportVerifiedArtifact,repeated
    assert any(event['event']=='hit' for event in cache.events)
    before={entry[0].predecessor for entry in cache.entries.values() if entry[0].kind=='observation'}
    changed=compress_quantization_graph(model,replace(request,recipes=(replace(recipes[0],transform=TransformSpec(Method.PTQ,weight=QuantizedTensorSpec(8))),recipes[1])),calibration=x,cache=cache)
    assert type(changed) is ExportVerifiedArtifact,changed
    after={entry[0].predecessor for entry in cache.entries.values() if entry[0].kind=='observation'}
    assert len(after-before)==1
    assert first.evidence['operators'][1]['source_version']!=changed.evidence['operators'][1]['source_version']


def test_actual_fedcore_optimizer_search_uses_all_methods_and_graph_rewrite():
    model,x,space=fixture()
    initial=[]
    for i,spec in enumerate(space.domains[1].choices):
        first=space.domains[0].choices[0]
        if first.method is not Method.TENDER:first=replace(first,prequantize=QuantizedTensorSpec(4) if i%2 else None)
        initial.append(QuantizationGenotype((QuantizationGene(('0',),first,recipe_graph(first)),QuantizationGene(('2',),spec,recipe_graph(spec)))))
    target=model(x).detach()
    result=run_quantization_search(model,space,GrammarSearchConfig(population_size=3,generations=1,max_evaluations=6,max_seconds=45,max_proposals=16),
        calibration=x,evaluate=lambda loaded:float((loaded(x)-target).square().mean()),initial_genotypes=tuple(initial))
    assert result['engine']=='FedcoreEvoOptimizer'
    assert result['charged_evaluations']>=3,result
    evaluated=[event for event in result['trace'] if event['event']=='evaluated']
    assert all(event['outcome']=='GrammarFitness' for event in evaluated),evaluated
    methods={g['transform']['method'] for event in evaluated for g in event['genotype']['genes']}
    assert {'lplr','caldera_cpu_scalar','tender_linear_reference'}.issubset(methods)
    assert result['resource_ledger']['stages']>0
    assert result['artifact_cache']['bytes']>0


def test_search_training_gene_executes_recovery_in_same_workflow():
    from fedcore.algorithm.quantization.program import RecoveryStageSpec
    model=nn.Sequential(nn.Linear(4,3));x,y=torch.randn(6,4),torch.randn(6,3);batches=((x,y),)
    stage=RecoveryStageSpec(('0',),steps=1)
    space=prepare_quantization_space(model,{'0':(TransformSpec(Method.DSVD,rank=2),)},calibration=x,evaluation_context='toy-recovery-selection',
        training_choices=(None,stage),training_batches=batches)
    spec=space.domains[0].choices[0]
    initial=QuantizationGenotype((QuantizationGene(('0',),spec,recipe_graph(spec)),),training=stage)
    workflow=compile_genotype(initial,space)
    assert workflow.program.recovery==stage
    result=run_quantization_search(model,space,GrammarSearchConfig(population_size=1,generations=1,max_evaluations=1,max_seconds=30,max_proposals=4),
        calibration=x,training_batches=batches,evaluate=lambda loaded:float((loaded(x)-y).square().mean()),initial_genotypes=(initial,))
    assert result['engine']=='FedcoreEvoOptimizer'
    assert result['resource_ledger']['actual_training_steps']==2,result
    assert result['charged_evaluations']==1


def test_existing_g5_real_factors_then_ready_quantization_preserves_factors():
    from fedcore.algorithm.low_rank.compression_composition import CompositionRequest,CompositionRecipe,LowRank,prepare_composition,execute_composition
    from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
    from fedcore.algorithm.low_rank.structured_profiles import FactorPair
    # Existing SVD factors are a first-class source, not fed back through SVD.
    dense=nn.Sequential(nn.Linear(4,5))
    previous=execute_composition(dense,prepare_composition(dense,CompositionRequest((CompositionRecipe('0',(LowRank(2),)),))))
    model=previous.model
    left,right=model[0].left.weight.detach(),model[0].right.weight.detach()
    request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.FACTORS,left=QuantizedTensorSpec(4),right=QuantizedTensorSpec(8))),))
    result=compress_quantization_graph(model,request,calibration=torch.randn(3,4))
    assert type(result) is ExportVerifiedArtifact,result
    expected=PackedTensor.from_tensor(left.double(),QuantizedTensorSpec(4))
    assert torch.equal(result.model[0].left.packed,expected.packed)
    assert result.evidence['operators'][0]['process_trace'][0]['operation']=='ExtractFactors'
