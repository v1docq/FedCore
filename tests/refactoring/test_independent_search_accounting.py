"""Independent laws for opt-in attempt charging and shared dispatch safety."""
from dataclasses import replace

import pytest

from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
from fedcore.algorithm.low_rank.compression_grammar import (
    CompressionGenotype, GrammarSpace, ScopeDomain, ScopeGene, WeightedChoice,
    compile_genotype, normalize_genotype)
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
from fedcore.algorithm.low_rank.grammar_search import (
    G3_GRAMMAR_ADAPTER, GrammarFitness, GrammarSearchConfig, run_grammar_search)
from fedcore.algorithm.low_rank.plans import OperatorDescriptor, ValidationFailure, Violation


def space_and_prefix():
    choices=tuple(WeightedChoice(r) for r in (1,2,3,4))
    space=GrammarSpace(ModelVersion('attempt-test','a'*64),
        CalibrationRef('b'*64,(6,4),'float32'),
        (ScopeDomain(OperatorDescriptor('layer',(4,4),'Linear'),('layer',),choices),))
    prefix=tuple(CompressionGenotype((ScopeGene(('layer',),choice),)) for choice in choices)
    return space,prefix


def test_compilation_failures_are_charged_once_before_compile():
    space,prefix=space_and_prefix();compiled=[];events=[]
    def compile_failure(genotype, _space):
        assert events[-1]['event']=='evaluation_started'
        compiled.append(genotype)
        return ValidationFailure((Violation('ForcedFailure','layer','expected compile failure'),))
    adapter=replace(G3_GRAMMAR_ADAPTER,compile=compile_failure)
    result=run_grammar_search(space,GrammarSearchConfig(population_size=4,generations=1,
        max_evaluations=2,max_proposals=8,max_seconds=15),
        lambda _:pytest.fail('A failed compilation must not run the evaluator'),
        fitness_context='strict compile failures',initial_genotypes=prefix,
        grammar_adapter=adapter,charge_before_compile=True,trace_callback=events.append)
    starts=[e for e in events if e['event']=='evaluation_started']
    ends=[e for e in events if e['event']=='evaluated']
    assert result['charged_evaluations']==len(compiled)==len(starts)==len(ends)==2
    assert len({e['canonical_solution_id'] for e in starts})==2
    assert all(e['outcome']=='StructuralInvalid' and e['workflow_fingerprint'] is None for e in ends)
    assert result['finite_archive']==[] and result['failure_count']==2


def test_two_fitness_components_and_duplicate_proposals_do_not_retrain():
    space,prefix=space_and_prefix();compiled=[];evaluated=[]
    def compile_record(genotype, s):compiled.append(genotype);return compile_genotype(genotype,s)
    def evaluate(workflow):evaluated.append(workflow.fingerprint);return GrammarFitness(.5,.5)
    adapter=replace(G3_GRAMMAR_ADAPTER,compile=compile_record,
        mutate=lambda genotype,space,**kwargs:genotype,
        crossover=lambda left,right,space,**kwargs:(left,right))
    result=run_grammar_search(space,GrammarSearchConfig(population_size=4,generations=1,
        max_evaluations=4,max_proposals=8,max_seconds=15),evaluate,
        fitness_context='same genotype replay',initial_genotypes=(prefix[0],)*4,
        grammar_adapter=adapter,charge_before_compile=True)
    assert len(compiled)==len(evaluated)==result['charged_evaluations']==1
    assert result['component_reads']>=2 and result['created_proposals']>=4
    assert all(row['workflow_fingerprint']==evaluated[0] for row in result['finite_archive'])
    assert all(row['canonical_attempt_key'] for row in result['finite_archive'])


def test_typed_failure_is_cached_and_never_receives_finite_fitness():
    space,prefix=space_and_prefix();calls=[]
    def evaluate(workflow):calls.append(workflow.fingerprint);return ResourceFailure('Rejected','fixed resource limit')
    result=run_grammar_search(space,GrammarSearchConfig(population_size=4,generations=1,
        max_evaluations=4,max_proposals=8,max_seconds=15),evaluate,
        fitness_context='typed failures',initial_genotypes=(prefix[0],)*4,
        grammar_adapter=replace(G3_GRAMMAR_ADAPTER,
            mutate=lambda genotype,space,**kwargs:genotype,
            crossover=lambda left,right,space,**kwargs:(left,right)),charge_before_compile=True)
    assert result['charged_evaluations']==len(calls)==1
    assert result['failure_count']==1 and result['finite_archive']==[]


def test_context_failure_invalidates_previously_finite_results():
    space,prefix=space_and_prefix();bad=[False]
    def evaluate(_):bad[0]=True;return GrammarFitness(.1,.2)
    def guard():return ResourceFailure('SourceChanged','immutable source changed') if bad[0] else None
    result=run_grammar_search(space,GrammarSearchConfig(population_size=4,generations=1,
        max_evaluations=4,max_proposals=8,max_seconds=15),evaluate,
        fitness_context='source guards',initial_genotypes=prefix,
        grammar_adapter=replace(G3_GRAMMAR_ADAPTER,context_guard=guard),charge_before_compile=True)
    assert result['charged_evaluations']==1 and result['stop_reason']=='context_invalidated'
    assert result['finite_archive']==result['best_genotypes']==[]
    assert result['context_failure']['code']=='SourceChanged'


def test_charge_before_compile_requires_explicit_bool():
    space,prefix=space_and_prefix()
    with pytest.raises(ValueError):
        run_grammar_search(space,GrammarSearchConfig(),lambda _:None,fitness_context='typed flag',
            initial_genotypes=prefix,charge_before_compile=1)
