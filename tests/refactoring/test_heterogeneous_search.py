"""Exercise the actual G1 public search facade without training a network."""
from dataclasses import replace
import numpy as np
import pytest

from fedcore.algorithm.heterogeneous.cv import (
    BlockRecipe,HeterogeneousDomain,HeterogeneousSpace,default_alternatives)
from fedcore.algorithm.heterogeneous.policy import SearchArm
from fedcore.algorithm.heterogeneous.search import common_initial,run_search
from fedcore.algorithm.low_rank.compression_graph import CalibrationRef,ModelVersion
from fedcore.algorithm.low_rank.cv_grammar import CVBlockDomain,CVGrammarSpace
from fedcore.algorithm.low_rank.grammar_search import GrammarFitness,GrammarSearchConfig
from fedcore.experiments.heterogeneous_data import split_roles


def metadata_space():
    catalog = default_alternatives()
    paths = ('layer1.0','layer1.1')
    structural = CVGrammarSpace(ModelVersion('g1-test','a'*64),
        CalibrationRef('b'*64,(2,3,32,32),'float32'),
        tuple(CVBlockDomain(path,tuple(dict.fromkeys(r.structural for r in catalog[path]))) for path in paths),
        'c'*64,44734408)
    return HeterogeneousSpace(structural,tuple(HeterogeneousDomain(path,catalog[path]) for path in paths))


@pytest.mark.parametrize('arm',tuple(SearchArm))
def test_evo_facade_uses_the_shared_paid_prefix_and_real_spea2(arm):
    space = metadata_space()
    config = GrammarSearchConfig(seed=42,population_size=2,generations=3,
        max_evaluations=5,max_proposals=40,max_seconds=30,mutation_only=True)
    prefix = common_initial(space,2,42)
    seen = []
    def evaluate(workflow):
        seen.append(workflow.fingerprint)
        return GrammarFitness(1.+len(workflow.ptq_regions),float(workflow.structural_workflow.expected_storage_bytes))
    result = run_search(space,config,evaluate,arm=arm,fitness_context='fixed metadata test',initial_genotypes=prefix)
    assert result['engine'] == 'FedcoreEvoOptimizer' and result['search_arm'] == arm.value
    assert result['charged_evaluations'] == len(set(seen)) == len(seen)
    assert len(seen) >= 2 and result['charge_before_compile'] and result['strict_deadline']
    assert result['finite_archive']
    assert any(event['event']=='population' and event['selection']=='SPEA2' for event in result['trace'])


def test_prefix_rejects_duplicates_and_wrong_length_before_evaluation():
    space = metadata_space(); prefix = common_initial(space,2,42)
    config = GrammarSearchConfig(population_size=2,max_evaluations=5)
    for initial in (prefix[:1],(prefix[0],prefix[0])):
        with pytest.raises(ValueError,match='distinct'):
            run_search(space,config,lambda w:pytest.fail('No evaluation permitted'),
                arm=SearchArm.STAGED,fitness_context='test',initial_genotypes=initial)


def test_new_roles_partition_and_stratify_training_only():
    labels = np.repeat(np.arange(10,dtype=np.int64),5000)
    roles = split_roles(labels)
    assert roles == split_roles(labels) and roles.sha256 == split_roles(labels).sha256
    groups = (roles.fit,roles.calibration,roles.search,roles.report)
    for ids,per_class in zip(groups,(4000,200,400,400)):
        assert np.array_equal(np.bincount(labels[list(ids)]),np.full(10,per_class))
    assert len(set(i for group in groups for i in group)) == 50000
    assert sum(map(len,groups)) == 50000
