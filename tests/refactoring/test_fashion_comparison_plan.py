"""Pure preparation checks: no models, data, training or searches."""
import pytest
from fedcore.experiments.fashion_comparison_plan import ComparisonPlan


def test_complete_paired_blocks_and_determinism():
    plan = ComparisonPlan()
    jobs = plan.jobs()
    assert jobs == plan.jobs() and len(jobs) == len(set(jobs)) == 60
    for model in plan.model_seeds:
        for seed in plan.search_seeds:
            assert {(j.space, j.optimizer) for j in jobs if j.model_seed == model and j.search_seed == seed} == {
                ('svd_only', 'evo'), ('svd_only', 'random'), ('heterogeneous', 'evo'), ('heterogeneous', 'random')}
    assert plan.preparation_manifest()['execution_authorized'] is False
    assert plan.preparation_manifest()['maximum_search_and_pick_seconds'] == 54000


@pytest.mark.parametrize('kwargs', [{'model_seeds': (1, 1)}, {'search_seeds': (True,)},
                                  {'evaluations': 0}, {'population': 25}, {'search_seconds': float('inf')}])
def test_invalid_design_rejected(kwargs):
    with pytest.raises(ValueError): ComparisonPlan(**kwargs)



def test_svd_only_domain_has_no_quantization_or_prefixes():
    from types import SimpleNamespace
    from fedcore.experiments.fashion_protocol import default_manifest, RAW_FILES
    from fedcore.experiments.fashion_comparison_plan import search_domain
    from fedcore.experiments.fashion_worker import unique_initial
    from fedcore.algorithm.quantization.grammar import prepare_quantization_space, genotype_payload
    from fedcore.algorithm.quantization import Method
    from fedcore.experiments.fashion_worker import make_source
    plan=SimpleNamespace(payload=default_manifest('a'*40,{k:'b'*64 for k in RAW_FILES}))
    domain,prefixes=search_domain(plan,'svd_only')
    assert prefixes==() and [len(v) for v in domain.values()]==[4,4,4]
    assert all(s.method in (Method.DENSE,Method.SVD) and s.prequantize is None for v in domain.values() for s in v)
    space=prepare_quantization_space(make_source(),domain,evaluation_context='test',prequantize_bits=prefixes)
    points=unique_initial(space,24,101)
    assert points[:8]==unique_initial(space,8,101)
    assert len({str(genotype_payload(g)) for g in points})==24
    assert all(gene.transform.method in (Method.DENSE,Method.SVD) and gene.transform.prequantize is None for g in points for gene in g.genes)
