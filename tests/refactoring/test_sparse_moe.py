"""Independent routing and combinatorial oracles for bounded sparse MoE."""
from copy import deepcopy
import itertools

import pytest
import torch
from torch.nn import functional as F

from fedcore.algorithm.pruning.sparse.moe import (
    ReferenceMoE, remove_experts, adapt_mixtral, intervention_scores,
    expert_candidates, kt_block_search, allocate_expert_ranks, compress_experts,
    genetic_expert_candidates, allocate_layer_deletions, prepare_expert_recovery,
    MoECompressionSpec, moe_i2,
    next_token_loss,
)
from fedcore.algorithm.pruning.sparse.specs import SparseDomainError


def example():
    torch.manual_seed(71)
    model = ReferenceMoE(4, 7, 4, 2, shared=True, router_bias=True, dtype=torch.float64)
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    return model, x


def oracle(model, x, retained):
    flat = x.flatten(0, -2)
    logits = F.linear(flat, model.router.weight[list(retained)], model.router.bias[list(retained)])
    probabilities = logits.float().softmax(-1)
    values, ids = probabilities.topk(model.top_k, -1)
    values = (values / values.sum(-1, keepdim=True)).to(x.dtype)
    rows = []
    for t in range(len(flat)):
        row = sum(values[t, k]*model.experts[retained[int(ids[t,k])]](flat[t])
                  for k in range(model.top_k))
        rows.append(row + model.shared(flat[t]))
    return torch.stack(rows).reshape_as(x)


def test_removal_matches_independent_token_oracle_and_leaves_source():
    model, x = example()
    saved = deepcopy(model.state_dict())
    compact, evidence = remove_experts(model, [3, 1, 0])
    torch.testing.assert_close(compact(x), oracle(model, x, (3,1,0)), atol=1e-12, rtol=1e-12)
    assert len(compact.experts) == compact.router.out_features == 3
    assert compact.expert_ids.tolist() == [3,1,0]
    assert evidence['active_experts_per_token'] == 2
    assert all(torch.equal(saved[k], v) for k,v in model.state_dict().items())
    assert compact.route(x)[1].max() < 3
    assert not any(k.startswith('experts.3.') for k in compact.state_dict())


def test_all_retained_and_permutation_preserve_function():
    model, x = example()
    full, _ = remove_experts(model, range(4))
    torch.testing.assert_close(full(x), model(x), atol=0, rtol=0)
    permuted, _ = remove_experts(model, [2,0,3,1])
    torch.testing.assert_close(permuted(x), model(x), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('retained', [(1,), (1,1), (False,1), (0,5)])
def test_invalid_map_preflight(retained):
    model, _ = example()
    with pytest.raises(SparseDomainError):
        remove_experts(model, retained)


def test_losses_are_absolute_and_layer_sum_not_mean():
    model, _ = example()
    # A deterministic callback independent of neural loss implementation.
    scores = intervention_scores(model, lambda m: 10 + sum(m.expert_ids.tolist()))
    assert scores['absolute_losses'] == [16,15,14,13]
    assert scores['layer_sum'] == 58
    assert scores['layer_mean'] == 14.5
    assert scores['delta_losses'] == [0,-1,-2,-3]


def test_candidates_and_joint_search_match_exhaustive_oracle():
    model, x = example()
    candidates, evidence = expert_candidates(model, x, 2, best_k=3)
    expected = sorted((float(torch.linalg.vector_norm(oracle(model,x,ids)-model(x))), ids)
                      for ids in itertools.combinations(range(4), 2))
    assert [c.retained for c in candidates] == [ids for _,ids in expected[:3]]
    assert evidence['evaluations'] == 6
    assert evidence['global_optimality']
    score = lambda maps: abs(sum(maps[0])-sum(maps[1])-1)
    selected, joint = kt_block_search([candidates,candidates], score)
    assert (score(selected), selected) == min((score(m),m) for m in
        itertools.product([c.retained for c in candidates], repeat=2))
    assert joint['evaluations'] == 9
    with pytest.raises(SparseDomainError, match='exceed'):
        kt_block_search([candidates]*10, score, max_evaluations=10)


def test_bounded_subset_search_replays_generator_and_maps():
    model, x = example()
    a, ea = expert_candidates(model,x,2,best_k=2,max_evaluations=3,seed=81)
    b, eb = expert_candidates(model,x,2,best_k=2,max_evaluations=3,seed=81)
    assert a == b and ea == eb and ea['evaluations'] <= 3


@pytest.mark.parametrize('budget', range(3,13))
def test_integer_rank_budget_and_caps(budget):
    ranks = allocate_expert_ranks([0,10,2], [2,4,6], budget)
    assert sum(ranks) == budget
    assert all(1 <= r <= c for r,c in zip(ranks,[2,4,6]))
    assert ranks == allocate_expert_ranks([0,10,2], [2,4,6], budget)


def test_weighted_full_rank_preserves_function_and_router():
    model, x = example()
    result, evidence = compress_experts(model, x, [4]*4)
    torch.testing.assert_close(result(x), model(x), atol=1e-7, rtol=1e-6)
    assert torch.equal(result.router.weight, model.router.weight)
    assert torch.equal(result.route(x)[1], model.route(x)[1])
    assert all(e['assigned_tokens'] > 0 for e in evidence['experts'])
    assert all(hasattr(e.gate_proj,'left') for e in result.experts)


def test_unobserved_expert_refused_before_commit():
    model = ReferenceMoE(4,8,3,1)
    with torch.no_grad():
        model.router.weight.zero_()
    with pytest.raises(SparseDomainError) as err:
        compress_experts(model, torch.ones(3,4), [2]*3)
    assert err.value.code == 'insufficient_coverage'
    assert all(isinstance(e.up_proj,torch.nn.Linear) for e in model.experts)


def test_real_mixtral_adapter_matches_tuple_output():
    pytest.importorskip('transformers',reason='Real Mixtral adapter requires the llm extra')
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    config = MixtralConfig(hidden_size=8,intermediate_size=12,num_local_experts=4,
                           num_experts_per_tok=2,router_jitter_noise=0)
    original = MixtralSparseMoeBlock(config).eval()
    x = torch.randn(2,3,8)
    adapted = adapt_mixtral(original)
    torch.testing.assert_close(adapted(x), original(x)[0], atol=1e-7, rtol=1e-6)
    assert adapted.num_experts == 4


def test_genetic_search_cardinality_budget_and_replay():
    model,x = example()
    a,ea = genetic_expert_candidates(model,x,2,population=3,generations=3,max_evaluations=5,seed=11)
    b,eb = genetic_expert_candidates(model,x,2,population=3,generations=3,max_evaluations=5,seed=11)
    assert a == b and ea == eb and ea['evaluations'] <= 5
    assert not ea['global_optimality']
    assert all(len(c.retained) == len(set(c.retained)) == 2 for c in a)
    for candidate in a:
        expected = float(torch.linalg.vector_norm(oracle(model,x,candidate.retained)-model(x)))
        assert candidate.error == pytest.approx(expected,rel=1e-12)


def test_layer_allocation_is_named_explicit_choice_and_respects_caps():
    assert allocate_layer_deletions([1,10],[3,3],3) == (3,0)
    assert allocate_layer_deletions([1,10],[1,3],3) == (1,2)
    assert allocate_layer_deletions([0,0],[2,2],3) == (2,1)


def test_factor_recovery_runs_real_updates_and_merges_into_short_factors():
    from fedcore.experiments.protocol import TensorSplit
    from fedcore.experiments.runner import train_model
    from fedcore.algorithm.low_rank.factor_recovery import finish_factor_lora_stage
    model,x = example()
    compact,_ = compress_experts(model,x,[2]*4)
    stage = prepare_expert_recovery(compact,factor='left',adapter_rank=1)
    y = model(x).detach()
    split = TensorSplit(x,y,tuple(f'sample{i}' for i in range(len(x))))
    trained = train_model(stage.model,split,'regression',epochs=2,batch_size=1,
                          learning_rate=.01,max_updates=2,optimizer_validator=stage.validate_optimizer)
    final,record = finish_factor_lora_stage(stage)
    assert trained['training_steps'] == len(trained['train_loss']) == 2
    assert any(v > 0 for v in record['delta_norms'].values())
    assert all(e.up_proj.left.weight.shape[1] == 2 for e in final.experts)
    assert all(not p.requires_grad for p in final.parameters())


def test_moe_constructor_rejects_multiplicative_resource_explosion():
    model,_ = example()
    config = model.representation_config()
    config.update(d_model=4096,intermediate_size=4096,num_experts=4096)
    with pytest.raises(SparseDomainError) as err:
        ReferenceMoE.from_config(config)
    assert err.value.code == 'resource_limit'


def test_whole_target_budget_includes_router_shared_and_identity_buffers():
    from fedcore.experiments.measurement import tensor_state_bytes
    model,x = example()
    spec = MoECompressionSpec((0,1,2),(2,2,2))
    compressed,evidence = moe_i2(model,x,spec)
    assert evidence['target_moe_tensor_bytes'] == tensor_state_bytes(compressed)
    cost = evidence['target_moe_tensor_bytes']
    with pytest.raises(SparseDomainError) as err:
        moe_i2(model,x,MoECompressionSpec((0,1,2),(2,2,2),budget_bytes=cost-1))
    assert err.value.code == 'budget_infeasible'
    # Removing only experts needs no calibration, including through the facade.
    removed,evidence = moe_i2(model,None,MoECompressionSpec((1,2,3)))
    assert evidence['target_moe_tensor_bytes'] == tensor_state_bytes(removed)


def test_workspace_limit_and_nonlinearity_are_checked_before_factorization():
    model,x = example()
    with pytest.raises(SparseDomainError) as err:
        compress_experts(model,x,(2,)*4,max_workspace_bytes=1)
    assert err.value.code == 'resource_limit'
    model.experts[0].activation = 'relu'
    with pytest.raises(SparseDomainError,match='original ReferenceMLP'):
        compress_experts(model,x,(2,)*4)


def test_bias_budget_and_failed_wrapper_preflight_do_not_consume_rng(monkeypatch):
    from fedcore.experiments.measurement import tensor_state_bytes
    model = ReferenceMoE(4,6,1,1,dtype=torch.float64)
    model.experts[0].up_proj.bias = torch.nn.Parameter(torch.ones(6,dtype=torch.float64))
    model.experts[0].down_proj.bias = torch.nn.Parameter(torch.ones(4,dtype=torch.float64))
    x = torch.randn(10,4,dtype=torch.float64)
    result,evidence = moe_i2(model,x,MoECompressionSpec((0,),(1,)))
    assert evidence['target_moe_tensor_bytes'] == tensor_state_bytes(result) == 368
    with pytest.raises(SparseDomainError) as err:
        moe_i2(model,x,MoECompressionSpec((0,),(1,),budget_bytes=288))
    assert err.value.code == 'budget_infeasible'
    rng = torch.random.get_rng_state().clone()
    remove_experts(model,(0,))
    assert torch.equal(rng,torch.random.get_rng_state())
    # Any copying is forbidden for a configuration already known to be invalid.
    import fedcore.algorithm.pruning.sparse.moe as implementation
    def forbidden(*args,**kwargs):
        raise AssertionError('copy reached before validation')
    monkeypatch.setattr(implementation,'deepcopy',forbidden)
    for spec in (MoECompressionSpec((0,),(5,)),MoECompressionSpec((0,),(1,),max_workspace_bytes=1)):
        with pytest.raises(SparseDomainError):
            moe_i2(model,x,spec)
        assert torch.equal(rng,torch.random.get_rng_state())


def test_routing_workspace_and_population_one_limits(monkeypatch):
    model = ReferenceMoE(1,1,1000,1,dtype=torch.float64)
    def forbidden(*args,**kwargs):
        raise AssertionError('route reached despite insufficient workspace')
    monkeypatch.setattr(model,'route',forbidden)
    with pytest.raises(SparseDomainError) as err:
        compress_experts(model,torch.ones(1000,1,dtype=torch.float64),(1,)*1000,max_workspace_bytes=100000)
    assert err.value.code == 'resource_limit'
    small,x = example()
    candidates,evidence = genetic_expert_candidates(small,x,2,best_k=1,population=1,generations=1,seed=1)
    assert len(evidence['history'][0]['population']) == evidence['evaluations'] == len(candidates) == 1
    with pytest.raises(SparseDomainError):
        genetic_expert_candidates(small,x,2,best_k=2,max_evaluations=1)


def test_low_rank_experts_match_independent_cholesky_svd_oracle():
    torch.manual_seed(812)
    model = ReferenceMoE(3,5,1,1,dtype=torch.float64)
    x = torch.randn(19,3,dtype=torch.float64)
    result,_ = compress_experts(model,x,(1,),ridge=.03)
    source,target = model.experts[0],result.experts[0]
    features = F.silu(source.gate_proj(x))*source.up_proj(x)
    expected = {}
    for name,rows in (('gate_proj',x),('up_proj',x),('down_proj',features)):
        weight = getattr(source,name).weight.detach()
        gram = rows.T@rows/len(rows)+.03*torch.eye(rows.shape[1],dtype=rows.dtype)
        chol = torch.linalg.cholesky(gram)
        u,s,vh = torch.linalg.svd(weight@chol,full_matrices=False)
        expected[name] = ((u[:,:1]*s[:1])@vh[:1])@torch.linalg.inv(chol)
        factorized = getattr(target,name)
        torch.testing.assert_close(factorized.left.weight@factorized.right.weight,expected[name],atol=1e-12,rtol=1e-12)
    y = F.linear(F.silu(F.linear(x,expected['gate_proj']))*F.linear(x,expected['up_proj']),expected['down_proj'])
    torch.testing.assert_close(result(x),y,atol=1e-12,rtol=1e-12)


def test_two_moe_layers_joint_next_token_search_matches_exhaustive_model():
    torch.manual_seed(551)
    first = ReferenceMoE(3,5,3,1,dtype=torch.float64)
    second = ReferenceMoE(3,5,3,1,dtype=torch.float64)
    head = torch.nn.Linear(3,7,dtype=torch.float64)
    x = torch.randn(2,5,3,dtype=torch.float64)
    tokens = torch.randint(7,(2,5))
    first_candidates,_ = expert_candidates(first,x,2,best_k=3)
    second_candidates,_ = expert_candidates(second,first(x).detach(),2,best_k=3)
    def evaluate(maps):
        a,_ = remove_experts(first,maps[0])
        b,_ = remove_experts(second,maps[1])
        return float(next_token_loss(head(b(a(x))),tokens))
    chosen,evidence = kt_block_search((first_candidates,second_candidates),evaluate)
    reference = min((evaluate(maps),maps) for maps in itertools.product(
        itertools.combinations(range(3),2),repeat=2))
    assert chosen == reference[1]
    assert evidence['evaluations'] == 9
    # Independent scalar CE using manually selected shifted probabilities.
    logits = head(second(first(x))).detach()
    expected = -logits[:,:-1].log_softmax(-1).gather(-1,tokens[:,1:,None]).mean()
    torch.testing.assert_close(next_token_loss(logits,tokens),expected)


def test_moe_rejects_reshape_compatible_but_semantically_wrong_hidden_axis():
    model = ReferenceMoE(4,6,3,1)
    with pytest.raises(SparseDomainError) as err:
        model(torch.ones(4,3))
    assert err.value.code == 'unsupported_input'
