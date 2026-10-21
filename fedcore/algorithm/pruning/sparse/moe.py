"""Bounded MoE removal and weighted expert compression.

The reference executes rerouting among retained experts, not zeroing their old
outputs. Selection rules missing from MoE-I2 are named engineering choices.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.coupled_layers import ReferenceMLP
from fedcore.algorithm.low_rank.approximation import solve_weighted
from fedcore.algorithm.low_rank.plans import MetricPolicy
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair
from .specs import SparseDomainError


def _fail(message, code="invalid_specification"):
    raise SparseDomainError(code, message)


def _ids(values, size, minimum=1):
    values = tuple(values)
    if (len(values) < minimum or len(set(values)) != len(values)
            or any(type(i) is not int or not 0 <= i < size for i in values)):
        _fail("Distinct expert indices within the source range are required")
    return values


class ReferenceMoE(nn.Module):
    """Linear router, top-k, selected-weight renormalization, optional shared MLP."""

    def __init__(self, d_model, intermediate_size, num_experts, top_k=2,
                 *, shared=False, router_bias=False, dtype=None, device=None):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in
                (d_model, intermediate_size, num_experts, top_k))
                or top_k > num_experts or type(shared) is not bool
                or type(router_bias) is not bool):
            _fail("Positive bounded MoE dimensions and boolean flags required")
        if (dtype not in (None,torch.float32,torch.float64)
                or (device is not None and torch.device(device).type != "cpu")
                or 3*d_model*intermediate_size*(num_experts+int(shared))+d_model*num_experts > 16_000_000):
            _fail("ReferenceMoE requires bounded CPU float32/float64 storage", "resource_limit")
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=router_bias, dtype=dtype, device=device)
        self.experts = nn.ModuleList([
            ReferenceMLP(d_model, intermediate_size, gated=True, bias=False).to(device=device, dtype=dtype)
            for _ in range(num_experts)])
        self.shared = (ReferenceMLP(d_model, intermediate_size, gated=True, bias=False).to(device=device, dtype=dtype)
                       if shared else None)
        self.register_buffer("expert_ids", torch.arange(num_experts, device=device))
        self.register_buffer("load_counts", torch.zeros(num_experts, dtype=torch.long, device=device))

    def route(self, x):
        if not isinstance(x,torch.Tensor) or x.ndim < 1 or x.shape[-1] != self.d_model:
            _fail("MoE input must preserve its declared last dimension", "unsupported_input")
        logits = self.router(x.reshape(-1, self.d_model))
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(x.dtype), indices, logits

    def forward(self, x):
        flat = x.reshape(-1, self.d_model)
        weights, indices, _ = self.route(x)
        result = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == i)
            result.index_add_(0, tokens, expert(flat[tokens]) * weights[tokens, slots, None])
        if self.shared is not None:
            result = result + self.shared(flat)
        return result.reshape(x.shape)

    def representation_config(self):
        return {"layer_type": "ReferenceMoE", "version": 1, "d_model": self.d_model,
                "intermediate_size": self.intermediate_size, "num_experts": self.num_experts,
                "top_k": self.top_k, "shared": self.shared is not None,
                "router_bias": self.router.bias is not None}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        expected = {"layer_type", "version", "d_model", "intermediate_size", "num_experts",
                    "top_k", "shared", "router_bias"}
        if set(config) != expected or config["layer_type"] != "ReferenceMoE" or config["version"] != 1:
            _fail("Unsupported MoE constructor")
        dims = [config[k] for k in ("d_model", "intermediate_size", "num_experts", "top_k")]
        if any(type(v) is not int or v < 1 for v in dims):
            _fail("Positive integer MoE dimensions required")
        d, width, experts, _ = dims
        if max(dims) > 4096 or 3*d*width*(experts+int(config["shared"]))+d*experts > 16_000_000:
            _fail("MoE constructor exceeds bounded dimensions", "resource_limit")
        return cls(**{k: config[k] for k in expected - {"layer_type", "version"}}, device=device, dtype=dtype)

    def validate_state(self, state=None, prefix=""):
        state = self.state_dict() if state is None else state
        for key, shape in (("expert_ids", (self.num_experts,)), ("load_counts", (self.num_experts,))):
            tensor = state[prefix + key]
            if tensor.shape != shape or tensor.dtype != torch.long or bool((tensor < 0).any()):
                _fail("Invalid MoE identity/load metadata", "invalid_checkpoint")
        ids = state[prefix + "expert_ids"]
        if ids.unique().numel() != self.num_experts:
            _fail("Duplicate original expert identifiers", "invalid_checkpoint")

    validate_checkpoint_state = validate_state


def remove_experts(module, retained):
    """Copy and physically remove experts and router rows; never mutate source."""
    if type(module) is not ReferenceMoE:
        _fail("Only ReferenceMoE is supported", "unsupported_architecture")
    retained = _ids(retained, module.num_experts, module.top_k)
    result = deepcopy(module)
    index = torch.tensor(retained, device=module.router.weight.device)
    result.experts = nn.ModuleList([deepcopy(module.experts[i]) for i in retained])
    with torch.random.fork_rng(devices=[]):
        result.router = nn.Linear(module.d_model, len(retained), bias=module.router.bias is not None,
                                  device=module.router.weight.device, dtype=module.router.weight.dtype)
    with torch.no_grad():
        result.router.weight.copy_(module.router.weight[index])
        if result.router.bias is not None:
            result.router.bias.copy_(module.router.bias[index])
    result.router.train(module.router.training)
    result.router.weight.requires_grad_(module.router.weight.requires_grad)
    if result.router.bias is not None:
        result.router.bias.requires_grad_(module.router.bias.requires_grad)
    result.expert_ids = module.expert_ids[index].clone()
    result.load_counts = module.load_counts[index].clone()
    result.num_experts = len(retained)
    return result, {"method": "moe_physical_removal", "routing": "reroute_then_renormalize",
                    "retained": list(retained), "original_ids": result.expert_ids.tolist(),
                    "stored_experts": len(retained), "active_experts_per_token": result.top_k,
                    "shared_experts": int(result.shared is not None)}


def adapt_mixtral(module):
    """Explicit transformers 4.44 Mixtral adapter, inference and zero jitter only.

    The returned ReferenceMoE has Tensor output; the caller owns the enclosing
    HF tuple/logit protocol and must not blindly replace the HF block.
    """
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    if type(module) is not MixtralSparseMoeBlock or module.jitter_noise != 0:
        _fail("Only exact zero-jitter MixtralSparseMoeBlock is admitted", "unsupported_architecture")
    first = module.gate.weight
    result = ReferenceMoE(module.hidden_dim, module.ffn_dim, module.num_experts,
                          module.top_k, dtype=first.dtype, device=first.device)
    with torch.no_grad():
        result.router.weight.copy_(module.gate.weight)
        for source, target in zip(module.experts, result.experts):
            if type(source.act_fn) is not nn.SiLU:
                _fail("Only Mixtral SiLU gated experts are admitted", "unsupported_architecture")
            target.gate_proj.weight.copy_(source.w1.weight)
            target.up_proj.weight.copy_(source.w3.weight)
            target.down_proj.weight.copy_(source.w2.weight)
    result.eval()
    return result


def intervention_scores(module, evaluate_loss):
    """Published absolute loss after single removal, plus named delta evidence."""
    if type(module) is not ReferenceMoE or module.num_experts <= module.top_k:
        _fail("Single deletion must leave at least top-k experts")
    baseline = float(evaluate_loss(deepcopy(module)))
    values = []
    for i in range(module.num_experts):
        candidate, _ = remove_experts(module, [j for j in range(module.num_experts) if j != i])
        values.append(float(evaluate_loss(candidate)))
    if not all(math.isfinite(v) for v in (baseline, *values)):
        _fail("Loss callback returned nonfinite evidence", "numerical_failure")
    return {"absolute_losses": values, "layer_sum": sum(values), "layer_mean": sum(values)/len(values),
            "baseline_loss": baseline, "delta_losses": [v-baseline for v in values],
            "evaluations": len(values)+1}


def next_token_loss(logits, token_ids):
    """Explicit mean shifted cross entropy, for published intervention scoring."""
    if (not isinstance(logits,torch.Tensor) or logits.ndim != 3 or logits.shape[1] < 2
            or not isinstance(token_ids,torch.Tensor) or token_ids.dtype != torch.long
            or token_ids.shape != logits.shape[:2] or token_ids.device != logits.device
            or not bool(torch.isfinite(logits).all()) or logits.shape[-1] < 1
            or bool(((token_ids < 0)|(token_ids >= logits.shape[-1])).any())):
        _fail("Finite [batch,tokens,vocabulary] logits and aligned token identities required")
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),token_ids[:,1:].reshape(-1))


@dataclass(frozen=True)
class ExpertCandidate:
    retained: tuple[int, ...]
    error: float


@dataclass(frozen=True)
class MoECompressionSpec:
    retained: tuple[int, ...]
    ranks: tuple[int, ...] | None = None
    ridge: float = 1e-5
    budget_bytes: int | None = None
    max_workspace_bytes: int = 128 * 1024**2

    def __post_init__(self):
        if type(self.retained) is not tuple or not self.retained or any(type(i) is not int or i < 0 for i in self.retained) or len(set(self.retained)) != len(self.retained):
            _fail("Immutable distinct retained expert identities required")
        if self.ranks is not None and (type(self.ranks) is not tuple or len(self.ranks) != len(self.retained)
                or any(type(r) is not int or r < 1 for r in self.ranks)):
            _fail("One positive rank per retained expert required")
        if type(self.ridge) not in (int,float) or not math.isfinite(self.ridge) or self.ridge <= 0:
            _fail("Positive finite ridge required")
        if (self.budget_bytes is not None and (type(self.budget_bytes) is not int or self.budget_bytes < 0)
                or type(self.max_workspace_bytes) is not int or self.max_workspace_bytes < 1):
            _fail("Nonnegative storage budget and positive workspace bound required")


def moe_i2(module, inputs, spec):
    if not isinstance(spec, MoECompressionSpec):
        _fail("MoECompressionSpec required")
    if type(module) is not ReferenceMoE:
        _fail("ReferenceMoE required", "unsupported_architecture")
    retained = _ids(spec.retained,module.num_experts,module.top_k)
    if spec.ranks is not None:
        if (not isinstance(inputs,torch.Tensor) or inputs.ndim < 2
                or inputs.shape[-1] != module.d_model or not inputs.numel()
                or inputs.dtype != module.router.weight.dtype or inputs.device != module.router.weight.device
                or not bool(torch.isfinite(inputs).all())):
            _fail("Compatible finite calibration required before expert removal")
        for i,r in zip(retained,spec.ranks):
            expert = module.experts[i]
            if (type(expert) is not ReferenceMLP or not expert.gated or expert.activation != "silu"
                    or not 1 <= r <= min(module.d_model,expert.intermediate_size)
                    or any(type(getattr(expert,name)) is not nn.Linear for name in ("gate_proj","up_proj","down_proj"))):
                _fail("Original gated SiLU experts and admissible ranks required")
    precision = module.router.weight.element_size()
    # Explicitly account for the local whole MoE, including shared/router/maps.
    fixed = len(retained)*module.d_model*precision + len(retained)*16
    if module.router.bias is not None:
        fixed += len(retained)*precision
    if module.shared is not None:
        fixed += sum(v.numel()*v.element_size() for v in module.shared.state_dict().values())
    if spec.ranks is None:
        cost = fixed + sum(sum(v.numel()*v.element_size() for v in module.experts[i].state_dict().values()) for i in retained)
    else:
        cost = fixed + sum(3*r*(module.d_model+module.experts[i].intermediate_size)*precision
            + sum(layer.bias.numel()*layer.bias.element_size() for layer in
                  (module.experts[i].gate_proj,module.experts[i].up_proj,module.experts[i].down_proj)
                  if layer.bias is not None) for i,r in zip(retained,spec.ranks))
    if spec.budget_bytes is not None and cost > spec.budget_bytes:
        _fail("Whole target MoE exceeds declared budget_bytes", "budget_infeasible")
    source_bytes = sum(v.numel()*v.element_size() for v in module.state_dict().values())
    reserve = source_bytes*2
    if spec.ranks is not None:
        tokens = inputs.numel()//module.d_model
        width = max(module.experts[i].intermediate_size for i in retained)
        reserve += 8*(16*(module.d_model**2+width**2+module.d_model*width)
                     +tokens*(module.d_model+4*width+4*len(retained)))
        if tokens > 65536:
            _fail("Calibration token limit exceeded before expert removal", "resource_limit")
    if reserve > spec.max_workspace_bytes:
        _fail("Expert removal copies exceed workspace bound", "resource_limit")
    result, removal = remove_experts(module, spec.retained)
    if spec.ranks is None:
        return result, {**removal,"target_moe_tensor_bytes":cost}
    result, factorization = compress_experts(result, inputs, spec.ranks, ridge=spec.ridge,
                                             max_workspace_bytes=spec.max_workspace_bytes)
    return result, {"method": "moe_i2_fixed_map_fedcore", "removal": removal,
                    "factorization": factorization,"target_moe_tensor_bytes":cost}


def genetic_expert_candidates(module, inputs, keep, *, best_k=2, population=8,
                              generations=4, max_evaluations=64, seed=0):
    """Bounded seeded elitist crossover/mutation, explicitly an engineering profile.

    Candidate fitness is the sum of unsquared batch Frobenius errors. Unique
    cached evaluations and each generation are retained for exact replay.
    """
    if type(module) is not ReferenceMoE or type(keep) is not int or not module.top_k <= keep <= module.num_experts:
        _fail("Invalid retention cardinality")
    if any(type(v) is not int or v < 1 for v in (best_k,population,generations,max_evaluations)) or best_k > population:
        _fail("Positive bounded genetic-search limits required")
    if best_k > min(max_evaluations,math.comb(module.num_experts,keep)):
        _fail("best_k exceeds available unique evaluation budget")
    if type(seed) is not int or not 0 <= seed < 2**63:
        _fail("Nonnegative seed required")
    batches = (inputs,) if isinstance(inputs,torch.Tensor) else tuple(inputs)
    if not batches:
        _fail("Calibration batches required")
    generator = torch.Generator().manual_seed(seed)
    population = min(population, math.comb(module.num_experts,keep), max_evaluations)
    cache = {}
    history = []
    def random_map():
        return tuple(sorted(torch.randperm(module.num_experts,generator=generator)[:keep].tolist()))
    pool = {tuple(range(keep))}
    for _ in range(population*20):
        if len(pool) >= population:
            break
        pool.add(random_map())
        if len(pool) >= population:
            break
    with torch.no_grad():
        references = [module(x) for x in batches]
        for generation in range(generations):
            for retained in sorted(pool):
                if retained in cache:
                    continue
                if len(cache) >= max_evaluations:
                    break
                candidate,_ = remove_experts(module,retained)
                error = sum(float(torch.linalg.vector_norm(candidate(x)-ref)) for x,ref in zip(batches,references))
                if not math.isfinite(error):
                    _fail("Nonfinite genetic fitness", "numerical_failure")
                cache[retained] = error
            ranked = sorted(cache, key=lambda ids:(cache[ids],ids))
            history.append({"generation":generation,"population":[list(ids) for ids in sorted(pool)],
                            "best":list(ranked[0]),"evaluations":len(cache)})
            if len(cache) >= min(max_evaluations,math.comb(module.num_experts,keep)):
                break
            elite = ranked[:max(1,population//2)]
            pool = set(elite)
            for _ in range(population*20):
                if len(pool) >= population:
                    break
                a = elite[int(torch.randint(len(elite),(1,),generator=generator))]
                b = elite[int(torch.randint(len(elite),(1,),generator=generator))]
                union = sorted(set(a+b))
                order = torch.randperm(len(union),generator=generator)[:keep].tolist()
                child = {union[i] for i in order}
                available = sorted(set(range(module.num_experts))-child)
                if available:
                    old = sorted(child)[int(torch.randint(keep,(1,),generator=generator))]
                    new = available[int(torch.randint(len(available),(1,),generator=generator))]
                    child.remove(old); child.add(new)
                pool.add(tuple(sorted(child)))
                if len(pool) >= population:
                    break
    records = sorted((ExpertCandidate(ids,error) for ids,error in cache.items()),key=lambda c:(c.error,c.retained))
    return tuple(records[:best_k]), {"search":"fedcore_bounded_elitist_genetic_v1",
           "global_optimality":False,"seed":seed,"evaluations":len(cache),"history":history,
           "generator_state":generator.get_state().tolist(),
           "candidates":[{"retained":list(c.retained),"error":c.error} for c in records]}


def expert_candidates(module, inputs, keep, *, best_k=2, max_evaluations=256, seed=0):
    """Enumerate small domains, seeded subset search otherwise; sum of norms.

    This explicitly named bounded engineering search is not a claim that the
    paper's incompletely specified genetic heuristic is reproduced.
    """
    if type(module) is not ReferenceMoE or type(keep) is not int or not module.top_k <= keep <= module.num_experts:
        _fail("Invalid expert retention cardinality")
    if any(type(v) is not int or v < 1 for v in (best_k, max_evaluations)) or best_k > max_evaluations:
        _fail("Positive bounded search limits required")
    if type(seed) is not int or seed < 0:
        _fail("Nonnegative seed required")
    count = math.comb(module.num_experts, keep)
    generator = torch.Generator().manual_seed(seed)
    if count <= max_evaluations:
        maps = list(itertools.combinations(range(module.num_experts), keep))
    else:
        maps = set()
        # Bound random attempts too; no hidden unbounded coupon collector.
        for _ in range(max_evaluations * 20):
            maps.add(tuple(sorted(torch.randperm(module.num_experts, generator=generator)[:keep].tolist())))
            if len(maps) == max_evaluations:
                break
        maps = sorted(maps)
    batches = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not batches:
        _fail("Nonempty calibration batches required")
    with torch.no_grad():
        references = [module(x) for x in batches]
        records = []
        for retained in maps:
            candidate, _ = remove_experts(module, retained)
            error = sum(float(torch.linalg.vector_norm(candidate(x)-y)) for x, y in zip(batches, references))
            if not math.isfinite(error):
                _fail("Nonfinite candidate error", "numerical_failure")
            records.append(ExpertCandidate(retained, error))
    records.sort(key=lambda c: (c.error, c.retained))
    return tuple(records[:best_k]), {"search": "bounded_exhaustive_or_seeded_subset",
           "global_optimality": count <= max_evaluations, "evaluations": len(records),
           "seed": seed, "generator_state": generator.get_state().tolist(),
           "candidates": [{"retained": list(c.retained), "error": c.error} for c in records]}


def kt_block_search(candidate_tables, evaluate, *, max_evaluations=256):
    """Joint K^T map selection, not knowledge transfer or distillation."""
    tables = tuple(tuple(t) for t in candidate_tables)
    if not tables or any(not t for t in tables) or type(max_evaluations) is not int or max_evaluations < 1:
        _fail("Nonempty candidate tables and positive evaluation bound required")
    count = math.prod(map(len, tables))
    if count > max_evaluations:
        _fail("K^T combinations exceed explicit search bound", "resource_limit")
    records = []
    for combination in itertools.product(*tables):
        maps = tuple(c.retained for c in combination)
        loss = float(evaluate(maps))
        if not math.isfinite(loss):
            _fail("Nonfinite KT score", "numerical_failure")
        records.append((loss, maps))
    records.sort()
    return records[0][1], {"method": "blockwise_kt_reception_field", "evaluations": count,
                           "records": [{"loss": loss, "maps": maps} for loss, maps in records]}


def allocate_expert_ranks(importances, caps, budget, *, epsilon=1e-8):
    """Named floor + capped largest-deficit integer allocation, exponent 0.15.

    Budget is a sum of ranks, not bytes. Heterogeneous projection byte costs
    must be checked separately before committing the model.
    """
    values, caps = tuple(importances), tuple(caps)
    if (not values or len(values) != len(caps) or any(type(c) is not int or c < 1 for c in caps)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or type(budget) is not int or not len(caps) <= budget <= sum(caps)
            or type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0):
        _fail("Finite nonnegative importance and feasible rank budget required")
    weights = [(v+epsilon)**0.15 for v in values]
    total = sum(weights)
    targets = [budget*w/total for w in weights]
    ranks = [min(c, max(1, math.floor(t))) for c,t in zip(caps, targets)]
    while sum(ranks) > budget:
        i = min((i for i,r in enumerate(ranks) if r > 1), key=lambda i:(targets[i]-ranks[i], i))
        ranks[i] -= 1
    while sum(ranks) < budget:
        i = max((i for i,r in enumerate(ranks) if r < caps[i]), key=lambda i:(targets[i]-ranks[i], -i))
        ranks[i] += 1
    return tuple(ranks)


def allocate_layer_deletions(layer_sums, capacities, total_deletions):
    """Explicit inverse-sensitivity engineering allocation, not a paper formula."""
    values, caps = tuple(layer_sums), tuple(capacities)
    if (not values or len(values) != len(caps)
            or any(type(c) is not int or c < 0 for c in caps)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values)
            or type(total_deletions) is not int or not 0 <= total_deletions <= sum(caps)):
        _fail("Finite nonnegative sensitivities and feasible deletion budget required")
    result = [0]*len(caps)
    weights = [1/(v+1e-8) for v in values]
    targets = [total_deletions*w/sum(weights) for w in weights]
    for _ in range(total_deletions):
        i = max((i for i in range(len(caps)) if result[i] < caps[i]),
                key=lambda i:(targets[i]-result[i],-i))
        result[i] += 1
    return tuple(result)


def prepare_expert_recovery(module, *, factor, adapter_rank, alpha=1.0):
    """Reuse existing independent factor LoRA; caller owns train_model and data.

    Compression rank and adapter rank are independent. Finalize with existing
    finish_factor_lora_stage to merge into compact factors, never dense W.
    """
    from fedcore.algorithm.low_rank.factor_recovery import prepare_factor_lora_stage
    if type(module) is not ReferenceMoE:
        _fail("ReferenceMoE required", "unsupported_architecture")
    paths = []
    for i, expert in enumerate(module.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            if type(getattr(expert,projection,None)) is not FactorizedLinear:
                _fail("Factorized experts required before recovery")
            paths.append(f"experts.{i}.{projection}")
    return prepare_factor_lora_stage(module,paths,factor=factor,rank=adapter_rank,
                                     alpha=alpha,adapter_name="moe_i2_fedcore")


def compress_experts(module, inputs, ranks, *, ridge=1e-5, max_tokens=65536,
                     max_workspace_bytes=128 * 1024**2):
    """Weighted SVD on actual assigned gate/up and post-product down inputs."""
    if type(module) is not ReferenceMoE or len(ranks) != module.num_experts:
        _fail("One explicit compression rank per retained expert required")
    if (not isinstance(inputs,torch.Tensor) or inputs.ndim < 2 or inputs.shape[-1] != module.d_model
            or inputs.dtype != module.router.weight.dtype or inputs.device != module.router.weight.device
            or type(max_tokens) is not int or max_tokens < 1
            or type(max_workspace_bytes) is not int or max_workspace_bytes < 1):
        _fail("Compatible CPU calibration and positive resource bounds required")
    flat = inputs.reshape(-1, module.d_model)
    if flat.shape[0] < 1 or flat.shape[0] > max_tokens or not bool(torch.isfinite(flat).all()):
        _fail("Finite bounded nonempty calibration required", "resource_limit")
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
        _fail("Positive explicit regularization required for rare experts")
    for expert, rank in zip(module.experts, ranks):
        if (type(expert) is not ReferenceMLP or not expert.gated or expert.activation != "silu"
                or type(rank) is not int or not 1 <= rank <= min(module.d_model, expert.intermediate_size)
                or any(type(getattr(expert,name)) is not nn.Linear for name in ("gate_proj","up_proj","down_proj"))):
            _fail("Only original ReferenceMLP experts and admissible ranks supported")
    width = max(e.intermediate_size for e in module.experts)
    source_bytes = sum(v.numel()*v.element_size() for v in module.state_dict().values())
    workspace = 2*source_bytes + 8*(16*(module.d_model**2+width**2+module.d_model*width)
                   +len(flat)*(module.d_model+4*width+4*module.num_experts))
    if workspace > max_workspace_bytes:
        _fail("Expert factorization exceeds workspace bound before copying/statistics", "resource_limit")
    with torch.no_grad():
        _, assigned, _ = module.route(inputs)
        coverage = [(assigned == i).any(dim=-1).nonzero().flatten() for i in range(module.num_experts)]
        if any(not tokens.numel() for tokens in coverage):
            _fail("Unobserved expert requires additional calibration", "insufficient_coverage")
        result = deepcopy(module)
        evidence = []
        for i, (source, target, rank) in enumerate(zip(module.experts, result.experts, ranks)):
            tokens = coverage[i]
            x = flat[tokens]
            features = F.silu(source.gate_proj(x))*source.up_proj(x)
            for name, calibration in (("gate_proj", x), ("up_proj", x), ("down_proj", features)):
                layer = getattr(source, name)
                calibration64 = calibration.to(torch.float64)
                gram = calibration64.T @ calibration64 / len(calibration64)
                solved = solve_weighted(layer.weight, gram, rank, MetricPolicy(ridge=ridge))
                pair = FactorPair(solved.u*solved.s, solved.vh)
                setattr(target, name, FactorizedLinear.from_factors(pair, layer.bias))
            evidence.append({"expert": i, "original_id": int(module.expert_ids[i]),
                             "assigned_tokens": len(tokens), "rank": rank})
    return result, {"method": "moe_weighted_svd_fedcore", "numerical_version": "solve_weighted_support_only_v1",
                    "ridge": ridge, "router_unchanged": True, "experts": evidence,
                    "stored_experts": result.num_experts, "active_experts_per_token": result.top_k,
                    "workspace_reserved_bytes":workspace}
