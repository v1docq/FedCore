"""Weight-only PruneNet with a consistent finite-horizon REINFORCE objective.

This is the named ``paper_left_no_bias_finite_cost_v1`` correction of the
2501.15296v3 mechanism, not reproduction of the divergent author main branch.
The ordered action and its exact conditional noise are retained together.
Policy parameters never become part of the exported compact FFN.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .channels import inspect_mlp, slice_mlp


AUTHOR_MAIN_DIFFERENCES = (
    ("projection", "paper: q^T W A^T, without bias", "author main: W A^T q, with bias"),
    ("objective", "minimize finite expected KS cost", "reciprocal KS reward with zero-distance special case"),
    ("return", "sum future discounted costs including current layer", "discount individual rewards"),
    ("action_probability", "ordered conditional probability without replacement", "sum unnormalized selected log weights"),
    ("noise", "reuse the action's exact sampled noise", "a fresh policy call before the loss"),
)


@dataclass(frozen=True)
class PruneNetSpec:
    keep_count: int = 1
    episodes: int = 4
    learning_rate: float = 5e-4
    discount: float = .99
    seed: int = 0
    policy_variant: str = "paper_left_no_bias"
    max_workspace_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if (type(self.keep_count) is not int or self.keep_count < 1
                or type(self.episodes) is not int or not 1 <= self.episodes <= 20
                or type(self.seed) is not int or self.seed < 0
                or type(self.learning_rate) not in (int, float) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or type(self.discount) not in (int, float) or not math.isfinite(self.discount) or not 0 <= self.discount <= 1
                or self.policy_variant != "paper_left_no_bias"
                or type(self.max_workspace_bytes) is not int or self.max_workspace_bytes < 1):
            raise ValueError("Invalid named PruneNet finite-cost profile")


class WeightPolicy(nn.Module):
    """Equation (3)'s LEFT projection, with its original parameterization.

    (q^T W) A^T avoids an h*h temporary without changing parameters or their
    gradients. A code-main right projection is not silently substituted.
    """
    def __init__(self, width, d_model, *, generator=None, dtype=torch.float64):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (width, d_model)) or width*d_model+width > 1_000_000:
            raise ValueError("PruneNet bounded policy dimension limit exceeded before allocation")
        self.width, self.d_model = width, d_model
        self.intermediate = nn.Parameter(torch.randn((width, d_model), generator=generator, dtype=dtype)/math.sqrt(d_model))
        self.projection = nn.Parameter(torch.randn((width,), generator=generator, dtype=dtype)/math.sqrt(width))

    def forward(self, weight):
        if weight.shape != (self.width, self.d_model):
            raise ValueError("PruneNet policy state must match its fixed weight dimensions")
        # No gradient path from a policy state into the frozen source model.
        state = weight.detach().to(self.intermediate)
        logits = (self.projection @ state) @ self.intermediate.T
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("PruneNet policy logits exceed the finite numerical domain")
        return logits


@dataclass(frozen=True)
class OrderedAction:
    order: tuple[int, ...]
    noise: torch.Tensor
    log_probability: torch.Tensor
    log_weights: torch.Tensor


def _log_weights(logits, noise):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 1 or not logits.numel()
            or logits.dtype not in (torch.float32, torch.float64) or logits.device.type != "cpu"
            or not bool(torch.isfinite(logits).all()) or not isinstance(noise, torch.Tensor)
            or noise.shape != logits.shape or noise.dtype != logits.dtype or noise.device != logits.device
            or not bool(torch.isfinite(noise).all()) or bool(((noise <= 0) | (noise >= 1)).any())):
        raise ValueError("PruneNet noise must be finite compatible and strictly inside (0,1)")
    noisy_logits = logits + noise.log() - torch.log1p(-noise)
    result = F.logsigmoid(noisy_logits)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("PruneNet transformed probabilities exceed the finite numerical domain")
    return result


def ordered_log_probability(logits, noise, order):
    """Probability of exactly this ordered weighted action, conditional on noise."""
    logs = _log_weights(logits, noise)
    if (not isinstance(order, (tuple, list)) or not order or len(set(order)) != len(order)
            or any(type(i) is not int or not 0 <= i < len(logits) for i in order)):
        raise ValueError("Ordered action must contain distinct valid channels")
    available = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
    terms = []
    for index in order:
        terms.append(logs[index] - torch.logsumexp(logs.masked_fill(~available, -torch.inf), 0))
        available = available.clone()
        available[index] = False
    return torch.stack(terms).sum()


def sample_ordered_action(logits, keep_count, *, generator, noise=None):
    if type(keep_count) is not int or not 1 <= keep_count <= logits.numel():
        raise ValueError("PruneNet action cardinality must fit the FFN width")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("PruneNet sampling requires an explicit CPU generator")
    if noise is None:
        epsilon = torch.finfo(logits.dtype).eps
        noise = torch.rand(logits.shape, generator=generator, dtype=logits.dtype).clamp(epsilon, 1-epsilon)
    noise = noise.detach().clone()
    logs = _log_weights(logits, noise)
    available = torch.ones(len(logits), dtype=torch.bool)
    order = []
    for _ in range(keep_count):
        candidate = logs.detach().masked_fill(~available, -torch.inf)
        probabilities = torch.softmax(candidate, 0)
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        order.append(index)
        available[index] = False
    order = tuple(order)
    return OrderedAction(order, noise, ordered_log_probability(logits, noise, order), logs)


def ks_distance(left, right):
    """Two empirical CDFs of actual SVD lists; missing zero values are not added."""
    if (any(not isinstance(x, torch.Tensor) or x.ndim != 1 or not x.numel()
            or not bool(torch.isfinite(x).all()) or bool((x < 0).any()) for x in (left, right))):
        raise ValueError("KS requires two nonempty finite nonnegative singular-value lists")
    a, b = left.detach().double().sort().values, right.detach().double().sort().values
    thresholds = torch.cat((a, b)).sort().values
    fa = torch.searchsorted(a, thresholds, right=True).double()/len(a)
    fb = torch.searchsorted(b, thresholds, right=True).double()/len(b)
    return float((fa-fb).abs().max())


def spectral_cost(weight, order, *, reference_spectrum=None):
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.numel():
        raise ValueError("Spectral cost requires a nonempty matrix")
    original = torch.linalg.svdvals(weight.detach().double()) if reference_spectrum is None else reference_spectrum
    candidate = weight.detach().index_select(0, torch.tensor(order, dtype=torch.long))
    selected = torch.linalg.svdvals(candidate.double())
    return ks_distance(original, selected)


def future_costs(costs, discount):
    if (not isinstance(costs, (tuple, list)) or not costs
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in costs)
            or type(discount) not in (int, float) or not math.isfinite(discount) or not 0 <= discount <= 1):
        raise ValueError("Future costs require finite nonnegative costs and a discount in [0,1]")
    accumulated, result = 0., []
    for cost in reversed(costs):
        accumulated = float(cost) + discount*accumulated
        if not math.isfinite(accumulated):
            raise ValueError("Future cost exceeds the finite numerical domain")
        result.append(accumulated)
    return tuple(reversed(result))


def reinforce_cost_loss(actions, costs, discount):
    if len(actions) != len(costs):
        raise ValueError("One cost is required for each ordered layer action")
    returns = future_costs(costs, discount)
    # Positive cost: Adam gradient descent MINIMIZES the expected discounted cost.
    return sum(discount**i * value * action.log_probability for i, (value, action) in enumerate(zip(returns, actions)))


def train_prunenet_policy(modules, spec=PruneNetSpec()):
    if type(spec) is not PruneNetSpec or not isinstance(modules, (tuple, list)) or not modules:
        raise ValueError("PruneNetSpec and a nonempty explicit FFN sequence required")
    views = tuple(inspect_mlp(module) for module in modules)
    h, d = views[0].width, views[0].d_model
    if any((view.width, view.d_model) != (h, d) for view in views) or spec.keep_count > h:
        raise ValueError("One PruneNet policy requires compatible fixed FFN widths and input dimensions")
    reserve = 8*(8*h*d + 4*h + 16*max(h, d)**2 + len(views)*(h*d + min(h, d)))
    if reserve > spec.max_workspace_bytes:
        raise ValueError("PruneNet workspace limit exceeded before policy/SVD allocation")
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    policy = WeightPolicy(h, d, generator=generator)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=spec.learning_rate, weight_decay=0.)
    # Llama's policy state is explicitly gate; ordinary FFN uses up.
    states = tuple((view.gate if view.gated else view.up).weight.detach().double() for view in views)
    original_spectra = tuple(torch.linalg.svdvals(state) for state in states)
    before = tuple(p.detach().clone() for p in policy.parameters())
    history = []
    for episode in range(spec.episodes):
        optimizer.zero_grad(set_to_none=True)
        actions = tuple(sample_ordered_action(policy(state), spec.keep_count, generator=generator) for state in states)
        costs = tuple(spectral_cost(state, action.order, reference_spectrum=spectrum)
                      for state, action, spectrum in zip(states, actions, original_spectra))
        loss = reinforce_cost_loss(actions, costs, spec.discount)
        if not bool(torch.isfinite(loss)):
            raise ValueError("PruneNet policy loss is nonfinite")
        loss.backward()
        parameters = tuple(policy.parameters())
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("PruneNet policy backward must produce finite gradients")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise ValueError("PruneNet policy optimizer produced nonfinite weights")
        history.append({"episode": episode, "costs": costs, "future_costs": future_costs(costs, spec.discount),
                        "orders": tuple(a.order for a in actions), "loss": float(loss.detach())})
    with torch.no_grad():
        final_actions = tuple(sample_ordered_action(policy(state), spec.keep_count, generator=generator) for state in states)
    evidence = {"method": "prunenet", "source_profile": "2501.15296v3-independent-bounded-v1",
                "corrected_objective": "paper_left_no_bias_finite_cost_v1", "policy_variant": spec.policy_variant,
                "author_main_revision": "unresolved-not-copied", "author_main_differences": AUTHOR_MAIN_DIFFERENCES,
                "selection_data": "weights_only", "calibration_used": False, "source_state_projection": "gate_if_gated_else_up",
                "original_spectrum_lengths": tuple(len(x) for x in original_spectra),
                "optimizer_steps": spec.episodes, "history": history,
                "policy_changed": any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())),
                "policy_parameters": sum(p.numel() for p in policy.parameters()),
                "final_orders": tuple(a.order for a in final_actions), "discount": spec.discount,
                "action_probability": "ordered_without_replacement_conditional_on_same_noise",
                "finite_minimization_sign": "positive_cost_times_log_probability",
                "workspace_reserved_bytes": reserve, "policy_exported": False,
                "spectral_cost_is_quality_guarantee": False}
    return policy, evidence


def prunenet(module, spec=PruneNetSpec()):
    _, evidence = train_prunenet_policy((module,), spec)
    result, selection = slice_mlp(module, evidence["final_orders"][0])
    evidence.update(selection=selection, selected_current_indices=result.current_indices,
                    selected_original_indices=result.channel_indices, compact_width=result.intermediate_size)
    return result, evidence
