"""Dobi-SVD objective and deterministic CPU spectral algebra, without a runner.

Paper: arXiv:2502.02723v1, Algorithm 1 (activation tanh cutoff) and
Algorithm 2 (sequential direction aggregation). This implementation explicitly
reconstructs two numerically ambiguous choices: equal gates inside a near
singular cluster, and uncentered incremental direction PCA. It does not copy
the author's Taylor surrogate backward or claim global activation-MSE
optimality for IPCA. Hard export also differs from the author's soft projector.
"""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from .approximation import NumericalDomainViolation
from .dobi_specs import DobiSVD
from .projection_recovery import observed_loss
from .structured_layers import FactorizedLinear
from .structured_profiles import FactorPair
from .topology import module_paths, replace_modules_atomically, validate_parameter_topology


DOBI_SOURCE = "https://arxiv.org/html/2502.02723v1#S3.SS1"
DOBI_REPOSITORY_COMMIT = "220ab55026dccfa520b68130d8ce88736a24e055"
DOBI_REPOSITORY = "https://github.com/wangqinsi1/Dobi-SVD/tree/" + DOBI_REPOSITORY_COMMIT


def _matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.numel()
            or value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())):
        raise NumericalDomainViolation(f"{name} requires a nonempty finite CPU float32/float64 matrix")


def _clusters(s, rtol, atol):
    """Contiguous single-link clusters; membership is held fixed in backward."""
    groups = []
    start = 0
    for i in range(1, len(s)):
        if float(s[i - 1] - s[i]) > atol + rtol * float(s[i - 1]):
            groups.append((start, i))
            start = i
    groups.append((start, len(s)))
    return tuple(groups)


def _gate_data(s, k, beta, rtol, atol, policy):
    groups = _clusters(s, rtol, atol)
    if policy == "error" and (any(end - start > 1 for start, end in groups)
                               or float(s[-1]) <= atol + rtol * float(s[0])):
        raise NumericalDomainViolation("Dobi-SVD error policy refuses near/repeated or near-zero spectrum")
    index = torch.arange(1, len(s) + 1, dtype=s.dtype)
    t = torch.tanh(beta * (k - index))
    gate = (1 + t) * 0.5
    derivative = 0.5 * beta * (1 - t.square())
    labels = torch.empty(len(s), dtype=torch.int64)
    for number, (start, end) in enumerate(groups):
        gate[start:end] = gate[start:end].mean()
        derivative[start:end] = derivative[start:end].mean()
        labels[start:end] = number
    return gate, derivative, labels


class _ActivationTruncation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, k, beta, rtol, atol, policy):
        u, s, vh = torch.linalg.svd(a.double(), full_matrices=False)
        gate, derivative, groups = _gate_data(s, k.double(), beta, rtol, atol, policy)
        ctx.save_for_backward(u, s, vh, gate, derivative, groups)
        ctx.input_dtype, ctx.k_dtype = a.dtype, k.dtype
        return ((u * (s * gate)) @ vh).to(a.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        u, s, vh, gate, derivative, groups = ctx.saved_tensors
        g = gradient.double()
        core = u.T @ g @ vh.T
        si, sj = s[:, None], s[None, :]
        fi, fj = (s * gate)[:, None], (s * gate)[None, :]
        same = groups[:, None] == groups[None, :]
        # For a shared gate c, f(s)=c*s on the whole cluster, so both divided
        # differences are exactly c, including zero/zero. Outside a cluster
        # the denominators have the declared nonzero gap. No epsilon division
        # and no derivative of arbitrarily chosen singular vectors is needed.
        p = torch.where(same, gate[:, None], (fi - fj) / torch.where(same, torch.ones_like(si - sj), si - sj))
        zero_sum = si + sj == 0
        q = torch.where(same, gate[:, None], (fi + fj) / torch.where(zero_sum, torch.ones_like(si + sj), si + sj))
        sym, skew = (core + core.T) * 0.5, (core - core.T) * 0.5
        result = u @ (p * sym + q * skew) @ vh
        # Thin rectangular SVD has complement terms, often omitted in square
        # formulas. f(s)/s=c remains defined at zero under cluster_equal.
        result += ((g @ vh.T - u @ core) * gate) @ vh
        result += (u * gate) @ (u.T @ g - core @ vh)
        dk = (core.diagonal() * s * derivative).sum()
        return result.to(ctx.input_dtype), dk.to(ctx.k_dtype), None, None, None, None


def smooth_activation_truncation(a, k, *, beta=5.0, spectrum_policy="cluster_equal",
                                 spectrum_rtol=1e-6, spectrum_atol=1e-12):
    """Differentiate U diag(s_i tanh_gate(k-i)) V.T with respect to A and k.

    Exact for distinct separated spectra. With ``cluster_equal``, exact locally
    for the *modified* map whose gates are averaged within fixed clusters;
    cluster membership boundaries are nonsmooth. Higher derivatives are not
    supported. This finite-beta map generally retains full algebraic rank.
    """
    _matrix(a, "activation")
    if (not isinstance(k, torch.Tensor) or k.ndim != 0 or k.device.type != "cpu"
            or k.dtype != a.dtype or not bool(torch.isfinite(k))):
        raise NumericalDomainViolation("cutoff k must be a finite CPU scalar of activation dtype")
    DobiSVD(beta=beta, spectrum_policy=spectrum_policy,
            spectrum_rtol=spectrum_rtol, spectrum_atol=spectrum_atol)
    return _ActivationTruncation.apply(a, k, beta, spectrum_rtol, spectrum_atol, spectrum_policy)


def _canonical_rows(vh):
    indices = vh.abs().argmax(dim=1)
    signs = torch.sign(vh[torch.arange(len(vh)), indices])
    return vh * torch.where(signs == 0, torch.ones_like(signs), signs)[:, None]


def _digest(value):
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class DirectionIPCAState:
    """Ordered uncentered truncated-SVD summary, retaining no original batches."""
    dimension: int
    rank: int
    components: torch.Tensor
    singular_values: torch.Tensor
    rows_seen: int = 0
    batch_hashes: tuple[str, ...] = ()

    def __post_init__(self):
        if (type(self.dimension) is not int or type(self.rank) is not int
                or not 0 <= self.rank <= self.dimension or self.dimension < 1
                or type(self.rows_seen) is not int or self.rows_seen < 0
                or not isinstance(self.batch_hashes, tuple)
                or any(not isinstance(value, str) or len(value) != 64 for value in self.batch_hashes)):
            raise NumericalDomainViolation("Invalid ordered direction IPCA state metadata")
        if (not isinstance(self.components, torch.Tensor) or not isinstance(self.singular_values, torch.Tensor)
                or self.components.device.type != "cpu" or self.singular_values.device.type != "cpu"
                or self.components.dtype != torch.float64 or self.singular_values.dtype != torch.float64
                or self.components.ndim != 2 or self.components.shape[1] != self.dimension
                or len(self.components) > self.rank or self.singular_values.shape != (len(self.components),)
                or not bool(torch.isfinite(self.components).all())
                or not bool(torch.isfinite(self.singular_values).all())
                or bool((self.singular_values < 0).any())):
            raise NumericalDomainViolation("Invalid direction IPCA summary tensors")


def empty_direction_ipca(dimension, rank):
    if type(dimension) is not int or type(rank) is not int or not 0 <= rank <= dimension or dimension < 1:
        raise NumericalDomainViolation("IPCA needs positive dimension and rank in [0, dimension]")
    return DirectionIPCAState(dimension, rank, torch.empty((0, dimension), dtype=torch.float64),
                              torch.empty(0, dtype=torch.float64))


def update_direction_ipca(state, directions):
    """Pure state transition: concatenate S_old V_old.T and the next row block.

    Truncation after every update makes this generally order-dependent. It is
    not equivalent to full-batch PCA and does not minimize aggregate ||X E||.
    Input directions use deterministic canonical signs because they originate
    in SVD; covariance/projector semantics do not depend on those signs.
    """
    if not isinstance(state, DirectionIPCAState):
        raise NumericalDomainViolation("An explicit DirectionIPCAState is required")
    _matrix(directions, "IPCA directions")
    if directions.shape[1] != state.dimension:
        raise NumericalDomainViolation("IPCA direction dimension changed")
    block = directions.detach().double().clone()
    if state.rank:
        summary = state.singular_values[:, None] * state.components
        _, s, vh = torch.linalg.svd(torch.cat((summary, block)), full_matrices=False)
        components = _canonical_rows(vh[:state.rank]).clone()
        values = s[:state.rank].clone()
    else:
        components, values = state.components.clone(), state.singular_values.clone()
    return DirectionIPCAState(state.dimension, state.rank, components, values,
                              state.rows_seen + len(block), state.batch_hashes + (_digest(block),))


def fixed_output_factors(weight, basis):
    """W[out,in] -> V[out,k] and V.T@W[k,in], preserving X W.T V V.T."""
    _matrix(weight, "weight")
    _matrix(basis, "output basis")
    if (basis.shape[0] != weight.shape[0] or basis.shape[1] > min(weight.shape)
            or not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-7, rtol=1e-7)):
        raise NumericalDomainViolation("Output basis must have compatible orthonormal columns")
    v = basis.to(weight)
    return FactorPair(v.clone(), v.T @ weight)


def _selected(model, paths, spec):
    if type(spec) is not DobiSVD:
        raise NumericalDomainViolation("DobiSVD options are required")
    if (not isinstance(model, nn.Module) or not isinstance(paths, (tuple, list)) or not paths
            or any(not isinstance(p, str) for p in paths) or len(set(paths)) != len(paths)):
        raise NumericalDomainViolation("Dobi-SVD requires explicit unique Linear paths")
    modules = module_paths(model)
    if any(path not in modules or type(modules[path]) is not nn.Linear for path in paths):
        raise NumericalDomainViolation("Dobi-SVD targets must be exact standard Linear modules")
    topology = validate_parameter_topology(model, paths)
    def affected(parameter_path):
        owner = parameter_path.rpartition(".")[0]
        return any(path == "" or owner == path or owner.startswith(path + ".") for path in paths)
    if (any(set(group) & set(paths) for group in topology.module_aliases)
            or any(any(affected(path) for path in group) for group in topology.parameter_aliases)
            or any(any(affected(path) for path in group.parameter_paths) for group in topology.storage_aliases)):
        raise NumericalDomainViolation("Dobi-SVD independent cutoffs reject aliases and storage views")
    for value in (*model.parameters(), *model.buffers()):
        if value.device.type != "cpu" or (value.is_floating_point() and
                (value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all()))):
            raise NumericalDomainViolation("Dobi-SVD requires a finite private CPU float32/float64 model")
    return tuple((path, modules[path]) for path in paths)


def estimate_dobi_bytes(model, paths, spec):
    """Named tensor lower bounds before allocation; excludes backend SVD workspace.

    SVD uses float64 irrespective of the source weight dtype. The bounds include
    retained activations, thin SVD factors, backward matrices and Adam's two
    scalar states. Unselected model activations and backend workspaces remain
    unknown: their omission must not be presented as an upper/peak bound.
    """
    selected = _selected(model, paths, spec)
    r = spec.max_activation_rows
    dims = [(layer.in_features, layer.out_features, layer.weight.element_size()) for _, layer in selected]
    costs = {
        "cutoff_parameter_bytes": sum(size for _, _, size in dims),
        "cutoff_gradient_bytes": sum(size for _, _, size in dims),
        "adam_scalar_state_lower_bound_bytes": sum(2 * size + 4 for _, _, size in dims),
        "activation_and_output_lower_bound_bytes": sum(2 * r * out * size + 8 * r * out for _, out, size in dims),
        "svd_saved_tensors_lower_bound_bytes": sum(8 * (r * min(r, out) + min(r, out) * out + 3 * min(r, out)) for _, out, _ in dims),
        "svd_backward_lower_bound_bytes": sum(8 * (3 * r * out + 8 * min(r, out)**2) for _, out, _ in dims),
        "ipca_state_lower_bound_bytes": sum(8 * min(i, o) * (o + 1) for i, o, _ in dims),
        "ipca_svd_lower_bound_bytes": sum(8 * (2 * min(i, o) * o + 3 * min(i, o)**2) for i, o, _ in dims),
        "export_factor_lower_bound_bytes": sum(size * min(i, o) * (i + o) for i, o, size in dims),
    }
    if spec.remapping != "none":
        from .dobi_remapping import remapped_tensor_bytes
        costs.update(
            remap_resvd_and_reconstruction_lower_bound_bytes=sum(8*(3*i*o + min(i,o)*(i+o+1)) for i,o,_ in dims),
            remap_encoding_and_decode_lower_bound_bytes=sum(8*(3*min(i,o)**2+min(i,o)*(i+o)+2*min(i,o)) for i,o,_ in dims),
            remap_calibration_error_lower_bound_bytes=sum(8*r*(i+4*o) for i,o,_ in dims),
            remap_storage_lower_bound_bytes=sum(remapped_tensor_bytes(i,o,min(i,o),
                tail_dtype=spec.remap_tail_dtype,scale_dtype=spec.remap_scale_dtype,bias=False) for i,o,_ in dims))
    return costs


class _DobiLinear(nn.Module):
    @property
    def weight(self):
        from .learned_layers import UnsupportedLearnedAccessError
        raise UnsupportedLearnedAccessError('P4 requires module calls; functional dense weight access is unsupported')

    def __init__(self, base, spec):
        super().__init__()
        self.base, self.spec = base, spec
        self.in_features, self.out_features = base.in_features, base.out_features
        self.maximum_rank = min(self.in_features, self.out_features)
        self.k = nn.Parameter(base.weight.new_tensor(spec.initial_fraction * self.maximum_rank))
        self.export_mode = False
        self.calls = 0
        self.last_groups = ()

    def activation(self, x):
        # Bias is preserved outside the output projection. The paper's weight
        # transfer identity is for XW, not an affine column silently folded in.
        if not isinstance(x,torch.Tensor) or x.ndim < 2 or not x.numel() or x.shape[-1] != self.in_features:
            raise NumericalDomainViolation("Dobi-SVD requires nonempty compatible activation rows")
        if x.numel() // self.in_features > self.spec.max_activation_rows:
            raise NumericalDomainViolation("ResourceLimitExceeded: Dobi-SVD activation rows")
        return F.linear(x, self.base.weight, None)

    def forward(self, x):
        a = self.activation(x)
        if a.ndim < 2 or not a.numel():
            raise NumericalDomainViolation("Dobi-SVD requires at least one activation row")
        rows = a.reshape(-1, self.out_features)
        if len(rows) > self.spec.max_activation_rows:
            raise NumericalDomainViolation("ResourceLimitExceeded: Dobi-SVD activation rows")
        self.calls += 1
        if self.export_mode:
            result = a
        else:
            result = smooth_activation_truncation(rows, self.k, beta=self.spec.beta,
                spectrum_policy=self.spec.spectrum_policy, spectrum_rtol=self.spec.spectrum_rtol,
                spectrum_atol=self.spec.spectrum_atol).reshape_as(a)
        return result if self.base.bias is None else result + self.base.bias


class DobiObjective:
    """Owned-model objective; the existing training boundary owns all updates."""
    def __init__(self, model, layers, spec, costs):
        self.model, self.layers, self.spec = model, layers, spec
        self.costs = costs
        self.active = False
        self.calls = self.steps = 0
        self.finalized = False
        self.evidence = {
            "objective": "activation_tanh_task_plus_factor_budget", "source": DOBI_SOURCE,
            "repository": DOBI_REPOSITORY,
            "repository_pin_status": "verified_with_github_fetch_file_at_commit",
            "source_blob_sha": {"modules/stable_svd.py": "e15c1274e7e0534f62764a18b122ff2ebe1ee738",
                "modules/IncrementalPCA.py": "20fbdaf352fa1f76265a232fc2541576164ff2f3",
                "weight_updater.py": "5284571e67e321f12327152a5ecb51f42812a3bf"},
            "variant": spec.variant, "derivative_variant": "fedcore_cluster_spectral_frechet_v1",
            "spectrum_policy": spec.spectrum_policy, "spectrum_rtol": spec.spectrum_rtol,
            "spectrum_atol": spec.spectrum_atol, "cluster_boundary_derivative": "not_defined_at_membership_changes",
            "paper_taylor_surrogate_reproduced": False, "finite_beta_is_exact_rank": False,
            "budget_variant": "fedcore_unremapped_two_factor_weight_elements_v1",
            "budget_denominator": "selected_original_weight_elements; bias_excluded",
            "target_fraction_is_hard_guarantee": False, "target_fraction": spec.target_fraction,
            "initial_rank_fraction": spec.initial_fraction, "beta": spec.beta,
            "loss": spec.loss, "reduction": spec.reduction, "budget_weight": spec.budget_weight,
            "trainability": "selected_cutoff_scalars_only", "requested_updates": spec.updates,
            "seed": spec.seed, "rng_policy": "no_random_operations",
            "costs": costs, "cost_bound": "tensor_lower_bounds; backend_workspace_and_other_activations_unknown",
            "remapping": "deferred; ordinary_same_dtype_factors_only", "bias_policy": "unchanged_outside_projection",
        }
        if spec.remapping != "none":
            self.evidence.update(remapping=spec.remapping,
                budget_variant="fedcore_actual_remapped_weight_tensor_bytes_v1",
                budget_denominator="selected_original_weight_tensor_bytes; bias_excluded",
                remapping_source=DOBI_REPOSITORY.replace("/tree/","/blob/")+"/modules/remapping.py",
                remapping_source_blob_sha="796b6dfb2513b2ae58064d13093909856244f73a",
                paper_bnb_remapping_reproduced=False,
                remap_precision={"left_codes":"int8","right_codes":"int8",
                    "tail":spec.remap_tail_dtype,"scales":spec.remap_scale_dtype,
                    "compute":spec.remap_compute_dtype,"bias":"source"})

    def __enter__(self):
        if self.active or self.finalized:
            raise NumericalDomainViolation("Dobi-SVD objective cannot be nested or reused after finalize")
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False
        self.evidence.update(objective_calls=self.calls, completed_optimizer_steps=self.steps,
                             hooks_removed=True, hook_count=0)
        return False

    def __call__(self, model, inputs, labels):
        if not self.active or model is not self.model:
            raise NumericalDomainViolation("Dobi-SVD objective requires its owned model and active context")
        previous = {path: layer.calls for path, layer in self.layers.items()}
        task = observed_loss(model(inputs), labels, self.spec.loss, self.spec.reduction)
        if any(layer.calls != previous[path] + 1 for path, layer in self.layers.items()):
            raise NumericalDomainViolation("Every Dobi-SVD target must execute exactly once per objective call")
        numerator = sum(layer.k * (layer.in_features + layer.out_features) for layer in self.layers.values())
        denominator = sum(layer.in_features * layer.out_features for layer in self.layers.values())
        if self.spec.remapping != "none":
            from .dobi_remapping import remapped_tensor_bytes
            numerator = sum(layer.k * remapped_tensor_bytes(layer.in_features,layer.out_features,1,
                tail_dtype=self.spec.remap_tail_dtype,scale_dtype=self.spec.remap_scale_dtype,bias=False)
                for layer in self.layers.values())
            denominator = sum(layer.base.weight.numel()*layer.base.weight.element_size()
                              for layer in self.layers.values())
        ratio = numerator / denominator
        value = task + self.spec.budget_weight * (ratio - self.spec.target_fraction).abs()
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise NumericalDomainViolation("Dobi-SVD objective must be a finite scalar")
        self.calls += 1
        self.evidence["last_loss_components"] = {"task": float(task.detach()), "soft_fraction": float(ratio.detach())}
        return value

    def after_step(self, model, optimizer):
        if model is not self.model or not self.active:
            raise NumericalDomainViolation("Dobi-SVD step belongs to its active owned model")
        with torch.no_grad():
            for layer in self.layers.values():
                if not bool(torch.isfinite(layer.k)):
                    raise NumericalDomainViolation("Nonfinite learned Dobi-SVD cutoff")
                layer.k.clamp_(0, layer.maximum_rank)
        self.steps += 1

    def finalize(self, model, calibration, batch_size, *, tensor_byte_budget=None):
        """Fix learned integer ranks, collect bases sequentially, replace atomically.

        Calibration is one CPU input tensor split into deterministic contiguous
        batches. All targets observe the same dense frozen-base model during
        collection. Deployment uses fixed matrices and performs no input SVD.
        """
        if self.active or self.finalized or model is not self.model:
            raise NumericalDomainViolation("Finalize Dobi-SVD once, outside the objective context")
        if (type(batch_size) is not int or batch_size < 1 or not isinstance(calibration, torch.Tensor)
                or calibration.device.type != "cpu" or calibration.ndim < 2 or not len(calibration)
                or not bool(torch.isfinite(calibration).all())):
            raise NumericalDomainViolation("Dobi-SVD export requires finite CPU calibration and positive batch_size")
        batches = math.ceil(len(calibration) / batch_size)
        if batches > self.spec.max_calibration_batches:
            raise NumericalDomainViolation("ResourceLimitExceeded: Dobi-SVD calibration batches")
        if self.spec.remapping != "none":
            from .dobi_remapping_export import finalize_remapped
            return finalize_remapped(self,model,calibration,batch_size,tensor_byte_budget=tensor_byte_budget)
        rounders = {"floor": math.floor, "ceil": math.ceil, "nearest": lambda value: math.floor(value + 0.5)}
        ranks = {path: max(0, min(layer.maximum_rank, rounders[self.spec.export_rounding](float(layer.k.detach()))))
                 for path, layer in self.layers.items()}
        states = {path: empty_direction_ipca(layer.out_features, ranks[path]) for path, layer in self.layers.items()}
        handles, captured = [], set()
        observed_batches = []
        def collect(path):
            def hook(layer, inputs):
                if path in captured:
                    raise NumericalDomainViolation("Dobi-SVD export target executed more than once")
                captured.add(path)
                if ranks[path] in (0, layer.maximum_rank):
                    return
                a = layer.activation(inputs[0]).detach().reshape(-1, layer.out_features).double()
                _matrix(a, "export activation")
                if len(a) > self.spec.max_activation_rows:
                    raise NumericalDomainViolation("ResourceLimitExceeded: Dobi-SVD export activation rows")
                _, _, vh = torch.linalg.svd(a, full_matrices=False)
                directions = _canonical_rows(vh[:ranks[path]])
                states[path] = update_direction_ipca(states[path], directions)
            return hook
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            for path, layer in self.layers.items():
                layer.export_mode = True
                handles.append(layer.register_forward_pre_hook(collect(path)))
            with torch.no_grad():
                for start in range(0, len(calibration), batch_size):
                    captured.clear()
                    block = calibration[start:start + batch_size]
                    model(block)
                    if captured != set(self.layers):
                        raise NumericalDomainViolation("Dobi-SVD export did not observe every target")
                    observed_batches.append(_digest(block))
        finally:
            for handle in handles:
                handle.remove()
            for layer in self.layers.values():
                layer.export_mode = False
            for module, training in modes.items():
                module.training = training
        replacements, records = {}, {}
        for path, layer in self.layers.items():
            rank, state = ranks[path], states[path]
            if rank == 0:
                from .learned_layers import ZeroLinear
                replacement = ZeroLinear.from_linear(layer.base)
                basis_hash = None
            elif rank == layer.maximum_rank:
                # Preserve all weights even for rank-deficient calibration.
                replacement = layer.base
                basis_hash = None
            else:
                if len(state.components) < rank:
                    raise NumericalDomainViolation("Insufficient calibration directions for fixed Dobi-SVD export rank")
                basis = state.components.T
                pair = fixed_output_factors(layer.base.weight.detach(), basis)
                # Existing constructor initializes Linear before copying. Its
                # local RNG isolation keeps preparation globally nonmutating.
                with torch.random.fork_rng(devices=[]):
                    replacement = FactorizedLinear.from_factors(pair, layer.base.bias)
                basis_hash = _digest(basis)
            replacement.requires_grad_(False)
            replacements[path] = replacement
            records[path] = {
                "continuous_k": float(layer.k.detach()), "rank": rank,
                "maximum_rank": layer.maximum_rank, "rounding": self.spec.export_rounding,
                "representation": type(replacement).__name__, "basis_sha256": basis_hash,
                "ipca_rows_seen": state.rows_seen, "ipca_direction_batch_hashes": list(state.batch_hashes),
                "parameter_elements": sum(p.numel() for p in replacement.parameters()),
                "export_policy": "exact_dense_full_rank" if rank == layer.maximum_rank else ("bias_only_zero_rank" if rank == 0 else "hard_fixed_output_projector"),
            }
        self.model = replace_modules_atomically(model, replacements)
        self.finalized = True
        record = {"stage": "fixed_basis_export", "ipca_variant": "fedcore_uncentered_direction_ipca_v1",
            "ipca_state_order": "contiguous_calibration_batches; truncate_after_each_batch",
            "ipca_is_global_activation_mse_optimum": False,
            "source_centered_ipca_reproduced": False, "order_dependent": True,
            "basis_nonunique_at_rank_boundary_ties": True,
            "calibration_batches": batches, "calibration_batch_sha256": observed_batches,
            "deployment_activation_svd": False, "layers": records, "hooks_removed": True}
        self.evidence["export"] = record
        return record


def prepare_dobi(model, paths, spec):
    """Install transient layers atomically on the caller-owned private model."""
    selected = _selected(model, paths, spec)
    costs = estimate_dobi_bytes(model, paths, spec)
    replacements = {path: _DobiLinear(layer, spec) for path, layer in selected}
    result = replace_modules_atomically(model, replacements)
    result.requires_grad_(False)
    for layer in replacements.values():
        layer.k.requires_grad_(True)
    return DobiObjective(result, replacements, spec, costs)
