"""Bounded calibration export and exact byte rebudget for portable Dobi remap."""
import math

import torch
from torch.nn import functional as F

from .approximation import NumericalDomainViolation
from .dobi_remapping import RemappedLinear, remapped_tensor_bytes
from .learned_layers import ZeroLinear
from .topology import replace_modules_atomically


def _bytes(module):
    from .method_execution import _tensor_state_bytes
    return _tensor_state_bytes(module)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise NumericalDomainViolation("Nonfinite Dobi remapping "+name)
    return value


def _observe(objective, calibration, batch_size, callback):
    """Replay frozen dense targets, with hooks removed even on failed export."""
    captured, handles, hashes = set(), [], []
    modes = {module: module.training for module in objective.model.modules()}
    from .dobi_profiles import _digest
    def hook(path):
        def run(layer, inputs):
            if path in captured:
                raise NumericalDomainViolation("Dobi-SVD export target executed more than once")
            captured.add(path)
            x = inputs[0]
            if (not isinstance(x,torch.Tensor) or x.ndim<2 or x.shape[-1]!=layer.in_features
                    or not x.numel() or x.numel()//layer.in_features>objective.spec.max_activation_rows):
                raise NumericalDomainViolation("ResourceLimitExceeded: Dobi-SVD export activation rows")
            callback(path,layer,x)
        return run
    try:
        objective.model.eval()
        for path, layer in objective.layers.items():
            layer.export_mode = True
            handles.append(layer.register_forward_pre_hook(hook(path)))
        with torch.no_grad():
            for start in range(0,len(calibration),batch_size):
                captured.clear()
                block = calibration[start:start+batch_size]
                _finite(objective.model(block),"dense calibration output")
                if captured != set(objective.layers):
                    raise NumericalDomainViolation("Dobi-SVD export did not observe every target")
                hashes.append(_digest(block))
    finally:
        for handle in handles:
            handle.remove()
        for layer in objective.layers.values():
            layer.export_mode = False
        for module, training in modes.items():
            module.training = training
    return hashes


def finalize_remapped(objective, model, calibration, batch_size, *, tensor_byte_budget=None):
    from .dobi_profiles import (empty_direction_ipca, update_direction_ipca, _canonical_rows,
                                fixed_output_factors, _digest)
    if tensor_byte_budget is not None and (type(tensor_byte_budget) is not int or tensor_byte_budget<0):
        raise NumericalDomainViolation("Tensor byte budget must be a nonnegative integer")
    spec = objective.spec
    rounders = {"floor":math.floor,"ceil":math.ceil,"nearest":lambda value:math.floor(value+0.5)}
    if any(not bool(torch.isfinite(layer.k)) for layer in objective.layers.values()):
        raise NumericalDomainViolation("Nonfinite learned Dobi-SVD cutoff")
    ranks = {path:max(0,min(layer.maximum_rank,rounders[spec.export_rounding](float(layer.k.detach()))))
             for path,layer in objective.layers.items()}
    initial_ranks = dict(ranks)
    unchanged_bytes = _bytes(model)-sum(_bytes(layer) for layer in objective.layers.values())
    def layer_cost(path,rank):
        layer = objective.layers[path]
        return remapped_tensor_bytes(layer.in_features,layer.out_features,rank,
            tail_dtype=spec.remap_tail_dtype,scale_dtype=spec.remap_scale_dtype,
            bias_dtype=str(layer.base.weight.dtype)[6:],bias=layer.base.bias is not None)
    def total_cost():
        return unchanged_bytes+sum(layer_cost(path,rank) for path,rank in ranks.items())
    # Every nonzero rank costs c*k plus unchanged bias. Select the largest byte
    # saving, stable path tie-break. This is deterministic feasibility, not an
    # optimal allocation for task loss. No SVD or calibration run before this.
    reductions = []
    while tensor_byte_budget is not None and total_cost()>tensor_byte_budget:
        candidates = [(layer_cost(path,rank)-layer_cost(path,rank-1),path)
                      for path,rank in ranks.items() if rank]
        if not candidates:
            raise NumericalDomainViolation("BudgetInfeasible: unchanged tensors and biases exceed tensor byte budget")
        saving,path = sorted(candidates,key=lambda item:(-item[0],item[1]))[0]
        ranks[path] -= 1
        reductions.append({"path":path,"rank":ranks[path],"saved_tensor_bytes":saving})
    states = {path:empty_direction_ipca(layer.out_features,ranks[path])
              for path,layer in objective.layers.items()}
    def collect(path,layer,x):
        rank = ranks[path]
        if rank in (0,layer.maximum_rank):
            return
        a = _finite(layer.activation(x).detach().reshape(-1,layer.out_features).double(),"activation")
        _,_,vh = torch.linalg.svd(a,full_matrices=False)
        states[path] = update_direction_ipca(states[path],_canonical_rows(vh[:rank]))
    hashes = _observe(objective,calibration,batch_size,collect)
    replacements,records,hard_weights = {},{},{}
    for path,layer in objective.layers.items():
        rank,state = ranks[path],states[path]
        weight = layer.base.weight.detach()
        basis_hash = None
        if rank == 0:
            replacement = ZeroLinear.from_linear(layer.base)
            hard_weights[path] = None
        else:
            if rank == layer.maximum_rank:
                hard = weight.double()
            else:
                if len(state.components)<rank:
                    raise NumericalDomainViolation("Insufficient calibration directions for fixed Dobi-SVD export rank")
                basis = state.components.T
                pair = fixed_output_factors(weight.double(),basis)
                hard = _finite(pair.left @ pair.right,"projected weight")
                basis_hash = _digest(basis)
            # Preserve double reconstruction through the re-SVD. Bias and
            # external operator dtype are restored explicitly after encoding.
            replacement = RemappedLinear.from_weight(hard,rank,
                None if layer.base.bias is None else layer.base.bias.detach().double(),
                tail_dtype=spec.remap_tail_dtype,scale_dtype=spec.remap_scale_dtype,
                compute_dtype=spec.remap_compute_dtype)
            replacement.bias_dtype = str(weight.dtype)[6:]
            if replacement.bias is not None:
                replacement.bias = torch.nn.Parameter(layer.base.bias.detach().clone(),requires_grad=False)
            hard_weights[path] = hard
        replacement.requires_grad_(False)
        replacements[path] = replacement
        actual_bytes = _bytes(replacement)
        if actual_bytes != layer_cost(path,rank):
            raise NumericalDomainViolation("Actual remapping tensors differ from declared byte cost")
        records[path] = {"continuous_k":float(layer.k.detach()),"rank":rank,
            "rank_before_byte_rebudget":initial_ranks[path],"maximum_rank":layer.maximum_rank,
            "rounding":spec.export_rounding,"representation":type(replacement).__name__,
            "basis_sha256":basis_hash,"ipca_rows_seen":state.rows_seen,
            "ipca_direction_batch_hashes":list(state.batch_hashes),
            "parameter_elements":sum(p.numel() for p in replacement.parameters()),
            "tensor_bytes":actual_bytes,"buffer_bytes":sum(v.numel()*v.element_size() for v in replacement.buffers()),
            "precision":replacement.representation_config(),
            "export_policy":"bias_only_zero_rank" if rank==0 else "hard_fixed_output_projector_then_remap",
            "full_rank_lossless_before_remapping":rank==layer.maximum_rank}
    # Independent per-target output metric: exact hard operator and decoded
    # operator see precisely the same original dense calibration inputs.
    errors = {path:{"elements":0,"truncation_squared_error":0.,"remapping_squared_error":0.,
                    "total_squared_error":0.,"source_squared_norm":0.,"hard_squared_norm":0.}
              for path in objective.layers}
    def measure(path,layer,x):
        rows = x.detach().reshape(-1,layer.in_features)
        source = _finite(F.linear(rows.double(),layer.base.weight.detach().double(),None),"source output")
        hard = source.new_zeros(source.shape) if hard_weights[path] is None else F.linear(rows.double(),hard_weights[path])
        bias = layer.base.bias
        if bias is not None:
            source = source+bias.detach().double()
            hard = hard+bias.detach().double()
        decoded = _finite(replacements[path](rows).double(),"decoded output")
        error = errors[path]
        error["elements"] += source.numel()
        for name,value in (("truncation_squared_error",source-hard),("remapping_squared_error",hard-decoded),
                           ("total_squared_error",source-decoded),("source_squared_norm",source),("hard_squared_norm",hard)):
            error[name] += float(_finite(value.square(),"squared error").sum())
    _observe(objective,calibration,batch_size,measure)
    for path,error in errors.items():
        for kind in ("truncation","remapping","total"):
            error[kind+"_mse"] = error[kind+"_squared_error"]/error["elements"]
        error["relative_total_squared_error"] = (error["total_squared_error"]/error["source_squared_norm"]
                                                 if error["source_squared_norm"] else None)
        if any(not math.isfinite(value) for value in error.values() if value is not None):
            raise NumericalDomainViolation("Nonfinite Dobi remapping local calibration error")
        records[path]["calibrated_output_error"] = error
    # Keep only one pair of end-to-end batch outputs. Temporary graph swaps are
    # restored in finally; original target input distributions differ from
    # compressed ones, so this metric is separate from the local error above.
    end_to_end = {"elements":0,"squared_error":0.,"source_squared_norm":0.}
    modes = {module:module.training for module in model.modules()}
    current = model
    try:
        model.eval()
        for layer in objective.layers.values():
            layer.export_mode = True
        with torch.no_grad():
            for start in range(0,len(calibration),batch_size):
                block = calibration[start:start+batch_size]
                source = _finite(current(block),"source end-to-end output").double()
                current = replace_modules_atomically(current,replacements)
                decoded = _finite(current(block),"decoded end-to-end output").double()
                current = replace_modules_atomically(current,objective.layers)
                end_to_end["elements"] += source.numel()
                end_to_end["squared_error"] += float(_finite((source-decoded).square(),"end-to-end squared error").sum())
                end_to_end["source_squared_norm"] += float(_finite(source.square(),"end-to-end squared norm").sum())
    finally:
        current = replace_modules_atomically(current,objective.layers)
        for layer in objective.layers.values():
            layer.export_mode = False
        for module,training in modes.items():
            module.training = training
    end_to_end["mse"] = end_to_end["squared_error"]/end_to_end["elements"]
    end_to_end["relative_squared_error"] = (end_to_end["squared_error"]/end_to_end["source_squared_norm"]
                                            if end_to_end["source_squared_norm"] else None)
    if any(not math.isfinite(value) for value in end_to_end.values() if value is not None):
        raise NumericalDomainViolation("Nonfinite Dobi remapping calibration error")
    objective.model = replace_modules_atomically(model,replacements)
    objective.finalized = True
    record = {"stage":"fixed_basis_export","remapping":spec.remapping,
        "paper_bnb_remapping_reproduced":False,"ipca_variant":"fedcore_uncentered_direction_ipca_v1",
        "ipca_state_order":"contiguous_calibration_batches; truncate_after_each_batch",
        "ipca_is_global_activation_mse_optimum":False,"source_centered_ipca_reproduced":False,
        "order_dependent":True,"basis_nonunique_at_rank_boundary_ties":True,
        "calibration_batches":len(hashes),"calibration_batch_sha256":hashes,
        "deployment_activation_svd":False,"layers":records,"hooks_removed":True,
        "calibration_error_scope":"local outputs on original dense target inputs; end-to-end output MSE separately measured",
        "end_to_end_calibrated_output_error":end_to_end,
        "tensor_byte_budget":tensor_byte_budget,"tensor_bytes":_bytes(objective.model),
        "unchanged_tensor_bytes":unchanged_bytes,"byte_rebudget_policy":"greedy_largest_byte_saving_then_path; not_loss_optimal",
        "rank_reductions":reductions,"target_fraction_is_hard_guarantee":False,
        "runtime":"decode two factors per call; no persistent dense cache; temporary factors and outputs excluded from storage bytes"}
    if record["tensor_bytes"] != total_cost():
        raise NumericalDomainViolation("Actual remapped model bytes differ from declared export cost")
    objective.evidence["export"] = record
    return record
