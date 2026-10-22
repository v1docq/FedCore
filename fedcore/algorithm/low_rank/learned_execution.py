"""Bounded lifecycle for three explicit learned-structure algorithms.

This module owns copying, provenance and publication. The existing PETRA
training boundary owns batches and Adam; each method owns its mathematical
objective and structural state transitions. No universal learned-rank solver.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
import time

import torch
from torch import nn

from .allocation import exact_unique_cost
from .execution import _fingerprint, _storage_records, plan_weighted, WeightedProfileError
from .method_execution import (MethodTransformResult, _check_data, _rng_isolated,
                               _tensor_state_bytes, _explicit_encoder_graph)
from .projection_execution import _selected
from .topology import module_paths
from .learned_layers import UnsupportedLearnedAccessError


class LearnedProfileError(WeightedProfileError):
    pass


def _implementation(spec):
    from .ars_specs import ARS
    from .dobi_specs import DobiSVD
    from .rankdyna_specs import RankDyna
    if type(spec) is ARS:
        from .ars_profiles import prepare_ars, estimate_ars_bytes
        return 'ars', prepare_ars, estimate_ars_bytes
    if type(spec) is DobiSVD:
        from .dobi_profiles import prepare_dobi, estimate_dobi_bytes
        return 'dobi_svd', prepare_dobi, estimate_dobi_bytes
    if type(spec) is RankDyna:
        from .rankdyna_profiles import prepare_rankdyna, estimate_rankdyna_bytes
        return 'rankdyna', prepare_rankdyna, estimate_rankdyna_bytes
    raise LearnedProfileError('Unknown learned-structure method')


def plan_learned(model, spec, *, rank=None, rank_ratio=None, parameter_fraction=None,
                 ranks=None, target_paths=None, max_peak_bytes=12*1024**3):
    from .method_specs import method_name
    name = method_name(spec)
    if any(value is not None for value in (rank,rank_ratio,parameter_fraction,ranks)):
        raise LearnedProfileError('Learned structure uses its explicit method budget, not a fixed rank policy')
    paths = _selected(model,target_paths)
    for path in paths:
        layer = model.get_submodule(path)
        if (layer._modules or layer._buffers or set(layer._parameters) != {'weight','bias'}
                or 'forward' in layer.__dict__ or layer._forward_hooks or layer._forward_pre_hooks
                or layer._backward_hooks or getattr(layer,'_backward_pre_hooks',{})):
            raise LearnedProfileError('P4 targets require unmodified standard Linear without hooks or extra state')
    plan = plan_weighted(model,rank_ratio=1.,target_paths=paths,max_peak_bytes=max_peak_bytes)
    return replace(plan,request=replace(plan.request,objective=name+'_structure_training',
                                       representation='learned_then_compact'))


class _TrainingBoundary:
    """Count real objective calls/transitions rather than trusting a report."""
    def __init__(self, objective, updates):
        self.objective,self.updates = objective,updates
        self.calls,self.steps,self.active = 0,0,False
        self.pending = None

    @property
    def evidence(self):
        return dict(self.objective.evidence, observed_objective_calls=self.calls,
                    observed_optimizer_steps=self.steps)

    def __enter__(self):
        if self.active:
            raise LearnedProfileError('Training objective context cannot be nested')
        self.objective.__enter__()
        self.active = True
        return self

    def __exit__(self,*exc):
        self.active = False
        return self.objective.__exit__(*exc)

    def __call__(self,model,inputs,labels):
        if not self.active or self.calls >= self.updates or self.calls != self.steps or self.pending is not None:
            raise LearnedProfileError('Training objective exceeds its explicit active update budget')
        loss = self.objective(model,inputs,labels)
        self.calls += 1
        return loss

    def before_step(self,model,optimizer):
        if not self.active or self.calls != self.steps+1 or self.pending is not None:
            raise LearnedProfileError('Backward/optimizer transition order is invalid')
        if type(optimizer) not in (torch.optim.Adam,torch.optim.AdamW):
            raise LearnedProfileError('Structure training requires its live Adam optimizer')
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        if ({id(p) for p in parameters} != {id(p) for p in model.parameters() if p.requires_grad}
                or len(parameters) != len({id(p) for p in parameters})
                or not any(p.grad is not None for p in parameters)
                or any(not bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)):
            raise LearnedProfileError('Backward must produce finite gradients for the live optimizer parameters')
        self.pending = (optimizer,tuple((p,self._step(optimizer,p),p.grad is not None) for p in parameters))
        callback = getattr(self.objective,'before_step',None)
        if callback is not None:
            callback(model,optimizer)

    def after_step(self,model,optimizer):
        if (not self.active or self.calls != self.steps+1 or self.pending is None
                or self.pending[0] is not optimizer):
            raise LearnedProfileError('Optimizer transition requires its matching backward and live optimizer')
        if any(self._step(optimizer,p) != old+int(has_grad) for p,old,has_grad in self.pending[1]):
            raise LearnedProfileError('Adam did not execute exactly one optimizer step')
        callback = getattr(self.objective,'after_step',None)
        if callback is not None:
            callback(model,optimizer)
        self.steps += 1
        self.pending = None

    @staticmethod
    def _step(optimizer,parameter):
        value = optimizer.state.get(parameter,{}).get('step',0)
        if isinstance(value,torch.Tensor):
            if value.numel() != 1:
                raise LearnedProfileError('Adam step counter must be scalar')
            value = value.item()
        if type(value) not in (int,float) or value < 0 or int(value) != value:
            raise LearnedProfileError('Adam step counter must be a nonnegative integer')
        return int(value)


@_rng_isolated
def transform_learned(model,calibration,spec,*,rank=None,rank_ratio=None,
        parameter_fraction=None,ranks=None,target_paths=None,recovery_executor=None,
        batch_size=16,max_workspace_bytes=256*1024**2,max_peak_bytes=12*1024**3,
        graph_version='p4-profile-v1',parameter_budget=None,tensor_byte_budget=None,
        data_role='calibration'):
    from .method_specs import method_payload
    started = time.perf_counter()
    if data_role != 'calibration' or not isinstance(graph_version,str) or not graph_version:
        raise LearnedProfileError('Declare calibration role and graph version')
    if not callable(recovery_executor):
        raise LearnedProfileError('Learned structure requires an explicit fixed-update training executor')
    for value in (max_workspace_bytes,max_peak_bytes):
        if type(value) is not int or value <= 0:
            raise LearnedProfileError('Resource limits must be positive integers')
    for value in (parameter_budget,tensor_byte_budget):
        if value is not None and (type(value) is not int or value < 0):
            raise LearnedProfileError('Export budgets must be nonnegative integers')
    plan = plan_learned(model,spec,rank=rank,rank_ratio=rank_ratio,
        parameter_fraction=parameter_fraction,ranks=ranks,target_paths=target_paths,
        max_peak_bytes=max_peak_bytes)
    _check_data(model,calibration,batch_size,max_workspace_bytes,max_peak_bytes)
    name,prepare,estimate = _implementation(spec)
    paths = tuple(step.operator.path for step in plan.steps)
    estimate = estimate(model,paths,spec)
    if not isinstance(estimate,dict) or any(type(value) is not int or value < 0 for value in estimate.values()):
        raise LearnedProfileError('Method tensor reservations require named nonnegative byte counts')
    import psutil
    source_bytes = _tensor_state_bytes(model)
    solve_workspace = max(8*(8*n*n+8*m*n+4*m*m) for m,n in (step.operator.shape for step in plan.steps))
    if solve_workspace > max_workspace_bytes:
        raise LearnedProfileError('ResourceLimitExceeded: learned initialization solve workspace')
    reserve = (psutil.Process(os.getpid()).memory_info().rss+64*1024**2+
               5*source_bytes+sum(estimate.values())+calibration.numel()*calibration.element_size())
    if reserve+max_workspace_bytes > max_peak_bytes:
        raise LearnedProfileError('ResourceLimitExceeded: learned preparation tensor reservation')
    checkpoint_id,data_id = _fingerprint(model,calibration,graph_version)
    modes = {path:layer.training for path,layer in module_paths(model).items()}
    work = _explicit_encoder_graph(deepcopy(model)).eval()
    objective = prepare(work,paths,spec)
    work = getattr(objective,'model',work)
    trainable = tuple(p for p in work.parameters() if p.requires_grad)
    trainable_bytes = sum(p.numel()*p.element_size() for p in trainable)
    frozen = {key:p.detach().clone() for key,p in work.named_parameters() if not p.requires_grad}
    frozen_bytes = sum(value.numel()*value.element_size() for value in frozen.values())
    named = _tensor_state_bytes(work)+3*trainable_bytes+4*len(trainable)+frozen_bytes
    training_reserve = psutil.Process(os.getpid()).memory_info().rss+64*1024**2+named
    if training_reserve+max_workspace_bytes > max_peak_bytes:
        raise LearnedProfileError('ResourceLimitExceeded: learned gradient/Adam/state reservation')
    boundary = _TrainingBoundary(objective,spec.updates)
    try:
        training = recovery_executor(work,name,spec.updates,boundary)
    except UnsupportedLearnedAccessError as error:
        raise LearnedProfileError(str(error)) from error
    if (not isinstance(training,dict) or training.get('updates',training.get('training_steps')) != spec.updates
            or boundary.calls != spec.updates or boundary.steps != spec.updates or boundary.active):
        raise LearnedProfileError('Executor did not complete the exact context-managed structure update budget')
    json.dumps(training,allow_nan=False)
    if training.get('training_role') != 'train':
        raise LearnedProfileError('Structure optimizer steps require an explicit train data role')
    current = dict(work.named_parameters())
    if any(key not in current or not torch.equal(value,current[key].detach()) for key,value in frozen.items()):
        raise LearnedProfileError('Structure training modified frozen parameters')
    try:
        from .dobi_specs import DobiSVD
        if type(spec) is DobiSVD and spec.remapping != 'none':
            final = objective.finalize(work,calibration,batch_size,tensor_byte_budget=tensor_byte_budget)
        else:
            final = objective.finalize(work,calibration,batch_size)
    except UnsupportedLearnedAccessError as error:
        raise LearnedProfileError(str(error)) from error
    work = getattr(objective,'model',work)
    if not isinstance(final,dict):
        raise LearnedProfileError('Method finalization requires explicit export evidence')
    if any(not bool(torch.isfinite(value).all()) for value in (*work.parameters(),*work.buffers()) if value.is_floating_point()):
        raise LearnedProfileError('Nonfinite learned artifact')
    after = _storage_records(work)
    cost,tensor_bytes = exact_unique_cost(after),_tensor_state_bytes(work)
    if parameter_budget is not None and cost > parameter_budget:
        raise LearnedProfileError('BudgetInfeasible: actual compact parameter count')
    if tensor_byte_budget is not None and tensor_bytes > tensor_byte_budget:
        raise LearnedProfileError('BudgetInfeasible: actual compact tensor bytes')
    for path,layer in module_paths(work).items():
        layer.training = modes.get(path,False)
    evidence = {'schema_version':1,'profile':'p4_learned_cpu','method':name,
        'method_version':1,'method_spec':method_payload(spec),'checkpoint_id':checkpoint_id,
        'data_id':data_id,'calibration_id':data_id,'data_role':'calibration',
        'graph_version':graph_version,'training_role':'train','training':training,
        'structure_training':boundary.evidence,'export':final,
        'plan_semantics':'full-width initialization; final structure is recorded in export',
        'parameters_before':exact_unique_cost(_storage_records(model)),'parameters_after':cost,
        'parameter_bytes_after':exact_unique_cost(after,'bytes'),
        'tensor_bytes_before':source_bytes,'tensor_bytes_after':tensor_bytes,
        'buffer_bytes_after':tensor_bytes-exact_unique_cost(after,'bytes'),
        'representation':'compact','optimizer_rebuild_required':True,'training_cost_separate':True,
        'resources':{'method_tensor_reservations':estimate,'trainable_tensor_bytes_initial':trainable_bytes,
            'frozen_snapshot_bytes':frozen_bytes,'gradient_reserve_bytes':trainable_bytes,
            'adam_moments_reserve_bytes':2*trainable_bytes,'adam_step_scalar_reserve_bytes':4*len(trainable),
            'estimated_peak_bytes':max(reserve,training_reserve)+max_workspace_bytes,
            'limit_bytes':max_peak_bytes,'workspace_bytes':max_workspace_bytes,
            'scope':'named tensors and reserved workspace; arbitrary forward and allocator RSS are not hard-bounded'},
        'timings':{'total_seconds':time.perf_counter()-started}}
    json.dumps(evidence,allow_nan=False)
    work._fedcore_method_evidence = evidence
    work._fedcore_transform_plan = plan
    work._fedcore_requires_optimizer_rebuild = True
    return MethodTransformResult(work,plan,evidence)
