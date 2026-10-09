"""Stateful index aggregation without mutating parameter values or gradient mode."""
from copy import deepcopy
import torch
from torch import nn


def _rebuild_resolver(data,requires_grad,state):
    result=IndexResolvingParameter(data,requires_grad=requires_grad,group_ids=state['group_ids'],
                                   aggregation_mode=state['aggregation_mode'],module_name=state['module_name'])
    result.load_resolver_state(state)
    return result


class IndexResolvingParameter(nn.Parameter):
    def __new__(cls,data,group_ids=None,aggregation_mode='union',module_name=None,requires_grad=None):
        flag=data.requires_grad if requires_grad is None else requires_grad
        return super().__new__(cls,data.detach().clone(),flag)

    def __init__(self,data,group_ids=None,aggregation_mode='union',module_name=None,requires_grad=None):
        if data.ndim==0:raise ValueError('An indexed parameter needs at least one dimension')
        ids=list(range(data.shape[0])) if group_ids is None else list(group_ids)
        if len(ids)!=data.shape[0] or len(set(ids))!=len(ids):raise ValueError('Group IDs must be unique and match the first dimension')
        if aggregation_mode not in ('union','intersect'):raise ValueError('Unknown aggregation mode')
        self.num_original_groups=len(ids);self.group_ids=ids;self.aggregation_mode=aggregation_mode
        self.module_name=module_name;self._initialized=False
        self._update_mapping([])

    def _update_mapping(self,indices):
        self.original_to_current={idx:None for idx in self.group_ids}
        for current,original in enumerate(indices):self.original_to_current[original]=current
        self.current_to_original={v:k for k,v in self.original_to_current.items() if v is not None}

    def resolve_indices(self,important_groups):
        new=set(int(i) for i in important_groups)
        if not new.issubset(self.group_ids):raise ValueError('Unknown original group index')
        active=set(self.current_to_original.values())
        result=new if not self._initialized else (active|new if self.aggregation_mode=='union' else active&new)
        self._update_mapping(sorted(result));self._initialized=True

    def resolver_state(self):
        return {'version':1,'group_ids':list(self.group_ids),'aggregation_mode':self.aggregation_mode,
                'module_name':self.module_name,'initialized':self._initialized,
                'active_indices':[self.current_to_original[i] for i in sorted(self.current_to_original)]}

    def load_resolver_state(self,state):
        if state['version']!=1 or state['group_ids']!=self.group_ids or state['aggregation_mode']!=self.aggregation_mode:
            raise ValueError('Incompatible resolver state')
        active=state['active_indices']
        if len(set(active))!=len(active) or not set(active).issubset(self.group_ids):raise ValueError('Invalid active indices')
        if not state['initialized'] and active:raise ValueError('Uninitialized state cannot contain active indices')
        self.module_name=state['module_name'];self._initialized=bool(state['initialized']);self._update_mapping(active)

    def __deepcopy__(self,memo):
        if id(self) in memo:return memo[id(self)]
        result=_rebuild_resolver(self.detach().clone(),self.requires_grad,self.resolver_state())
        if self.grad is not None:result.grad=self.grad.detach().clone()
        memo[id(self)]=result;return result

    def __reduce_ex__(self,protocol):
        return _rebuild_resolver,(self.detach().clone(),self.requires_grad,self.resolver_state())


def wrap_parameters_with_resolver(module,param_filter=None,aggregation_mode='union',inplace=True):
    result=module if inplace else deepcopy(module)
    aliases={}
    for name,submodule in result.named_modules():
        for param_name,param in list(submodule.named_parameters(recurse=False)):
            if isinstance(param,IndexResolvingParameter):continue
            if param_filter is not None and not param_filter(submodule,param_name,param):continue
            if param.dim()==0 or param.shape[0]==0:continue
            if id(param) not in aliases:
                aliases[id(param)]=IndexResolvingParameter(param,aggregation_mode=aggregation_mode,
                    module_name=f'{name}.{param_name}'.lstrip('.'),requires_grad=param.requires_grad)
                if param.grad is not None:aliases[id(param)].grad=param.grad.detach().clone()
            setattr(submodule,param_name,aliases[id(param)])
    return result
