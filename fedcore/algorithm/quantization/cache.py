"""Bounded owned tensor cache with explicit dependency invalidation."""
from dataclasses import dataclass
import torch
from .graph_specs import fingerprint


@dataclass(frozen=True)
class CacheIdentity:
    kind: str
    source: str
    predecessor: str
    calibration: str
    coordinates: str
    operation: str
    policy: str
    execution: str = 'cpu_reference'
    recovery: str = 'none'
    evaluation: str = 'none'

    @property
    def digest(self):return fingerprint(self)


class DependencyCache:
    def __init__(self,max_bytes=64*1024**2):
        if type(max_bytes) is not int or max_bytes<0:raise ValueError('Nonnegative cache byte limit required')
        self.max_bytes,self.bytes,self.entries,self.events=max_bytes,0,{},[]
        self.scope_versions={}

    def observe_scope(self,scope,predecessor):
        previous=self.scope_versions.get(scope)
        if previous is not None and previous!=predecessor:self.invalidate(previous)
        self.scope_versions[scope]=predecessor

    def put(self,key,tensors,*,dependencies=()):
        if type(key) is not CacheIdentity or key.digest in dependencies:raise ValueError('Typed acyclic cache identity required')
        if type(tensors) is not tuple or any(type(t) is not torch.Tensor or t.device.type!='cpu' or t.is_quantized for t in tensors):
            raise ValueError('Owned ordinary CPU tensors required')
        size=sum(t.numel()*t.element_size() for t in tensors)
        old=self.entries.get(key.digest)
        projected=self.bytes-(old[2] if old else 0)+size
        if projected>self.max_bytes:
            self.events.append({'event':'capacity_refused','key':key.digest,'bytes':size});return False
        # Dependencies can be external source versions. Refuse cycles through
        # existing entries, including replacement of an ancestor.
        def ancestors(node,seen):
            if node in seen:return
            seen.add(node)
            for parent in self.entries.get(node,(None,(),0,()))[3]:ancestors(parent,seen)
        seen=set()
        for dep in dependencies:ancestors(dep,seen)
        if key.digest in seen:raise ValueError('Cyclic cache dependency')
        self.entries[key.digest]=(key,tuple(t.detach().clone().contiguous() for t in tensors),size,tuple(dependencies))
        self.bytes=projected;self.events.append({'event':'put','key':key.digest,'bytes':size});return True

    def get(self,key):
        entry=self.entries.get(key.digest)
        self.events.append({'event':'hit' if entry else 'miss','key':key.digest})
        return None if entry is None else tuple(t.clone() for t in entry[1])

    def invalidate(self,dependency):
        removed=set();changed=True
        while changed:
            changed=False
            for digest,entry in tuple(self.entries.items()):
                if digest==dependency or dependency in entry[3] or removed.intersection(entry[3]):
                    removed.add(digest);self.bytes-=entry[2];del self.entries[digest];changed=True
                    self.events.append({'event':'invalidated','key':digest,'freed_bytes':entry[2]})
        return frozenset(removed)

    def clear(self):
        freed=self.bytes;self.entries.clear();self.bytes=0
        self.scope_versions.clear()
        self.events.append({'event':'clear','freed_bytes':freed})
