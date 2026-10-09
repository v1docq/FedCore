"""Resolve trusted, source-defined factories only when an operation is used."""
from collections.abc import MutableMapping
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class LazyFactory:
    module: str
    name: str

    def resolve(self):
        return getattr(import_module(self.module), self.name)

    def __call__(self, *args, **kwargs):
        return self.resolve()(*args, **kwargs)


class LazyRegistry(MutableMapping):
    """Mutable registry whose enumeration does not import optional backbones."""

    def __init__(self, entries):
        self._entries = dict(entries)

    def __getitem__(self, key):
        value = self._entries[key]
        if isinstance(value, LazyFactory):
            value = value.resolve()
            self._entries[key] = value
        return value

    def __setitem__(self, key, value):
        if not callable(value):
            raise TypeError("Model registry entries must be callable")
        self._entries[key] = value

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries

    def __eq__(self, other):
        if isinstance(other, LazyRegistry):
            return self._entries == other._entries
        return NotImplemented

    @classmethod
    def combine(cls, *registries):
        entries = {}
        for registry in registries:
            entries.update(registry._entries)
        return cls(entries)
