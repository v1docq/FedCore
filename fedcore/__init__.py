"""Neural network compression and automated model optimization.

Subpackages load on demand; importing FedCore does not initialize runtimes
or alter third-party module registrations.
"""
from importlib import import_module

__version__ = "0.0.5.4"
__all__ = ['api', 'algorithm', 'architecture', 'data', 'inference', 'interfaces',
           'losses', 'metrics', 'models', 'repository', 'tools', 'external']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = import_module('external' if name == 'external' else f'{__name__}.{name}')
    globals()[name] = module
    return module
