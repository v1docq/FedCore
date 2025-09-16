import sys

from tdecomp.matrix.decomposer import *
from torch.ao.quantization.utils import _normalize_kwargs # noqa

__functionals = {
    'rsvd': RandomizedSVD,
    'r2svd': TwoSidedRandomSVD,
    'cur': CURDecomposition
}
__all__ = list(__functionals)

__module = sys.modules[__name__]

__namespace = globals()

def __base_dec_gen(method_name, ):
    method = __functionals[method_name]
    exec(
f'''
def {method_name}(matrix, n_eigenvecs=None,  **kwargs):
    """
    {method.__doc__}
    """
    decomposer = {method.__name__}(**_normalize_kwargs({method.__name__}.__init__, kwargs))
    decomposition = decomposer.decompose(matrix, rank=n_eigenvecs)
    return decomposition
''', __namespace)
    return __namespace[method_name]

for __name in __functionals:
    setattr(__module, __name, __base_dec_gen(__name))
