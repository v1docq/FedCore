from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from functools import wraps, reduce, partial
from typing import *
from abc import ABC, abstractmethod

from fedcore.architecture.utils.misc import filter_kw_universal
# from fedcore.repository.constanst_repository import DIM_SUM_LIM, DIM_LIM

__all__ = [
    'SVDDecomposition',
    'RandomizedSVD',
    'CURDecomposition',
    'DECOMPOSERS',
]

DIM_SUM_LIM = 1024
DIM_LIM = 2048


def _need_t(f):
    """Performs matrix transposition for maximal projection effect"""
    @wraps(f)
    def _wrapper(self, W: torch.Tensor, *args, **kwargs):
            m, n = W.size(-2), W.size(-1)
            _is_transposed = m >= n
            weight = W.t() if _is_transposed else W
            tns = f(self, weight, *args, **kwargs)
            return (
                tns if not _is_transposed
                else tuple(t.t() for t in reversed(tns))
            )
    return _wrapper

def _ortho_gen(x: int, y: int):
    P = torch.empty(x, y)
    torch.nn.init.orthogonal_(P)
    return P

class Decomposer(ABC):
    def decompose(self, W: torch.Tensor, *args, **kwargs):
        if not self._is_big(W):
            return self._decompose(W, *args, **kwargs)
        else:
            return self._decompose_big(W, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())

    @abstractmethod
    def _decompose(self, W, *args, **kwargs):
        pass

    def _decompose_big(self, W, *args, **kwargs):
        return self._decompose(W, *args, **kwargs)
    
    def _get_stable_rank(self, W):
        n_samples = max(W.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.distortion_factors).tolist()
        return min((round(max(min_num_samples)), *W.size(), 1))
    
    def get_approximation_error(self, W, *result_matrices):
        approx = reduce(torch.matmul, result_matrices)
        return torch.linalg.norm(W - approx)


class SVDDecomposition(Decomposer):
    def _decompose(self, W: torch.Tensor) -> tuple:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            W: matrix to decompose
        Returns:
            u, s, vt: decomposition

        """
        # Return classic svd decomposition
        return torch.linalg.svd(W, full_matrices=False)
    
    def get_approximation_error(self, W, *result_matrices):
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)


class RandomizedSVD(Decomposer):
    """
    https://arxiv.org/pdf/2404.09276
    """
    _random_gens = {
        'normal': lambda x, y : torch.randn(x, y),
        'ortho' : _ortho_gen
    }

    @filter_kw_universal
    def __init__(self, *, power: int = 3,
                 distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.power = power
        self.distortion_factors = distortion_factor
        self.random_init = random_init
    
    @_need_t
    def _decompose_big(self, X):
        P = torch.randn(self._get_stable_rank(X), X.size(-2))
        G = P @ X @ (X.T @ P.T)
        Q, _ = torch.linalg.qr(
            (torch.pow(G, self.power) @ (P @ X)).T,
            mode='reduced')
        B = X @ Q
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return U, S, Vh @ Q.T
        
    @_need_t
    def _decompose(self, X):
        G = X @ X.T
        P = torch.randn(X.size(1), self._get_stable_rank(X))
        Q, _ = torch.linalg.qr(torch.pow(G, self.power) @ X @ P, mode='reduced')
        B = Q.T @ X
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return Q @ U, S, Vh
    
    def get_approximation_error(self, W, *result_matrices):
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)
    

class CURDecomposition(Decomposer):
    """
    CUR decomposition is a low-rank matrix decomposition method that is based on selecting
    a subset of columns and rows of the original matrix. The method is based on the
    Johnson-Lindenstrauss lemma and is used to approximate the original matrix with a
    low-rank matrix. The CUR decomposition is defined as follows:
    A = C @ U @ R
    where A is the original matrix, C is a subset of columns of A, U is a subset of rows of A,
    and R is a subset of rows of A. The selection of columns and rows is based on the
    probabilities p and q, which are computed based on the norms of the columns and rows of A.
    The selection of columns and rows is done in such a way that the approximation error is minimized.

    Args:
        params: the parameters of the operation
            rank: the rank of the decomposition
            tolerance: the tolerance of the decomposition
            return_samples: whether to return the samples or the decomposition matrices

    """

    @filter_kw_universal
    def __init__(self, *, rank: Optional[int] = None, distortion: Union[int, List[int]]):
        self.stable_rank = rank
        self.distortion = distortion

    def get_aproximation_error(self, original_tensor, cur_matrices: tuple):
        C, U, R = cur_matrices
        return torch.linalg.norm(original_tensor - C @ U @ R)

    def _decompose(self, tensor: torch.Tensor):
        if self.stable_rank is None:
            self.stable_rank = self._get_stable_rank(tensor)
        # create sub matrices for CUR-decompostion
        c, w, r = self.select_rows_cols(tensor)
        # evaluate pseudoinverse for W - U^-1
        u = torch.linalg.pinv(w)
        # aprox U using pseudoinverse
        return (c, u, r)

    def _importance(self, X, p):
        ax = 0
        X_scaled = (X - torch.min(X, dim=ax).values) / (torch.max(X, dim=ax).values - torch.min(X, dim=ax).values)
        torch.nan_to_num_(X_scaled, 0) 
        col_norms = torch.linalg.norm(X_scaled, ord=p, axis=0)
        row_norms = torch.linalg.norm(X_scaled, ord=p, axis=1)
        matrix_norm = torch.linalg.norm(X_scaled, 'fro')  # np.sum(np.power(matrix, 2))

        # Compute the probabilities for selecting columns and rows
        col_probs, row_probs = col_norms / matrix_norm, row_norms / matrix_norm
        return col_probs, row_probs

    def select_rows_cols(
            self, X: torch.Tensor,
            p=2) -> Tuple[torch.Tensor]:
        # Evaluate norms for columns and rows
        col_probs, row_probs = self._importance(X, p)

        rank = self.stable_rank

        column_indices = torch.sort(torch.argsort(col_probs, descending=True)[:rank]).values
        row_indices = torch.sort(torch.argsort(row_probs, descending=True)[:rank]).values

        C_matrix = X[:, column_indices] 
        R_matrix = X[row_indices, :]
        W_matrix = X[row_indices, :][:, column_indices]

        return C_matrix, W_matrix, R_matrix

DECOMPOSERS = {
    'svd': SVDDecomposition,
    'rsvd': RandomizedSVD,
    'cur': CURDecomposition,
    }
