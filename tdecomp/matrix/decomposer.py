import torch

from typing import *

from tdecomp._base import Decomposer, _need_t
from tdecomp.matrix.random_projections import RANDOM_GENS

__all__ = [
    'SVDDecomposition',
    'RandomizedSVD',
    'TwoSidedRandomSVD',
    'CURDecomposition'
]


class SVDDecomposition(Decomposer):
    def _decompose(self, W: torch.Tensor, rank) -> tuple:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            W: matrix to decompose
        Returns:
            u, s, vt: decomposition

        """
        # Return classic svd decomposition
        return torch.linalg.svd(W, full_matrices=False)


class RandomizedSVD(Decomposer):
    """
    https://arxiv.org/pdf/2404.09276
    """
    _random_gens = _random_gens = RANDOM_GENS

    def __init__(self, rank=None, power: int = 3,
                 distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        super().__init__(rank, distortion_factor, random_init)
        self.power = power

    def estimate_stable_rank(self, tensor: torch.Tensor) -> int:
        svals = torch.linalg.svdvals(tensor)
        stable_rank = (svals.sum() / svals.max())**2
        return max(1, min(tensor.size(-1), int(stable_rank * (1 / self.distortion_factor))))
    
    @_need_t
    def _decompose_big(self, X: torch.Tensor, rank):
        P = torch.randn(rank, X.size(-2), device=X.device, dtype=X.dtype)
        G = P @ X @ (X.T @ P.T)
        Q, _ = torch.linalg.qr(
            (torch.pow(G, self.power) @ (P @ X)).T,
            mode='reduced')
        B = X @ Q
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return U, S, Vh @ Q.T
        
    @_need_t
    def _decompose(self, X: torch.Tensor, rank):
        G = X @ X.T
        P = torch.randn(X.size(1), rank, device=X.device, dtype=X.dtype)
        Q, _ = torch.linalg.qr(torch.pow(G, self.power) @ X @ P, mode='reduced')
        B = Q.T @ X
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return Q @ U, S, Vh


class TwoSidedRandomSVD(RandomizedSVD):
    """
    Randomized Two-Sided SVD with explicit rank parameter support
    https://scispace.com/pdf/randomized-algorithms-for-computation-of-tucker-1stsnpusvv.pdf
    """
    def __init__(self, *, rank: int = None, distortion_factor: float = 0.6, 
                 random_init: str = 'normal', ):
        super().__init__(rank=rank, distortion_factor=distortion_factor, random_init=random_init)
        if random_init == 'lean_walsh' and rank is not None:
            if not (rank > 0 and (rank & (rank - 1) == 0)):
                raise ValueError(f"For lean_walsh, rank must be power of 2, got {rank}")
        self.rank = rank
    
    def _decompose(self, X: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        I, J = X.shape[-2], X.shape[-1]
        random_gen = self._random_gens[self.random_init]
        Omega1 = random_gen(J, rank).to(X.device, X.dtype)
        Omega2 = random_gen(I, rank).to(X.device, X.dtype)
            
        Y1 = X @ Omega1
        Y2 = X.T @ Omega2
            
        Q1, _ = torch.linalg.qr(Y1, mode='reduced')
        Q2, _ = torch.linalg.qr(Y2, mode='reduced')
            
        B = Q1.T @ X @ Q2
            
        U_bar, S, Vh_bar = torch.linalg.svd(B, full_matrices=False)
        U = Q1 @ U_bar
        Vh = (Q2 @ Vh_bar.T).T  
        return U, S, Vh


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
    def _decompose(self, X: torch.Tensor, rank: int = None):
        rank = rank or self.rank or self.estimate_stable_rank(X)
        # create sub matrices for CUR-decompostion
        c, w, r = self.select_rows_cols(X, rank)
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
            rank: int,
            p=2) -> Tuple[torch.Tensor]:
        # Evaluate norms for columns and rows
        col_probs, row_probs = self._importance(X, p)

        column_indices = torch.sort(torch.argsort(col_probs, descending=True)[:rank]).values
        row_indices = torch.sort(torch.argsort(row_probs, descending=True)[:rank]).values

        C_matrix = X[:, column_indices] 
        R_matrix = X[row_indices, :]
        W_matrix = X[row_indices, :][:, column_indices]

        return C_matrix, W_matrix, R_matrix

    def compose(self, *factors, **kwargs):
        C, U, R = factors
        return C @ U @ R
