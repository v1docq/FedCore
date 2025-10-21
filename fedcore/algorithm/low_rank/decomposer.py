"""
Low-rank matrix decomposition utilities for FedCore.

This module groups several decomposition strategies used to produce
compact (low-rank) approximations of weight matrices inside decomposed
layers:

- SVDDecomposition:        deterministic thin SVD via `torch.linalg.svd`.
- RandomizedSVD:           randomized range finder + power iterations + small SVD.
- CURDecomposition:        column–row (CUR) factorization using importance sampling.
- DECOMPOSERS:             a small registry that maps string keys to classes.

Key implementation ideas
------------------------
* For very large inputs a heuristic (`_is_big`) switches to a more memory-friendly
  path (if provided by the decomposer).
* The helper decorator `_need_t` may transpose a matrix before running a method
  to improve numerical behavior for projection-based algorithms (and then
  restores the original orientation of the returned factors).
* A Johnson–Lindenstrauss bound (`johnson_lindenstrauss_min_dim`) is used to
  estimate a "stable" target dimension when the requested rank is not provided.

Notes
-----
This module is intentionally minimal: the algorithms are pragmatic, with
defaults that work well for typical NN weight shapes. If you need tighter
error control, compute `get_approximation_error` and adjust rank/power.
"""

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
    """
    Decorator that transparently transposes the input matrix to ensure
    better numerical behavior for projection-based algorithms.

    Rationale
    ---------
    Some randomized / projection-based decompositions exhibit more stable
    behavior when the "tall" dimension is treated as columns. If the input
    matrix `W` has shape (m, n) with m >= n, we temporarily apply a transpose,
    run the wrapped method, and then reverse the transpose on the outputs.

    Parameters
    ----------
    f : Callable
        A decomposition method that accepts a 2D tensor and returns a tuple
        of tensors (e.g., factors of a low-rank decomposition).

    Returns
    -------
    Callable
        Wrapped function that may transpose `W` internally while keeping the
    outward-facing API unchanged.
    """
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
    """
    Generate a random orthogonal (semi-orthogonal) matrix using PyTorch initializer.

    Parameters
    ----------
    x : int
        Number of rows.
    y : int
        Number of columns.

    Returns
    -------
    torch.Tensor
        A tensor of shape (x, y) with orthogonal columns/rows depending on shape.
    """
    P = torch.empty(x, y)
    torch.nn.init.orthogonal_(P)
    return P


class Decomposer(ABC):
    """
    Abstract base class for matrix decomposers used in low-rank approximations.

    The class chooses an algorithm branch for "big" matrices to prevent excessive
    memory/compute usage. Concrete implementations should override `_decompose`
    and may override `_decompose_big` if a specialized path is needed.
    """

    def decompose(self, W: torch.Tensor, *args, **kwargs):
        """
        Public entrypoint for decomposition.

        If the input is considered "big" according to `_is_big`, the method
        delegates to `_decompose_big`, otherwise to `_decompose`.

        Parameters
        ----------
        W : torch.Tensor
            Matrix to decompose.
        *args, **kwargs :
            Implementation-specific arguments.

        Returns
        -------
        tuple
            A tuple of tensors representing the factorization (format depends on subclass).
        """
        if not self._is_big(W):
            return self._decompose(W, *args, **kwargs)
        else:
            return self._decompose_big(W, *args, **kwargs)

    def _is_big(self, W: torch.Tensor):
        """
        Heuristic check for large inputs.

        Returns True if the sum of dimensions exceeds `DIM_SUM_LIM` or any
        individual dimension exceeds `DIM_LIM`. Intended to switch to a more
        memory-conscious algorithm variant.

        Parameters
        ----------
        W : torch.Tensor
            Input matrix.

        Returns
        -------
        bool
        """
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())

    @abstractmethod
    def _decompose(self, W, *args, **kwargs):
        """
        Core decomposition routine for regular-sized inputs.

        Must be implemented by subclasses.

        Parameters
        ----------
        W : torch.Tensor
            Input matrix.

        Returns
        -------
        tuple
            Decomposition outputs (e.g., (U, S, Vh) for SVD-like methods).
        """
        pass

    def _decompose_big(self, W, *args, **kwargs):
        """
        Decomposition routine for large inputs.

        Subclasses may override to provide a numerically efficient or
        memory-aware variant. By default, it falls back to `_decompose`.

        Parameters
        ----------
        W : torch.Tensor
            Input matrix.

        Returns
        -------
        tuple
            Decomposition outputs.
        """
        return self._decompose(W, *args, **kwargs)

    def _get_stable_rank(self, W):
        """
        Estimate a target dimensionality using the Johnson–Lindenstrauss bound.

        The method queries scikit-learn's `johnson_lindenstrauss_min_dim` for
        the minimal embedding dimension given an allowed distortion and the
        maximum dimension of `W`. The result is clamped to a reasonable integer
        within the tensor's size envelope.

        Parameters
        ----------
        W : torch.Tensor
            Input matrix.

        Returns
        -------
        int
            Estimated stable rank / target dimension.
        """
        n_samples = max(W.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.distortion_factors).tolist()
        return min((round(max(min_num_samples)), *W.size(), 1))

    def get_approximation_error(self, W, *result_matrices):
        """
        Compute Frobenius norm of the reconstruction error.

        Parameters
        ----------
        W : torch.Tensor
            Original matrix.
        *result_matrices :
            Factors that reconstruct an approximation of `W` via chained matmul.

        Returns
        -------
        torch.Tensor
            Scalar tensor with ||W - (Π factors)||_F.
        """
        approx = reduce(torch.matmul, result_matrices)
        return torch.linalg.norm(W - approx)


class SVDDecomposition(Decomposer):
    """
    Deterministic thin SVD using `torch.linalg.svd`.

    Suitable when an exact (or numerically stable) SVD is feasible. Returns
    economical factors with `full_matrices=False`.
    """

    def _decompose(self, W: torch.Tensor) -> tuple:
        """
        Compute the thin SVD of a matrix.

        Parameters
        ----------
        W : torch.Tensor
            Matrix to decompose.

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            Tuple `(U, S, Vh)` such that `W ≈ U @ diag(S) @ Vh`.
        """
        # Return classic svd decomposition
        return torch.linalg.svd(W, full_matrices=False)

    def get_approximation_error(self, W, *result_matrices):
        """
        Reconstruction error specialized for SVD factors.

        Parameters
        ----------
        W : torch.Tensor
            Original matrix.
        *result_matrices :
            `(U, S, Vh)` from SVD.

        Returns
        -------
        torch.Tensor
            Frobenius norm of the difference between `W` and its SVD-based approximation.
        """
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)


class RandomizedSVD(Decomposer):
    """
    Randomized SVD with power iterations.

    This variant uses randomized range finding and power iteration to obtain
    a low-rank subspace, followed by a small SVD on the projected matrix.

    Notes
    -----
    - `power` increases spectral gap separation but adds cost.
    - `distortion_factor` controls the JL-based stable rank estimation.
    - `random_init` selects a distribution for test matrices ('normal' or 'ortho').

    Reference
    ---------
    See, e.g., Halko et al., and recent surveys (the exact algorithmic choices here
    follow a pragmatic randomized SVD pipeline).
    """
    _random_gens = {
        'normal': lambda x, y : torch.randn(x, y),
        'ortho' : _ortho_gen
    }

    @filter_kw_universal
    def __init__(self, *, power: int = 3,
                 distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        """
        Parameters
        ----------
        power : int, default=3
            Number of power iterations to sharpen the spectrum.
        distortion_factor : float, default=0.6
            Allowed distortion for JL dimension estimation (0, 1].
        random_init : str, {'normal', 'ortho'}, default='normal'
            Strategy for building random test matrices.
        """
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.power = power
        self.distortion_factors = distortion_factor
        self.random_init = random_init

    @_need_t
    def _decompose_big(self, X):
        """
        Randomized SVD path used for large inputs.

        Steps
        -----
        1) Build a random probe `P`.
        2) Form a powered Gram-like operator `G` to emphasize leading subspace.
        3) Orthonormalize to obtain `Q`.
        4) Compute a small SVD on `B = X @ Q`.
        5) Map back the right singular vectors.

        Parameters
        ----------
        X : torch.Tensor
            Input matrix.

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            `(U, S, Vh)` approximation.
        """
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
        """
        Randomized SVD path for regular-sized inputs.

        Parameters
        ----------
        X : torch.Tensor
            Input matrix.

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            `(U, S, Vh)` approximation.
        """
        G = X @ X.T
        P = torch.randn(X.size(1), self._get_stable_rank(X))
        Q, _ = torch.linalg.qr(torch.pow(G, self.power) @ X @ P, mode='reduced')
        B = Q.T @ X
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return Q @ U, S, Vh

    def get_approximation_error(self, W, *result_matrices):
        """
        Reconstruction error specialized for (randomized) SVD outputs.

        Parameters
        ----------
        W : torch.Tensor
            Original matrix.
        *result_matrices :
            `(U, S, Vh)` factors.

        Returns
        -------
        torch.Tensor
            Frobenius norm of the difference between `W` and its approximation.
        """
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)


class CURDecomposition(Decomposer):
    """
    Column–Row (CUR) decomposition for low-rank matrix approximation.

    The method selects a subset of columns (C) and rows (R) to form a
    small intersection matrix (W). The middle factor U is obtained as
    a pseudoinverse of W. The original matrix is approximated as:

        A ≈ C @ U @ R

    Selection is based on normalized column/row norms (importance sampling proxy).
    """

    @filter_kw_universal
    def __init__(self, *, rank: Optional[int] = None, distortion: Union[int, List[int]]):
        """
        Parameters
        ----------
        rank : Optional[int], default=None
            Target rank (number of columns/rows to select). If None, estimated
            via `_get_stable_rank`.
        distortion : int | list[int]
            Distortion parameter kept for API compatibility (not used directly
            in selection here).
        """
        self.stable_rank = rank
        self.distortion = distortion

    def get_aproximation_error(self, original_tensor, cur_matrices: tuple):
        """
        Compute reconstruction error for CUR factors.

        Parameters
        ----------
        original_tensor : torch.Tensor
            Original matrix A.
        cur_matrices : tuple
            `(C, U, R)` tuple.

        Returns
        -------
        torch.Tensor
            Frobenius norm of `A - C @ U @ R`.
        """
        C, U, R = cur_matrices
        return torch.linalg.norm(original_tensor - C @ U @ R)

    def _decompose(self, tensor: torch.Tensor):
        """
        Build CUR factors by selecting top-norm columns/rows.

        Parameters
        ----------
        tensor : torch.Tensor
            Input matrix.

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            `(C, U, R)` where `U = pinv(W)` and `W` is the intersection submatrix.
        """
        if self.stable_rank is None:
            self.stable_rank = self._get_stable_rank(tensor)
        # create sub matrices for CUR-decompostion
        c, w, r = self.select_rows_cols(tensor)
        # evaluate pseudoinverse for W - U^-1
        u = torch.linalg.pinv(w)
        # aprox U using pseudoinverse
        return (c, u, r)

    def _importance(self, X, p):
        """
        Compute column/row importance scores using p-norms on a min–max scaled matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input matrix.
        p : int | float
            Norm order (e.g., 1, 2).

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Normalized probabilities for selecting columns and rows.
        """
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
        """
        Select column/row indices by sorting importance scores and slice submatrices.

        Parameters
        ----------
        X : torch.Tensor
            Input matrix.
        p : int, default=2
            Norm order used in importance scoring.

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            `(C, W, R)` where `C` are selected columns, `R` selected rows,
            and `W` is their intersection.
        """
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
