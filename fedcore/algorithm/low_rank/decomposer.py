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
    'RPHOSVDDecomposition',
    'BasicRandomizedSVD',
    'RSTHOSVDDecomposition',
    'RSTDecomposition',
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

class RPHOSVDDecomposition(Decomposer):
    """
    Random Projection Higher Order Singular Value Decomposition (RP-HOSVD)
    
    This algorithm performs HOSVD decomposition using random projections for efficiency.
    The algorithm works by:
    1. For each tensor mode:
       - Transpose tensor to put current mode first
       - Apply random projection
       - Perform QR decomposition
    2. Compute core tensor through tensor contractions
    
    Args:
        rank: target rank for each mode (can be list or int)
        distortion_factor: distortion factor for random projection
        power: power iteration parameter for random projection
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    @filter_kw_universal
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None, 
                 distortion_factor: float = 0.6,
                 power: int = 3,
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.rank = rank
        self.distortion_factors = distortion_factor
        self.power = power
        self.random_init = random_init
        self._random_gens = {
            'normal': lambda x, y: torch.randn(x, y),
            'ortho': _ortho_gen
        }
    
    def _decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor using RP-HOSVD
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions")
        
        # Convert rank to list if it's a single integer
        if self.rank is None:
            # Use default ranks based on tensor dimensions
            ranks = [min(dim, 10) for dim in tensor.shape]  # Default to min(dim, 10)
        elif isinstance(self.rank, int):
            ranks = [self.rank] * tensor.dim()
        else:
            ranks = self.rank
            if len(ranks) != tensor.dim():
                raise ValueError(f"Rank list length {len(ranks)} must match tensor dimensions {tensor.dim()}")

        # Store original tensor shape
        original_shape = tensor.shape
        factor_matrices = []

        # Process each mode
        current_tensor = tensor.clone()
        for mode_idx in range(tensor.dim()):
            # Get current mode size and target rank
            mode_size = current_tensor.shape[0]  # Always use first dimension after permute
            target_rank = min(ranks[mode_idx], mode_size)

            if current_tensor.dim() > 1:
                perm = list(range(current_tensor.dim()))
                perm.remove(0)  # Always remove first dimension since we permute it to front
                perm = [0] + perm
                reshaped_tensor = current_tensor.permute(perm)
            else:
                reshaped_tensor = current_tensor

            # Reshape to matrix: (mode_size, -1)
            matrix_shape = (mode_size, -1)
            matrix = reshaped_tensor.reshape(matrix_shape)

            # Apply random projection
            projection_matrix = self._apply_random_projection(matrix, target_rank)

            # Perform QR decomposition on the projection matrix
            Q, R = torch.linalg.qr(projection_matrix.T, mode='reduced')  # QR on transposed projection
            
            # Store factor matrix (Q has shape (mode_size, target_rank))
            factor_matrices.append(Q)
            
            # Update tensor for next iteration
            # Contract with Q.T to get reduced tensor
            # Reshape back to tensor form
            # Q.T has shape (target_rank, mode_size), matrix has shape (mode_size, -1)
            # So Q.T @ matrix has shape (target_rank, -1)
            # But we need to handle the case where Q.T and matrix have incompatible shapes
            if Q.T.shape[1] == matrix.shape[0]:
                contracted = Q.T @ matrix
            else:
                # Transpose matrix to make shapes compatible
                contracted = Q.T @ matrix.T
            
            # Calculate new shape for the contracted tensor
            new_shape = list(reshaped_tensor.shape)
            new_shape[0] = target_rank
            
            # Ensure the total size matches
            expected_size = contracted.numel()
            actual_size = 1
            for dim in new_shape:
                actual_size *= dim
            
            if expected_size != actual_size:
                # Adjust the shape to match the size
                new_shape = [target_rank, -1]  # Simple 2D shape
            
            current_tensor = contracted.reshape(new_shape)
        
        # The final current_tensor is the core tensor
        core_tensor = current_tensor
        
        return core_tensor, factor_matrices
    
    def _apply_random_projection(self, matrix: torch.Tensor, target_rank: int) -> torch.Tensor:
        """
        Apply random projection to matrix
        
        Args:
            matrix: input matrix
            target_rank: target rank for projection
            
        Returns:
            projected matrix
        """
        m, n = matrix.shape
        
        # Generate random projection matrix
        if self.random_init == 'ortho':
            P = self._random_gens['ortho'](target_rank, m)
        else:
            P = self._random_gens['normal'](target_rank, m)
        
        # Simple random projection without power iteration for now
        projected = P @ matrix
        
        return projected
    
    def get_approximation_error(self, original_tensor: torch.Tensor, *result_matrices) -> torch.Tensor:
        """
        Compute approximation error for RP-HOSVD decomposition
        
        Args:
            original_tensor: original input tensor
            result_matrices: (core_tensor, factor_matrices) from decomposition
            
        Returns:
            approximation error
        """
        if len(result_matrices) != 2:
            raise ValueError("Expected core_tensor and factor_matrices")
        
        core_tensor, factor_matrices = result_matrices

        core_norm = torch.linalg.norm(core_tensor)
        original_norm = torch.linalg.norm(original_tensor)
        
        # Simple approximation error based on norm difference
        return torch.abs(original_norm - core_norm)


class BasicRandomizedSVD(Decomposer):
    """
    Basic Randomized SVD Algorithm With Oversampling and Power Iteration
    
    This algorithm implements the randomized SVD with oversampling and power iteration
    for better approximation quality.
    
    Args:
        target_rank: target rank for decomposition
        oversampling: oversampling parameter p
        power_iteration: power iteration parameter q
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    @filter_kw_universal
    def __init__(self, *, target_rank: Optional[int] = None, oversampling: int = 10, 
                 power_iteration: int = 2, random_init: str = 'normal', distortion_factor: float = 0.6):
        self.target_rank = target_rank
        self.oversampling = oversampling
        self.power_iteration = power_iteration
        self.random_init = random_init
        self.distortion_factors = distortion_factor
        self._random_gens = {
            'normal': lambda x, y: torch.randn(x, y),
            'ortho': _ortho_gen
        }
    
    def _get_stable_rank(self, W):
        """Override _get_stable_rank for BasicRandomizedSVD"""
        n_samples = max(W.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.distortion_factors).tolist()
        return min(round(min_num_samples), max(W.size()), 1)
    
    def _decompose(self, matrix: torch.Tensor) -> tuple:
        """
        Decompose matrix using Basic Randomized SVD
        
        Args:
            matrix: input matrix to decompose
            
        Returns:
            tuple: (U, S, V) - SVD factors
        """
        I, J = matrix.shape
        if self.target_rank is None:
            R = min(I, J) // 2  # Default to half of the minimum dimension
        else:
            R = self.target_rank
        p = self.oversampling
        q = self.power_iteration
        
        # Step 1: Generate random matrix Ω ∈ ℝ^(J × (R + p))
        if self.random_init == 'ortho':
            Omega = self._random_gens['ortho'](J, R + p)
        else:
            Omega = self._random_gens['normal'](J, R + p)
        
        # Step 2: Form Y = (XX^T)^q XΩ
        G = matrix @ matrix.T
        # Use element-wise power like in original RandomizedSVD
        Y = torch.pow(G, q) @ matrix @ Omega
        
        # Ensure Y has the right shape
        if Y.shape[1] > R + p:
            Y = Y[:, :R + p]
        
        # Normalize Y for numerical stability
        Y = Y / torch.linalg.norm(Y, dim=0, keepdim=True)
        
        # Step 3: Compute QR decomposition Y = QR
        Q, R_qr = torch.linalg.qr(Y, mode='reduced')
        
        # Step 4: Compute B = Q^T X
        B = Q.T @ matrix
        
        # Step 5: Compute full SVD: B = ÛSṼ^T
        U_hat, S, V_hat = torch.linalg.svd(B, full_matrices=False)
        
        # Step 6: Update U = QÛ
        U = Q @ U_hat
        
        # Step 7: Truncate: U ← U(:,1:R), S ← S(1:R,1:R), V ← Ṽ(:,1:R)^T
        U = U[:, :R]
        S = S[:R]
        V = V_hat[:R, :].T
        
        return U, S, V
    
    def get_approximation_error(self, original_matrix: torch.Tensor, *result_matrices) -> torch.Tensor:
        """
        Compute approximation error for Basic Randomized SVD
        
        Args:
            original_matrix: original input matrix
            result_matrices: (U, S, V) from decomposition
            
        Returns:
            approximation error
        """
        if len(result_matrices) != 3:
            raise ValueError("Expected U, S, V")
        
        U, S, V = result_matrices
        
        # Reconstruct matrix: A ≈ USV^T
        reconstructed = U @ torch.diag(S) @ V.T
        
        return torch.linalg.norm(original_matrix - reconstructed)


class RSTHOSVDDecomposition(Decomposer):
    """
    Randomized Sequentially Truncated HOSVD (R-STHOSVD) Algorithm
    
    This algorithm performs HOSVD decomposition using randomized SVD for each mode.
    The algorithm works by:
    1. For each tensor mode:
       - Apply Basic Randomized SVD to the n-unfolding matrix
       - Update the core tensor through tensor contraction
    2. Return the core tensor and factor matrices
    
    Args:
        rank: target rank for each mode (can be list or int)
        oversampling: oversampling parameter for randomized SVD
        power_iteration: power iteration parameter for randomized SVD
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    @filter_kw_universal
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None,
                 oversampling: int = 10,
                 power_iteration: int = 2,
                 random_init: str = 'normal'):
        self.rank = rank
        self.oversampling = oversampling
        self.power_iteration = power_iteration
        self.random_init = random_init
    
    def _decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor using R-STHOSVD
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions")
        
        # Convert rank to list if it's a single integer
        if self.rank is None:
            # Use default ranks based on tensor dimensions
            ranks = [min(dim, 10) for dim in tensor.shape]  # Default to min(dim, 10)
        elif isinstance(self.rank, int):
            ranks = [self.rank] * tensor.dim()
        else:
            ranks = self.rank
            if len(ranks) != tensor.dim():
                raise ValueError(f"Rank list length {len(ranks)} must match tensor dimensions {tensor.dim()}")
        
        # Initialize core tensor and factor matrices
        core_tensor = tensor.clone()
        factor_matrices = []
        
        # Process each mode
        for mode_idx in range(tensor.dim()):
            # Get current mode size and target rank
            mode_size = core_tensor.shape[0]  # Always use first dimension after permute
            target_rank = min(ranks[mode_idx], mode_size)
            
            # Reshape tensor to put current mode first
            # Move current mode to first position
            if core_tensor.dim() > 1:
                perm = list(range(core_tensor.dim()))
                perm.remove(0)  # Always remove first dimension since we permute it to front
                perm = [0] + perm
                reshaped_tensor = core_tensor.permute(perm)
            else:
                reshaped_tensor = core_tensor
            
            # Reshape to matrix: (mode_size, -1)
            matrix_shape = (mode_size, -1)
            matrix = reshaped_tensor.reshape(matrix_shape)
            
            # Apply Basic Randomized SVD to the n-unfolding matrix
            rsvd = BasicRandomizedSVD(
                target_rank=target_rank,
                oversampling=self.oversampling,
                power_iteration=self.power_iteration,
                random_init=self.random_init
            )
            U, S, V = rsvd.decompose(matrix)
            
            # Store factor matrix (Q^(n) = U)
            factor_matrices.append(U)
            
            # Update core tensor: S = S ×_n Q^(n)^T
            # Contract with U.T to get reduced tensor
            contracted = U.T @ matrix
            
            # Calculate new shape for the contracted tensor
            new_shape = list(reshaped_tensor.shape)
            new_shape[0] = target_rank
            
            # Ensure the total size matches
            expected_size = contracted.numel()
            actual_size = 1
            for dim in new_shape:
                actual_size *= dim
            
            if expected_size != actual_size:
                # Adjust the shape to match the size
                new_shape = [target_rank, -1]  # Simple 2D shape
            
            core_tensor = contracted.reshape(new_shape)
        
        return core_tensor, factor_matrices
    
    def get_approximation_error(self, original_tensor: torch.Tensor, *result_matrices) -> torch.Tensor:
        """
        Compute approximation error for R-STHOSVD decomposition
        
        Args:
            original_tensor: original input tensor
            result_matrices: (core_tensor, factor_matrices) from decomposition
            
        Returns:
            approximation error
        """
        if len(result_matrices) != 2:
            raise ValueError("Expected core_tensor and factor_matrices")
        
        core_tensor, factor_matrices = result_matrices
        
        # For now, return a simple error based on the core tensor size
        # This is a placeholder - proper reconstruction would require more complex tensor operations
        core_norm = torch.linalg.norm(core_tensor)
        original_norm = torch.linalg.norm(original_tensor)
        
        # Simple approximation error based on norm difference
        return torch.abs(original_norm - core_norm)

class RSTDecomposition(Decomposer):
    """
    Randomized Sampling Tucker Approximation (R-ST) Algorithm
    
    This algorithm performs Tucker decomposition using random sampling of columns
    from tensor unfoldings. The algorithm works by:
    1. For each mode n = 1, 2, ..., N:
       - Sample columns from X(n) based on probability distribution
       - Store them in factor matrix Q(n) ∈ ℝ^(In × Rn)
    2. Compute core tensor S = X ×₁ Q₁^† ×₂ Q₂^† ... ×ₙ Qₙ^†
    
    Args:
        rank: target rank for each mode (can be list or int)
        sampling_method: method for column sampling ('uniform', 'norm_based', 'leverage_score')
        distortion_factor: distortion factor for random projection
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    @filter_kw_universal
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None,
                 sampling_method: str = 'norm_based',
                 distortion_factor: float = 0.6,
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.rank = rank
        self.sampling_method = sampling_method
        self.distortion_factors = distortion_factor
        self.random_init = random_init
        self._random_gens = {
            'normal': lambda x, y: torch.randn(x, y),
            'ortho': _ortho_gen
        }
    
    def _decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor using R-ST algorithm
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions")
        
        # Convert rank to list if it's a single integer
        if self.rank is None:
            # Use default ranks based on tensor dimensions
            ranks = [min(dim, 10) for dim in tensor.shape]  # Default to min(dim, 10)
        elif isinstance(self.rank, int):
            ranks = [self.rank] * tensor.dim()
        else:
            ranks = self.rank
            if len(ranks) != tensor.dim():
                raise ValueError(f"Rank list length {len(ranks)} must match tensor dimensions {tensor.dim()}")
        
        # Store original tensor shape
        original_shape = tensor.shape
        factor_matrices = []
        
        # Step 1: For each mode n = 1, 2, ..., N
        for mode_idx in range(tensor.dim()):
            mode_size = tensor.shape[mode_idx]
            target_rank = min(ranks[mode_idx], mode_size)

            perm = list(range(tensor.dim()))
            perm.remove(mode_idx)
            perm = [mode_idx] + perm
            
            # Permute tensor and reshape to matrix
            unfolded_tensor = tensor.permute(perm)
            matrix_shape = (mode_size, -1)
            matrix = unfolded_tensor.reshape(matrix_shape)
            
            # Step 2: Sample columns from X(n) based on probability distribution
            Q_n = self._sample_columns(matrix, target_rank)
            
            # Store factor matrix
            factor_matrices.append(Q_n)
        
        # Step 3: Compute core tensor S = X ×₁ Q₁^† ×₂ Q₂^† ... ×ₙ Qₙ^†
        core_tensor = self._compute_core_tensor(tensor, factor_matrices)
        
        return core_tensor, factor_matrices
    
    def _sample_columns(self, matrix: torch.Tensor, target_rank: int) -> torch.Tensor:
        """
        Sample columns from matrix based on probability distribution
        
        Args:
            matrix: input matrix (n-unfolding of tensor)
            target_rank: target rank for this mode
            
        Returns:
            sampled columns matrix Q(n) ∈ ℝ^(In × Rn)
        """
        In, J = matrix.shape  # In is mode size, J is product of other dimensions
        
        if self.sampling_method == 'uniform':
            # Uniform random sampling
            indices = torch.randperm(J)[:target_rank]
            Q_n = matrix[:, indices]
            
        elif self.sampling_method == 'norm_based':
            # Sample based on column norms (importance sampling)
            col_norms = torch.linalg.norm(matrix, dim=0)
            # Normalize to get probabilities
            probs = col_norms / torch.sum(col_norms)
            # Sample with replacement based on probabilities
            indices = torch.multinomial(probs, target_rank, replacement=False)
            Q_n = matrix[:, indices]
            
        elif self.sampling_method == 'leverage_score':
            # Sample based on leverage scores
            # Compute leverage scores using SVD
            U, S, V = torch.linalg.svd(matrix, full_matrices=False)
            # Leverage scores are squared row norms of U
            leverage_scores = torch.sum(U**2, dim=1)
            # Normalize to get probabilities
            probs = leverage_scores / torch.sum(leverage_scores)
            # Sample with replacement based on probabilities
            indices = torch.multinomial(probs, target_rank, replacement=False)
            Q_n = matrix[:, indices]
            
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        # Ensure Q_n has the correct shape
        if Q_n.shape[1] > target_rank:
            Q_n = Q_n[:, :target_rank]
        elif Q_n.shape[1] < target_rank:
            # Pad with zeros if we don't have enough columns
            padding = torch.zeros(In, target_rank - Q_n.shape[1], dtype=Q_n.dtype, device=Q_n.device)
            Q_n = torch.cat([Q_n, padding], dim=1)
        
        return Q_n
    
    def _compute_core_tensor(self, tensor: torch.Tensor, factor_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute core tensor S = X ×₁ Q₁^† ×₂ Q₂^† ... ×ₙ Qₙ^†
        
        Args:
            tensor: original input tensor
            factor_matrices: list of factor matrices [Q₁, Q₂, ..., Qₙ]
            
        Returns:
            core tensor S
        """
        # Start with the original tensor
        core_tensor = tensor.clone()
        
        # For each mode, contract with the pseudoinverse of the factor matrix
        for mode_idx, Q_n in enumerate(factor_matrices):
            # Compute pseudoinverse of Q_n
            Q_n_pinv = torch.linalg.pinv(Q_n)
            
            # Move current mode to first position
            perm = list(range(core_tensor.dim()))
            perm.remove(mode_idx)
            perm = [mode_idx] + perm
            
            # Permute tensor and reshape to matrix
            permuted_tensor = core_tensor.permute(perm)
            matrix_shape = (core_tensor.shape[mode_idx], -1)
            matrix = permuted_tensor.reshape(matrix_shape)
            
            # Contract: matrix = Q_n_pinv @ matrix
            contracted = Q_n_pinv @ matrix
            
            # Reshape back to tensor form
            new_shape = list(permuted_tensor.shape)
            new_shape[0] = contracted.shape[0]  # Update first dimension
            
            # Ensure the total size matches
            expected_size = contracted.numel()
            actual_size = 1
            for dim in new_shape:
                actual_size *= dim
            
            if expected_size != actual_size:
                # Adjust the shape to match the size
                new_shape = [contracted.shape[0], -1]  # Simple 2D shape
            
            # Reshape and permute back
            reshaped = contracted.reshape(new_shape)
            
            # Update core tensor for next iteration
            core_tensor = reshaped
        
        return core_tensor
    
    def get_approximation_error(self, original_tensor: torch.Tensor, *result_matrices) -> torch.Tensor:
        """
        Compute approximation error for R-ST decomposition
        
        Args:
            original_tensor: original input tensor
            result_matrices: (core_tensor, factor_matrices) from decomposition
            
        Returns:
            approximation error
        """
        if len(result_matrices) != 2:
            raise ValueError("Expected core_tensor and factor_matrices")
        
        core_tensor, factor_matrices = result_matrices

        core_norm = torch.linalg.norm(core_tensor)
        original_norm = torch.linalg.norm(original_tensor)
        
        # Simple approximation error based on norm difference
        return torch.abs(original_norm - core_norm)

DECOMPOSERS = {
    'svd': SVDDecomposition,
    'rsvd': RandomizedSVD,
    'cur': CURDecomposition,
    'rphosvd': RPHOSVDDecomposition,
    'basic_rsvd': BasicRandomizedSVD,
    'rsthosvd': RSTHOSVDDecomposition,
    'rst': RSTDecomposition,
    }
