from typing import Callable

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csr_matrix
import torch
from torch.linalg import matrix_norm, vector_norm, svd, qr, inv
from torch.nn.modules import Module
from torch import Tensor
from fedcore.losses.regularization_losses import RegularizationLoss


class ManifoldRegularization:
    def __init__(self, method):
        self._manifold_type = ['exact', 'exactSVDS', 'orthogonal', 'uppertriangular', 'lowertriangular', 'diagonal']
        self._methods_impl = {'exact': self._evaluate_svd}
        self.method = method

    def _evaluate_svd(self, original_tensor: Tensor, manifold_tensor: Tensor, rank:int) -> Callable:
        Ux, Sx, Vx = svd(original_tensor, full_matrices=False) # step 1. Compute basis for original tensor
        Ux, Sx, Vx = Ux[:, :rank], np.diag(Sx[:rank]), Vx[:, :rank] # step 2. Truncation of basis by chosen rank
        # elif self.method == 'exactSVDS':
        #     Ux, Sx, Vx = svds(original_tensor, k=r)

        Atilde = Ux.T @ manifold_tensor @ Vx @ pinv(Sx) # step 3. Compute projection of original tensor basis onto chosen manifold tensor
        A = lambda v: Ux @ (Atilde @ (Ux.T @ v))
        return A
        # if nargout() == 2:
        #     return A, np.linalg.eigvals(Atilde)
        # elif nargout() > 2:
        #     eVals, eVecs = np.linalg.eig(Atilde)
        #     eVecs = manifold_tensor @ Vx @ pinv(Sx) @ eVecs / eVals
        #     return A, eVals, eVecs
    def _eval_orthogonal(self, original_tensor: Tensor, manifold_tensor: Tensor, rank:int) -> Callable:

        Ux, _, _ = svd(original_tensor, full_matrices=False) # step 1. Compute basis of left eigenvectors for original tensor
        Ux = Ux[:, :rank] # step 2. Truncation of basis by chosen rank
        Yproj = Ux.T @ manifold_tensor # step 3. Compute projection of original tensor basis onto chosen manifold tensor
        Xproj = Ux.T @ original_tensor # step 4. Compute projection of original tensor basis onto original tensor
        CovYX = Yproj @ Xproj.T # step 4. Compute covariance tensor between projected tensors
        Uyx, _, Vyx = svd(CovYX, full_matrices=False) # step 5. Compute "PCA" for covariance tensor
        Aproj = Uyx @ Vyx.T # step 6. Compute covariance tensor "principal components"
        A = lambda x: Ux @ (Aproj @ (Ux.T @ x))
        return A
        # if nargout() == 2:
        #     return A, np.linalg.eigvals(Aproj)
        # elif nargout() > 2:
        #     eVals, eVecs = np.linalg.eig(Aproj)
        #     if nargout() > 3:
        #         return A, np.diag(eVals), Ux @ eVecs, Aproj
        #     return A, np.diag(eVals), Ux @ eVecs

    def _eval_uppertriang(self, original_tensor: Tensor, manifold_tensor: Tensor, rank:int) -> Callable:
        R, Q = qr(original_tensor.T, mode='economic') # step 1. Compute QR for orthogonal basis of original tensor
        R, Q = R.T, Q.T # step 1. Compute QR for orthogonal basis of original tensor
        X_hat = manifold_tensor @ Q.T # step 2. Compute QR for orthogonal basis of original tensor
        Ut = np.triu(X_hat) # step 3. Compute QR for orthogonal basis of original tensor
        A = Ut @ inv(R) # step 3. Compute QR for orthogonal basis of original tenso
        return A

    # def _eval_diag(self):
    #     d = varargin[0] if varargin else np.ones((nx, 2))
    #     if np.isscalar(d):
    #         d = d * np.ones((nx, 2))
    #     elif d.size == nx:
    #         d = np.tile(d[:, None], (1, 2))
    #     elif d.shape != (nx, 2):
    #         raise ValueError('Diagonal number is not in an allowable format.')
    #
    #     Icell, Jcell, Rcell = [], [], []
    #     for j in range(nx):
    #         l1 = max(j - (d[j, 0] - 1), 0)
    #         l2 = min(j + (d[j, 1] - 1), nx - 1)
    #         C = X[l1:l2 + 1, :]
    #         b = Y[j, :]
    #         if method == 'diagonal':
    #             sol = np.linalg.solve(C.T, b.T).T
    #         elif method == 'diagonalpinv':
    #             sol = b @ pinv(C)
    #         elif method == 'diagonaltls':
    #             sol = np.linalg.lstsq(C.T, b.T, rcond=None)[0].T
    #
    #         Icell.append(j * np.ones(l2 - l1 + 1))
    #         Jcell.append(np.arange(l1, l2 + 1))
    #         Rcell.append(sol)
    #
    #     Imat = np.concatenate(Icell).astype(int)
    #     Jmat = np.concatenate(Jcell).astype(int)
    #     Rmat = np.concatenate(Rcell)
    #     Asparse = csr_matrix((Rmat, (Imat, Jmat)), shape=(nx, nx))
    #     A = lambda v: Asparse @ v
    #
    #     if nargout() == 2:
    #         eVals, _ = eigs(Asparse, k=nx)
    #         return A, eVals
    #     elif nargout() > 2:
    #         eVals, eVecs = eigs(Asparse, k=nx)
    #         return A, np.diag(eVals), eVecs
    def evaluate(self, original_tensor: Tensor, manifold_tensor: Tensor, rank: int = None):

        rank = rank if rank is not None else min(original_tensor.shape)


        elif method == 'uppertriangular':
            R, Q = qr(X.T, mode='economic')
            R = R.T
            Q = Q.T  # MATLAB rq = QR decomposition with Q*Q' = I
            Ut = np.triu(Y @ Q.T)
            A = Ut @ np.linalg.inv(R)
            return A

        elif method == 'lowertriangular':
            A = np.rot90(piDMD(np.flipud(X), np.flipud(Y), 'uppertriangular'), 2)
            return A

        elif method.startswith('diagonal'):


        return A


class LaiMSE(RegularizationLoss):
    """MSE with adaptive weighting based on residuals magnitude.

    Implements an adaptive MSE loss where the weight of each residual is determined
    by the combination of two terms, controlled by the balancing factor.

    Args:
        factor: Balances between error-sensitive and constant regularization terms
            (default: ``0.5``).
    """

    def __init__(self, factor: float = 0.5) -> None:
        super().__init__(factor=factor)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculates adaptive MSE loss.

        Args:
            y_pred: Model predictions tensor of shape (N, *).
            y_true: Ground truth tensor of same shape as predictions.
        """
        residuals = y_pred - y_true
        n = y_pred.size(0)
        mse_term = residuals.pow(2)

        k_i = 2 * residuals / n
        k_i_sq = k_i.pow(2)

        if self.factor >= 1:
            term1 = k_i_sq / (1 + k_i_sq)
            term2 = self.factor / (1 + k_i_sq)
        else:
            term1 = k_i_sq / (self.factor * (1 + k_i_sq))
            term2 = 1 / (1 + k_i_sq)

        weight = torch.max(term1, term2)
        return (mse_term * weight).mean()



