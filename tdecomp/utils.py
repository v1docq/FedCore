from functools import partial, wraps

import torch

from torch.ao.quantization.utils import _normalize_kwargs

def filter_kw_universal(f):
    """Automatically switches between fedot-style and conventional init"""
    @wraps(f)
    def _wrapping(self, *args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], dict) and not len(kwargs)):
            params = args[0]
            args = args[1:]
        elif 'params' in kwargs and len(kwargs) == 1:
            params = kwargs['params']
        else:
            params = kwargs
        new_kw = _normalize_kwargs(f, params)
        f(self, *args, **new_kw)

    return _wrapping

def conjugate_gradient(A, b, precond=None, x0=None, 
                       max_iter=100, tol=1e-6, 
                       verbose=False, 
                       device=None):
    """
    Solves Ax = b using the Conjugate Gradient method.
    
    Args:
        A: Linear operator (callable or matrix). If callable, A(x) should return A @ x.
        b: Right-hand side vector (n,)
        x0: Initial guess (n,). If None, uses zero vector.
        max_iter: Maximum iterations
        tol: Tolerance for residual norm
        verbose: Print progress
        
    Returns:
        x: Solution vector (n,)
        residuals: List of residual norms
    """
    eps_zero = 1e-8
    if not device and not callable(A):
        device = A.device
    if callable(precond):
        precond = precond.to(device)
    # Initialize
    x = torch.zeros_like(b, device=device) if x0 is None else x0.clone()
    # print(A.si b.size(), x.size())
    r = b - (A(x) if callable(A) else A @ x)
    if precond is not None:
        z = precond(r) if callable(precond) else precond @ r
    else:
        z = r
    p = z.clone()
    rs_old = r.dot(z)    
    residuals = [torch.norm(r).item()]
    if verbose:
        print(f"Iter 0: Residual = {residuals[-1]:.3e}")
    
    # CG iterations
    for k in range(1, max_iter + 1):
        Ap = A(p) if callable(A) else A @ p
        assert not torch.isnan(r).any(), f'iter {k}'
        alpha = rs_old / (p.dot(Ap) + eps_zero)
        print(p.dot(Ap), alpha)
        
        x += alpha * p
        r -= alpha * Ap
        assert not torch.isnan(x).any(), f'iter {k}'
        residuals.append(torch.norm(r).item())
        
        if verbose and (k % 10 == 0 or k == max_iter):
            print(f"Iter {k}: Residual = {residuals[-1]:.3e}")
        
        if residuals[-1] < tol:
            break
        if precond is None:
            z = p
        else:
            z = precond(r) if callable(precond) else precond @ r
        rs_new = r.dot(z)     
        beta = rs_new / (rs_old + eps_zero)
        p = z + beta * p
        rs_old = rs_new
    assert not torch.isnan(x).any()
    return x, residuals


def svd_solver_tikhonov(A: torch.Tensor, b: torch.Tensor, svd_func=None, tol=1e-6, maxiter=20):
    """
    Solve Ax = b
        A is (m x n)
        b is (m)
    Assume m >= n
    """
    if svd_func is None:
        svd_func = partial(torch.linalg.svd, full_matrices=False)
    lmbd = 1e-4 
    lmbd_decay = 0.8
    U, S, Vh = svd_func(A)
    Utb = U.T @ b
    S2 = torch.square(S)
    for _ in range(maxiter):
        Sinv = S / (S2 + lmbd**2)  # Wiener filter
        x = Vh.T @ (Sinv * Utb)
        if torch.norm(A @ x - b) < tol:
            break 
        lmbd *= lmbd_decay
    return x
    


