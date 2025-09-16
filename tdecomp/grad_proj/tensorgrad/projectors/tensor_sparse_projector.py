import torch
from torch.autograd.profiler import record_function
from typing import List
import numpy as np
import time

def mode_unfolding_norms(tensor, mode):
    """
    Returns L2 norm of each 'row' in the mode-unfolding for dimension=mode.
    If tensor.shape= (D0, D1, ..., Dn), the unfolding w.r.t. mode is shape
    (Dm, product_of_other_dims). We get a vector of length Dm with norms.
    """
    perm = list(range(tensor.ndim))
    # Move 'mode' to front
    perm[0], perm[mode] = perm[mode], perm[0]
    unfolded = tensor.permute(*perm).contiguous()
    unfolded = unfolded.view(tensor.shape[mode], -1)
    row_norms = torch.norm(unfolded, dim=1)
    return row_norms


class TensorGradSparseProjector:
    """
    N-D projector that uses dimension-wise structured sparsity
    instead of Tucker (like TensorGradLowRankProjector). For each dimension i,
    we sample a subset of indices according to norms (topk, probability, etc.).
    """
    def __init__(
        self,
        sparse_ratio: float = 0.25,
        sparse_type: str = "topk",
        verbose: bool = False,
        update_gap_scheduler = None,
        scale: float = 1.0,
        warm_restart: bool = False,
        n_iter_max: int = 10,
        scale_by_mask_ratio: bool = False,
    ):
        self.sparse_ratio = sparse_ratio
        self.sparse_type = sparse_type
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.scale = scale
        self.warm_restart = warm_restart
        self.n_iter_max = n_iter_max
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.scale_factor = 1.0 * self.scale  # will be updated if scale_by_mask_ratio=True

        # store one mask per dimension => a list of length = tensor.ndim
        self.masks = None
        self._orig_shape = None
        self._last_iter = -1

        print(f"TensorGradSparseProjector initialized with sparse_ratio={self.sparse_ratio}, sparse_type={self.sparse_type}, scale_by_mask_ratio={self.scale_by_mask_ratio}")
        
        
    def should_update_projector(self, iter):
        return self.update_gap_scheduler.should_update(iter)

    def project(self, full_rank_grad: torch.Tensor, iteration: int):
        with record_function("### TENSOR_SPARSE_PROJECT_FORWARD"):
            if self._orig_shape is None:
                self._orig_shape = full_rank_grad.shape

            # Only update masks if necessary
            if (self.masks is None) or self.should_update_projector(iteration):
                with torch.no_grad():
                    self.masks = self._build_masks(full_rank_grad)
    
            # Now transform => produce smaller sub-tensor
            smaller = self._transform(full_rank_grad)
            return smaller

    def project_back(self, small_grad: torch.Tensor, output_buffer=None, alpha=1.0, accumulate=False):
        with record_function("### TENSOR_SPARSE_PROJECT_BACK"):
            with torch.no_grad():
                # Create a temporary buffer if none provided
                if output_buffer is None:
                    output_buffer = torch.zeros(self._orig_shape, 
                                           dtype=small_grad.dtype, 
                                           device=small_grad.device)
                full = self._inverse_transform(small_grad, output_buffer, alpha=alpha, accumulate=accumulate)
                return full
    

    def _build_masks(self, tensor: torch.Tensor):
        """
        Build dimension-wise masks for tensor sparsity.
        """
        with torch.no_grad():
            # Ensure sparse_ratio is a list of floats, one per dimension.
            if isinstance(self.sparse_ratio, float):
                # make multipliers for each dimension
                non_one_dims = sum(1 for d in tensor.shape if d != 1)
                even_distributed_ratio = self.sparse_ratio ** (1/non_one_dims)
                # Handle dimensions of size 1 separately
                ratio_list = []
                for d in tensor.shape:
                    if d == 1:
                        ratio_list.append(1)  # Keep ratio 1 for dimensions of size 1
                    else:
                        # Round up for non-1 dimensions
                        ratio_list.append(even_distributed_ratio)
            elif isinstance(self.sparse_ratio, list):#
                # if list of floats, convert to multipliers
                if all(isinstance(r, float) and 0 < r < 1 for r in self.sparse_ratio):
                    ratio_list = [r ** (1/tensor.ndim) for r in self.sparse_ratio]
                else:
                    ratio_list = self.sparse_ratio
            else:
                raise ValueError(f"Invalid sparse_ratio: {self.sparse_ratio}")
            
            masks = []
            for mode in range(tensor.ndim):
                ratio = ratio_list[mode] if mode < len(ratio_list) else ratio_list[-1]
                dim_size = tensor.shape[mode]

                if self.sparse_type.lower() not in ['randomk', 'randk']:
                    # Compute norms directly using mode_unfolding_norms function
                    row_norms = mode_unfolding_norms(tensor, mode)
                else:
                    row_norms = None

                k = max(1, int(ratio * dim_size + 1))
                    
                if self.sparse_type.lower() == 'topk':
                    top_vals, top_idxs = torch.topk(row_norms, k)
                    mask = torch.zeros(dim_size, dtype=torch.bool, device=tensor.device)
                    mask[top_idxs] = True
                elif self.sparse_type.lower() == 'probability':
                    sum_norms = row_norms.sum() + 1e-12
                    probs = row_norms / sum_norms
                    sampled = torch.multinomial(probs, k, replacement=False)
                    mask = torch.zeros(dim_size, dtype=torch.bool, device=tensor.device)
                    mask[sampled] = True
                elif self.sparse_type.lower() in ('randk', 'randomk'):
                    perm = torch.randperm(dim_size, device=tensor.device)
                    picked = perm[:k]
                    mask = torch.zeros(dim_size, dtype=torch.bool, device=tensor.device)
                    mask[picked] = True
                else:
                    raise ValueError(f"Unsupported sparse_type={self.sparse_type}")

                # Keep mask on GPU
                masks.append(mask)
                if row_norms is not None:
                    del row_norms

            if getattr(self, "scale_by_mask_ratio", False):
                # Compute overall scaling factor more efficiently
                scale_factor = 1.0
                for m in masks:
                    kept = m.sum().item()
                    total = m.numel()
                    scale_factor *= total / (kept + 1e-8)
                scale_factor = torch.sqrt(torch.tensor(scale_factor))
                self.scale_factor = scale_factor * self.scale
                self.scale_by_mask_ratio = False
                print(f"[TensorGradSparseProjector] Setting the scale factor: {self.scale_factor}")
            
            if self.verbose:
                kept = [m.sum().item() for m in masks]
                print(f"[TensorGradSparseProjector] Recomputed masks => dims kept: {kept}")

            return masks
    
    def _transform(self, x: torch.Tensor):
        """
        Use masks to select elements from each dimension.
        The result is smaller: shape= (K0, K1, ..., Kn).
        """
        res = x
        for mode, mask in enumerate(self.masks):
            # Create a slice object for this dimension
            slice_obj = [slice(None)] * x.ndim
            slice_obj[mode] = mask
            slice_obj = tuple(slice_obj)
            # Apply the mask to this dimension
            res = res[slice_obj]

        return res

    def _inverse_transform(self, small_x: torch.Tensor, output_buffer=None, alpha=1.0, accumulate=False):
        """
        Start from zero[full_shape], fill in the sub-tensor 'small_x'
        dimension by dimension.
        """
        with torch.no_grad():
            res = small_x * self.scale_factor
            nd = len(self._orig_shape)
            
            # Create or use the provided buffer
            if output_buffer is None:
                output_buffer = torch.zeros(self._orig_shape, dtype=res.dtype, device=res.device)

                
            # Process each mode in reverse order
            for mode in reversed(range(nd)):
                # Mask is already on GPU
                mask = self.masks[mode]
                
                # Create slice object for the current mode
                slice_obj = [slice(None)] * nd
                slice_obj[mode] = mask
                slice_obj = tuple(slice_obj)
                
                # Create a temporary tensor of the correct shape
                shape_expanded = list(res.shape)
                shape_expanded[mode] = self._orig_shape[mode]
                bigger = torch.zeros(shape_expanded, dtype=res.dtype, device=res.device)
                
                # Fill in the values
                bigger[slice_obj] = res
                res = bigger
                
            # Final update to the buffer
            if accumulate:
                output_buffer.add_(res, alpha=alpha)
            else:
                output_buffer.copy_(res * alpha)
                
            return output_buffer
    
