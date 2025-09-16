import torch
from torch.autograd.profiler import record_function
import numpy as np

class TensorGradUnstructuredProjector:
    """
    N-D projector using unstructured (element-wise) sparsity,
    storing only the values at specified indices.
    """
    def __init__(
        self,
        sparse_ratio: float = 0.25,
        sparse_type: str = "randk",
        verbose: bool = False,
        update_gap_scheduler=None,
        scale: float = 1.0,
        proj_type: str = "std",  # for naming consistency
        warm_restart: bool = False,
        n_iter_max: int = 10,
        scale_by_mask_ratio: bool = False,
    ):
        """
        Parameters:
        -----------
        sparse_ratio: float
            Fraction of elements to keep (if in [0,1]). For topk, that means
            the top k = sparse_ratio * (tensor.numel()). 
        sparse_type: str
            'topk', 'probability', 'randk' (randomk). 
        update_gap_scheduler: Instance of UpdateGapScheduler
            Controls when to update the projection indices.
        scale_by_mask_ratio: bool
            If True, will scale up the recovered tensor by the ratio 
            (total_elem / kept_elem) to preserve overall magnitude.
        """
        self.sparse_ratio = sparse_ratio
        self.sparse_type = sparse_type
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.base_scale = scale
        self.proj_type = proj_type
        self.warm_restart = warm_restart
        self.n_iter_max = n_iter_max
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.scale_factor = scale  # Will be updated if scale_by_mask_ratio=True
        self.device = None
        self.should_update = False
        self._orig_shape = None
        self._indices = None
        self._last_iter = -1
        
        print(f"Update gap scheduler: {self.update_gap_scheduler}")
        print(f"UnstructuredSparseProjector initialized with sparse_ratio={self.sparse_ratio}, sparse_type={self.sparse_type}, scale_by_mask_ratio={self.scale_by_mask_ratio}")
        
    def should_update_projector(self, iteration):
        """Check if the projector indices should be updated in this iteration"""
        if self._indices is None:
            self.should_update = True
        if self.update_gap_scheduler is not None:
            self.should_update = self.update_gap_scheduler.should_update(iteration)
        return self.should_update

    def project(self, full_grad: torch.Tensor, iteration: int) -> torch.Tensor:
        with record_function("### UNSTRUCTURED_SPARSE_PROJECT_FORWARD"):
            if self._orig_shape is None:
                self._orig_shape = full_grad.shape

            # Only update indices if necessary
            if (self._indices is None) or self.should_update_projector(iteration):
                with torch.no_grad():
                    self._build_indices(full_grad)

            # Just return the values at the selected indices
            flat = full_grad.view(-1)
            result = flat[self._indices]
                
            return result
    
    def project_back(
        self,
        small_grad: torch.Tensor,
        output_buffer: torch.Tensor = None,
        alpha: float = 1.0,
        accumulate: bool = False
    ) -> torch.Tensor:
        """
        Back-project a sparse vector into `output_buffer`, either overwriting
        (accumulate=False) or adding to existing contents (accumulate=True).
        
        Args:
            small_grad: Tensor of values for the selected indices
            output_buffer: Pre-allocated buffer to write into or None to create a new tensor
            alpha: Scaling factor to apply to the values
            accumulate: If True, add to buffer values; if False, overwrite
            
        Returns:
            The output_buffer or a new tensor if output_buffer is None
        """
        with record_function("### UNSTRUCTURED_SPARSE_PROJECT_BACK"):
            # If no buffer provided, create a new one
            if output_buffer is None:
                # Create a zero tensor with the original shape
                output_buffer = torch.zeros(
                    self._orig_shape,
                    dtype=small_grad.dtype,  # Use input dtype for consistency
                    device=small_grad.device
                )
                # For new buffers, we always overwrite (accumulate flag is ignored)
                accumulate = False
            else:
                # Ensure buffer shape matches original shape
                assert output_buffer.shape == self._orig_shape, f"Buffer shape {output_buffer.shape} doesn't match original shape {self._orig_shape}"                
                # Ensure consistent dtype between small_grad and output_buffer
                if small_grad.dtype != output_buffer.dtype:
                    if self.verbose:
                        print(f"Converting small_grad from {small_grad.dtype} to {output_buffer.dtype} for consistency")
                    small_grad = small_grad.to(dtype=output_buffer.dtype)
            
            # Scale values once (combining alpha and scale_factor)
            # Convert to the same dtype as the output buffer to avoid dtype mismatch in scatter_
            vals = (small_grad * (self.scale_factor * alpha)).to(output_buffer.dtype)
            
            # Store original shape to reshape back at the end
            original_shape = output_buffer.shape
            
            # Get flattened version of buffer - try to use view first
            if output_buffer.is_contiguous():
                flat = output_buffer.view(-1)
            else:
                # Need to reshape if not contiguous, but this creates a new tensor
                # We'll need to copy back to the original buffer at the end
                flat = output_buffer.reshape(-1)
            
            # Check if we're using ComplexHalf dtype which doesn't support scatter operations
            is_complex_half = flat.dtype == torch.complex32
            
            # If using ComplexHalf, temporarily convert to a supported type
            if is_complex_half:
                temp_buffer = flat.to(torch.complex64)
                temp_vals = vals.to(torch.complex64)
                temp_indices = self._indices
                
                if not accumulate:
                    # Overwrite mode: clear only the positions we touch
                    temp_buffer.scatter_(0, temp_indices, temp_vals)
                else:
                    # Accumulate mode: add to whatever is already in the buffer
                    temp_buffer.scatter_add_(0, temp_indices, temp_vals)
                
                # Convert back to ComplexHalf
                flat.copy_(temp_buffer.to(torch.complex32))
            else:
                if not accumulate:
                    # Overwrite mode: clear only the positions we touch
                    flat.scatter_(0, self._indices, vals)
                else:
                    # Accumulate mode: add to whatever is already in the buffer
                    flat.scatter_add_(0, self._indices, vals)
            
            # If we had to reshape, copy the modified flat tensor back to output_buffer
            if not output_buffer.is_contiguous():
                # Reshape flat back to original shape
                modified = flat.reshape(original_shape)
                # Copy back to the original buffer
                output_buffer.copy_(modified)
                
            return output_buffer

    def _build_indices(self, x: torch.Tensor):
        """
        Build a 1D LongTensor of indices to keep in the flattened tensor.
        """
        with torch.no_grad():
            flat = x.view(-1)
            numel = flat.numel()
            k = max(1, int(self.sparse_ratio * numel))

            if self.sparse_type.lower() == "topk":
                # pick topk by absolute value
                vals = flat.abs()
                topk_vals, idx = torch.topk(vals, k)
            elif self.sparse_type.lower() == "probability":
                # Probability is weighted by absolute value
                vals = flat.abs()
                probs = vals / (vals.sum() + 1e-12)
                idx = torch.multinomial(probs, k, replacement=False)
            elif self.sparse_type.lower() in ("randk", "randomk"):
                # Use numpy for efficient CPU-side sampling
                cpu_idx = np.random.choice(numel, k, replace=False)
                idx = torch.from_numpy(cpu_idx).to(x.device, dtype=torch.long)
                del cpu_idx
            else:
                raise ValueError(f"Unsupported sparse_type: {self.sparse_type}")

            # Sort for better memory locality
            self._indices = idx.sort().values
            
            del idx
            torch.cuda.empty_cache()

            # Possibly compute scaling factor to preserve total norm
            if self.scale_by_mask_ratio:
                # sqrt to preserve L2 norm
                self.scale_factor = self.base_scale * torch.sqrt(torch.tensor(numel / k))
                # Only do it once
                self.scale_by_mask_ratio = False
                if self.verbose:
                    print(f"Set scale factor to {self.scale_factor:.4f}")
