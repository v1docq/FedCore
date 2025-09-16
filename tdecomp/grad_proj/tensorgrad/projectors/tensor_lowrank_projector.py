from typing import *

import torch
import json
import os
from tensorly.decomposition import tucker
from tensorly import tenalg 
from torch.autograd.profiler import record_function

from tensorly.tenalg.core_tenalg.n_mode_product import multi_mode_dot


class TensorGradLowRankProjector:
    def __init__(
        self, 
        rank, 
        update_gap_scheduler,  # Instance of UpdateGapScheduler
        verbose=False, 
        scale=1.0, 
        warm_restart=False,
        n_iter_max=10,
        svd_type: Union[Literal['truncated_svd', 'randomized_svd'], Callable]="truncated_svd", # "truncated_svd", "randomized_svd"
    ):
        """
        Args:
            rank: Target rank.
            update_gap_scheduler: Instance of UpdateGapScheduler.
            verbose: If True, prints diagnostic messages.
            scale: Scaling factor applied after back projection.
            warm_restart: Continue from the previous projection when updating.
            n_iter_max: Maximum number of iterations for the Tucker decomposition.
            svd_type: Type of SVD to use for Tucker decomposition. Options: "randomized_svd" or "truncated_svd".
        """
        # Then initialize other attributes
        self.rank = rank
        self.verbose = verbose
        self.scale = scale
        self.proj_tensor = None
        self.warm_restart = warm_restart
        self.n_iter_max = n_iter_max
        self.update_gap_scheduler = update_gap_scheduler
        self.num_updates = 0
        self.num_steps = 0
        self._rank_validated = False
        self.svd_type = svd_type 
        
        if self.verbose:
            print(f"TensorGradLowRankProjector initialized with rank={self.rank}, scale={self.scale}, warm_restart={self.warm_restart}, n_iter_max={self.n_iter_max}, svd_type={self.svd_type}")
        
    def should_update_projector(self, iter):
        return self.update_gap_scheduler.should_update(iter)

    def project(self, full_rank_grad, iter):
        with torch.no_grad(), record_function("### TENSOR_GRAD_PROJECT_FORWARD"):
            if self.proj_tensor is None or self.should_update_projector(iter):
                self.proj_tensor = self.get_projection_tensor(full_rank_grad)
                self.num_updates += 1
            if self.proj_tensor[0].device != full_rank_grad.device:
                self.proj_tensor = [f.to(full_rank_grad.device) for f in self.proj_tensor]
            self.num_steps = iter

            return self.transform(self.proj_tensor, full_rank_grad)

    def project_back(self, low_rank_grad, output_buffer=None, alpha=1.0, accumulate=False):
        with torch.no_grad(), record_function("#### TENSOR_GRAD_PROJECT_BACK"):
            # If out is provided, use it as the output buffer
            if output_buffer is not None:
                # Apply inverse transform with the provided buffer
                self.inverse_transform(self.proj_tensor, low_rank_grad, output_buffer=output_buffer, alpha=alpha*self.scale)
                return output_buffer
            else:
                # No buffer provided, let inverse_transform allocate a new tensor
                full_rank_grad = self.inverse_transform(self.proj_tensor, low_rank_grad)
                return full_rank_grad * self.scale
    
    
    # Tucker decomp: higher-order SVD
    def get_projection_tensor(self, weights):
        matrix = weights.data
        original_dtype = matrix.dtype
        
        # Validate rank format if not done yet
        self._validate_rank(self.rank, matrix.shape)
        
        # Always use full precision for tucker decomposition
        if matrix.dtype == torch.complex32:
            matrix = matrix.to(torch.complex64)
            
            
        # Handle initialization with warm restart
        init = None
        if self.warm_restart and self.proj_tensor is not None:
            # Convert factors to full precision temporarily for initialization
            # check if on same device as matrix
            factors = self.proj_tensor
            if factors[0].device != matrix.device:
                factors = [f.to(matrix.device) for f in factors]
            # check if full precision if not convert to full precision - check if complex32
            if torch.is_complex(factors[0]) and factors[0].dtype == torch.complex32:
                factors = [f.to(torch.complex64) for f in factors]
            
            init = factors
        else:
            init = "svd"
            
        try:
            _, factors = tucker(matrix, rank=self.rank, init=init, n_iter_max=self.n_iter_max, svd=self.svd_type)
            torch.cuda.empty_cache()
        except Exception as e:
            if self.verbose:
                print(f"Tucker decomposition failed with warm start, trying again with SVD init: {str(e)}")
            # lets try again 
            try:
                matrix = matrix + 1e-8 * torch.randn_like(matrix, dtype=matrix.dtype)  # Add noise for stability
                _, factors = tucker(matrix, rank=self.rank, init="svd", n_iter_max=self.n_iter_max*2, svd='randomized_svd') # try again
            except Exception as e:
                raise e
        torch.cuda.empty_cache()
        
        # Convert factors to half precision if mixed precision is enabled
        factors = [f.to(original_dtype) for f in factors]
        
        return factors
    
    def transform(self, proj_tensor, full_rank_grad):
        with torch.no_grad(), record_function("### TENSOR_GRAD_TRANSFORM"):
            return multi_mode_dot(full_rank_grad, proj_tensor, transpose=True)

    def inverse_transform(self, proj_tensor, x, output_buffer=None, alpha=1.0):
        with torch.no_grad(), record_function("### TENSOR_GRAD_INV_TRANSFORM"):
            if output_buffer is None:
                return multi_mode_dot(x, proj_tensor)
            else:
                result = multi_mode_dot(x, proj_tensor)
                output_buffer.add_(result, alpha=alpha)
                return output_buffer

    def _validate_rank(self, rank, matrix_shape):
        """
        Validates and formats the rank parameter for tensor low-rank projection.
        
        This method handles different rank specification formats and converts them
        into a list of per-dimension ranks appropriate for the tensor shape:
        
        1. Float (0 < rank < 1): Interpreted as a memory budget percentage.
           - Distributes the budget evenly across non-singleton dimensions
           - For example, rank=0.25 means using 25% of the original tensor size
        
        2. Integer (rank > 0): Applied as a fixed rank to each dimension
           - Automatically capped at each dimension's size
        
        3. List of values: Provides per-dimension control
           - List of floats (0-1): Each value is the percentage of that dimension
           - List of integers: Direct specification of rank for each dimension
           - Must match the number of tensor dimensions
        
        Parameters:
        -----------
        rank : float, int, or list
            The rank specification to validate and format
        matrix_shape : tuple or list
            The shape of the tensor to be decomposed
            
        Returns:
        --------
        None, but sets self.rank as a list of integers representing 
        the validated rank for each dimension
        
        Raises:
        -------
        ValueError: If rank specification is invalid or incompatible with the tensor shape
        """
        if self._rank_validated:
            return
        if self.verbose:
            print(f"Validating rank: {rank}")

        if isinstance(rank, float):
            if not (0 < rank < 1):
                raise ValueError(f"Float rank must be between 0 and 1, got {rank}")
            # find out which rank will give = rank percentage in total based on matrix_shape
            # count how many non 1 dimensions there are
            non_one_dims = sum(1 for d in matrix_shape if d != 1)
            # even_distributed_rank = rank ** (1/non_one_dims)
            # Handle dimensions of size 1 separately
            self.rank = []
            for d in matrix_shape:
                self.rank.append(max(1, int(rank * d)))

        elif isinstance(rank, int):
            if rank < 1:
                raise ValueError(f"Integer rank must be positive, got {rank}")
            self.rank = [min(rank, d) for d in matrix_shape]
        elif isinstance(rank, list):
            # Check if list contains either all floats between 0-1 or all positive integers
            all_valid_floats = all(isinstance(r, float) and 0 < r < 1 for r in rank)
            all_valid_ints = all(isinstance(r, int) and r > 0 for r in rank)
            
            if not (all_valid_floats or all_valid_ints):
                raise ValueError("All ranks in list must be either floats between 0 and 1 or positive integers")
            
            # If list is longer than matrix_shape, just use the first len(matrix_shape) elements
            if len(rank) > len(matrix_shape):
                rank = rank[:len(matrix_shape)]
                if self.verbose:
                    print(f"Rank list longer than tensor dimensions, using first {len(matrix_shape)} elements: {rank}")
            elif len(rank) < len(matrix_shape):
                raise ValueError(f"Rank list length {len(rank)} is shorter than tensor dimensions {len(matrix_shape)}")
            
            if all_valid_floats:
                # For floats, compute the rank as a percentage of dimension size
                self.rank = [max(1, int(r * d)) for r, d in zip(rank, matrix_shape)]
            else:
                # For integers, just cap at the dimension size
                self.rank = [min(r, d) for r, d in zip(rank, matrix_shape)]
        else:
            raise ValueError(f"Unsupported rank format: {rank}. Must be float, positive int, or list of ints.")
        
        if self.verbose:
            print(f"Validated rank: {self.rank}")
            print(f"Matrix shape: {matrix_shape}")
        self._rank_validated = True
        # what is the total percentage of params in rank vs matrix_shape?
        self.rank_percentage = sum(self.rank) / sum(matrix_shape)