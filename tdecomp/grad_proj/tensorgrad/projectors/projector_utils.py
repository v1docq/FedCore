"""
Projector Utilities for TensorGRaD

This module provides utilities for creating and managing gradient projectors used in TensorGRaD.
The projectors are responsible for transforming gradients into more memory-efficient representations
through various decomposition techniques:

1. Low-rank projectors: Based on Tucker decomposition for tensors or SVD for matrices
2. Structured sparse projectors: Apply sparsity along tensor dimensions 
3. Unstructured sparse projectors: Apply element-wise sparsity across the entire tensor

The module also provides utilities for scheduling when to update projections during training.
"""
from typing import Literal

from ..projectors.galore_projector import GaLoreProjector
from .tensor_lowrank_projector import TensorGradLowRankProjector
from .tensor_sparse_projector import TensorGradSparseProjector
from ..projectors.tensor_unstructured_sparse_projector import TensorGradUnstructuredProjector
import matplotlib.pyplot as plt
import copy


def get_projector(
    group, matrix_only=False, support_complex=False
):
    """
    Factory function to create appropriate projector(s) based on configuration.
    
    This is the main entry point for creating projectors. It examines the optimizer
    configuration and creates either a single projector or a pair of composite projectore.
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        matrix_only (bool, optional): If True, treats tensors as matrices. Defaults to False.
        n_iter_max (int, optional): Maximum iterations for decomposition algorithms. Defaults to 100.
        support_complex (bool, optional): Whether to support complex-valued tensors. Defaults to False.
        
    Returns:
        object or tuple: A single projector instance or a tuple of two projector instances
                         (for composite TensorGRaD configuration)
    """
    # Create update gap scheduler based on configuration
    update_gap_scheduler = UpdateGapScheduler(
        group['update_proj_gap'],
        group['update_proj_gap_end'],
        group['update_proj_gap_mode'],
        group['batch_size'],
        max(group['epochs'], group['scheduler_T_max']),
        group['training_samples']
    )
    
    # Create either composite projectors or a single projector based on optimizer type
    if 'tensorgrad' in group['optimizer_type']:
        return _create_composite_projectors(
            group, update_gap_scheduler, matrix_only, support_complex
        )
    else:
        return _create_single_projector(
            group, update_gap_scheduler, matrix_only, support_complex
        )

def _create_composite_projectors(group, update_gap_scheduler, matrix_only, support_complex):
    """
    Creates and returns a pair of projectors for composite optimization (TensorGRaD).
    
    This function creates two different projectors that are used together in TensorGRaD.
    Typically, one is a sparse projector and one is a low-rank projector, but any
    combination of projector types is supported.
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        update_gap_scheduler (UpdateGapScheduler): Scheduler for projection updates
        matrix_only (bool): If True, treats tensors as matrices
        support_complex (bool): Whether to support complex-valued tensors
        
    Returns:
        tuple: A pair of projector instances (first_projector, second_projector)
    """
    print("### Using Composite Projector Configuration ###")
    
    # Get projector types and their parameters
    proj_types = [group["proj_type"], group["second_proj_type"]]
    prefixes = ["", "second_"]
    
    # Function to get size metric for a projector
    def get_projector_size(proj_type, prefix):
        """Helper function to get the memory size of a projector as a fraction of original."""
        try:
            if proj_type == "low_rank":
                size = float(_get_param(group, "rank", prefix))
            elif proj_type in ["unstructured_sparse", "structured_sparse"]:
                size = float(_get_param(group, "sparse_ratio", prefix))
            else:
                size = float('inf')  # Default case
            return size
        except (ValueError, TypeError) as e:
            print(f"Warning: Error converting size parameter to float for {proj_type}: {e}")
            return float('inf')  # Return infinity as fallback
    
    # Get sizes for both projectors
    sizes = [get_projector_size(pt, prefix) for pt, prefix in zip(proj_types, prefixes)]
    
    # If optimizer_type is tensorgrad_sum, ensure smaller projector is first
    if group.get('optimizer_type', 'adam') == 'tensorgrad_sum':
        try:
            if sizes[1] < sizes[0]:
                proj_types = proj_types[::-1]
                prefixes = prefixes[::-1]
                print("    => Swapping projectors to ensure smaller one is first")
                print(f"    => Sizes after swap: first={sizes[1]}, second={sizes[0]}")
        except Exception as e:
            print(f"Warning: Could not compare projector sizes: {e}")
            print(f"    => Sizes were: first={sizes[0]}, second={sizes[1]}")
    
    projectors = []
    for proj_type, prefix in zip(proj_types, prefixes):
        # Make a copy of the update_gap_scheduler so it is not the same object
        update_gap_scheduler_copy = copy.deepcopy(update_gap_scheduler)
        proj = create_projector(
            proj_type=proj_type,
            group=group,
            update_gap_scheduler=update_gap_scheduler_copy,
            matrix_only=matrix_only,
            support_complex=support_complex,
            prefix=prefix
        )
        projectors.append(proj)
        if not group.get('optimizer_type', 'adam') == 'tensorgrad_sum':
            print(f"    => {'First' if not prefix else 'Second'} projector: {proj_type}")
        else:
            # Opposite of above
            print(f"    => {'Second' if not prefix else 'First'} projector: {proj_type}")
    
    return tuple(projectors)

def _create_single_projector(group, update_gap_scheduler, matrix_only, support_complex):
    """
    Creates and returns a single projector.
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        update_gap_scheduler (UpdateGapScheduler): Scheduler for projection updates
        matrix_only (bool): If True, treats tensors as matrices
        support_complex (bool): Whether to support complex-valued tensors
        
    Returns:
        object: A single projector instance
    """
    print("### Using Single Projector Configuration ###")
    proj_type = group['proj_type']
    projector = create_projector(
        proj_type=proj_type,
        group=group,
        update_gap_scheduler=update_gap_scheduler,
        matrix_only=matrix_only,
        support_complex=support_complex,
        prefix=""
    )
    print(f"    => Using projector: {proj_type}")
    return projector

def create_projector(proj_type, group, update_gap_scheduler, matrix_only, support_complex, prefix=""):
    """
    Creates an individual projector based on the specified type.
    
    This is a factory function that dispatches to the appropriate projector creation
    function based on the requested projection type.
    
    Args:
        proj_type (str): Type of projector to create ('low_rank', 'structured_sparse', 'unstructured_sparse')
        group (dict): Parameter group dictionary containing configuration options
        update_gap_scheduler (UpdateGapScheduler): Scheduler for projection updates
        matrix_only (bool): If True, treats tensors as matrices
        support_complex (bool): Whether to support complex-valued tensors
        prefix (str, optional): Prefix for parameter names in group dict. Defaults to "".
        
    Returns:
        object: A projector instance of the requested type
        
    Raises:
        ValueError: If an unknown projector type is specified
    """
    projector_classes = {
        "unstructured_sparse": _create_unstructured_sparse_projector,
        "structured_sparse": _create_structured_sparse_projector,
        "low_rank": _create_low_rank_projector
    }
    
    if proj_type not in projector_classes:
        raise ValueError(f"Unknown projector type: {proj_type}")
    
    svd_type = group.get('svd_type', 'truncated_svd')
    
    # Call the appropriate creator function with relevant parameters only
    if proj_type == "low_rank":
        return _create_low_rank_projector(
            group=group,
            update_gap_scheduler=update_gap_scheduler,
            matrix_only=matrix_only,
            support_complex=support_complex,
            prefix=prefix,
            svd_type=svd_type
        )
    else:
        # For sparse projectors, don't pass matrix_only, support_complex and svd_type
        return projector_classes[proj_type](
            group=group,
            update_gap_scheduler=update_gap_scheduler,
            prefix=prefix
        )

def _get_param(group, param_name, prefix=""):
    """
    Helper function to get parameters with potential prefix.
    
    This function handles the parameter naming convention for composite projectors,
    where the second projector's parameters have a prefix (e.g., 'second_rank').
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        param_name (str): Base parameter name to look up
        prefix (str, optional): Prefix for parameter names. Defaults to "".
        
    Returns:
        Any: The parameter value from the group dictionary
        
    Note:
        Only certain parameters can have prefixes (like 'rank', 'sparse_ratio', etc.).
        Other parameters are shared between projectors.
    """
    prefixable_params = {
        "proj_type", "sparse_ratio", "sparse_type", "scale", 
        "rank", "scale_by_mask_ratio"
    }
    
    if prefix and param_name in prefixable_params:
        prefixed_key = f"{prefix}{param_name}"
        return group[prefixed_key] if prefixed_key in group else group[param_name]
    return group[param_name]

def _create_unstructured_sparse_projector(group, update_gap_scheduler, prefix):
    """
    Creates an unstructured sparse projector.
    
    Unstructured sparse projectors apply element-wise sparsity to the gradients,
    keeping only the top-k elements by magnitude, random elements, or probabilistically
    sampled elements.
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        update_gap_scheduler (UpdateGapScheduler): Scheduler for projection updates
        prefix (str): Prefix for parameter names
        
    Returns:
        TensorGradUnstructuredProjector: An unstructured sparse projector instance
    """
    return TensorGradUnstructuredProjector(
        sparse_ratio=_get_param(group, "sparse_ratio", prefix),
        sparse_type=_get_param(group, "sparse_type", prefix),
        update_gap_scheduler=update_gap_scheduler,
        scale=_get_param(group, "scale", prefix),
        proj_type=_get_param(group, "proj_type", prefix),
        warm_restart=group.get('tucker_warm_restart', False),
        n_iter_max=group.get('n_iter_max_tucker', 10),
        scale_by_mask_ratio=_get_param(group, "scale_by_mask_ratio", prefix),
    )

def _create_structured_sparse_projector(group, update_gap_scheduler, prefix):
    """
    Creates a structured sparse projector.
    
    Structured sparse projectors apply sparsity along tensor dimensions,
    preserving the tensor structure while reducing memory usage.
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        update_gap_scheduler (UpdateGapScheduler): Scheduler for projection updates
        prefix (str): Prefix for parameter names
        
    Returns:
        TensorGradSparseProjector: A structured sparse projector instance
    """
    return TensorGradSparseProjector(
        update_gap_scheduler=update_gap_scheduler,
        scale=_get_param(group, "scale", prefix),
        sparse_ratio=_get_param(group, "sparse_ratio", prefix),
        sparse_type=_get_param(group, "sparse_type", prefix),
        warm_restart=group.get('tucker_warm_restart', False),
        n_iter_max=group.get('n_iter_max_tucker', 10),
        scale_by_mask_ratio=_get_param(group, "scale_by_mask_ratio", prefix),
    )

def _create_low_rank_projector(group, update_gap_scheduler, matrix_only, support_complex, prefix, svd_type):
    """
    Creates a low-rank projector, either matrix or tensor based on dimensionality.
    
    For matrices (or when matrix_only=True), creates a GaLoreProjector that uses
    matrix factorization. For higher-dimensional tensors, creates a TensorGradLowRankProjector
    that uses Tucker decomposition.
    
    Args:
        group (dict): Parameter group dictionary containing configuration options
        update_gap_scheduler (UpdateGapScheduler): Scheduler for projection updates
        matrix_only (bool): If True, treats tensors as matrices even if higher-dimensional
        support_complex (bool): Whether to support complex-valued tensors
        prefix (str): Prefix for parameter names
        svd_type (str): Type of SVD algorithm to use ("truncated_svd" or "randomized_svd")
        
    Returns:
        Union[GaLoreProjector, TensorGradLowRankProjector]: A low-rank projector instance
    """
    
    if group.get('dim', 4) <= 2 or matrix_only:
        return GaLoreProjector(
            _get_param(group, "rank", prefix),
            update_gap_scheduler=update_gap_scheduler,
            scale=_get_param(group, "scale", prefix),
            galore_2d_proj_type=_get_param(group, "galore_2d_proj_type", prefix),
            support_complex=support_complex,
            svd_type=svd_type,
        )
    else:
        return TensorGradLowRankProjector(
            _get_param(group, "rank", prefix),
            update_gap_scheduler=update_gap_scheduler,
            scale=_get_param(group, "scale", prefix),
            warm_restart=group.get('tucker_warm_restart', False),
            n_iter_max=group.get('n_iter_max_tucker', 10),
            svd_type=svd_type,
        )

class UpdateGapScheduler:
    """
    Scheduler for determining when to update projections during training.
    
    This class manages the frequency of projection updates, which can be fixed or
    change over time according to various schedules (linear, exponential, etc.).
    
    Attributes:
        update_gap (int): Initial update interval (iterations between updates)
        update_gap_end (int): Final update interval (for non-fixed modes)
        mode (str): Scheduling mode ('fixed', 'linear', 'exponential', or 'exponential2')
        batch_size (int): Batch size used in training
        epochs (int): Number of training epochs
        training_samples (int): Number of training samples
        iter_per_epoch (float): Iterations per epoch
        total_iters (int): Total number of iterations in training
        next_update (int): Iteration number for the next scheduled update
    """
    
    def __init__(self, start, end, mode: Literal['fixed', 'linear', 'exponential', 'exponential2'] = "fixed",
                 batch_size=1, epochs=1, training_samples=1, verbose=False):
        """
        Initialize the update gap scheduler.
        
        Args:
            start (int): Initial update interval (iterations between updates)
            end (int): Final update interval (for non-fixed modes)
            mode (str, optional): Scheduling mode. Defaults to "fixed".
                Options:
                - "fixed": Keep constant update interval
                - "linear": Linearly increase interval from start to end
                - "exponential": Exponentially increase interval
                - "exponential2": More aggressive exponential increase
            batch_size (int, optional): Batch size used in training. Defaults to 1.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            training_samples (int, optional): Number of training samples. Defaults to 1.
        """
        self.update_gap = start
        self.update_gap_end = end
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_samples = training_samples
        self.verbose = verbose
        
        # Compute iterations and related values
        self.iter_per_epoch = self.training_samples / self.batch_size
        self.total_iters = int(self.iter_per_epoch * self.epochs)
        
        # Initialize the first update at iteration 0
        self.next_update = 0
        
        # Only print gap end if not fixed
        if self.verbose:
            print(f"Update gap scheduler initialized with {self.update_gap} start, {self.update_gap_end} end, {self.mode} mode"
                  if self.mode != "fixed" else
                f"Update gap scheduler initialized with {self.update_gap} start"
            )
    
    def compute_gap(self, current_iter):
        """
        Compute the next update gap based on current iteration.
        
        The update gap changes over time according to the specified mode.
        
        Args:
            current_iter (int): Current iteration number
            
        Returns:
            float: The computed update gap (iterations until next update)
            
        Raises:
            ValueError: If an unknown scheduler mode is specified
        """
        if self.mode == "fixed":
            return self.update_gap
        elif self.mode == "linear":
            progress = self.next_update / self.total_iters
            return self.update_gap + (self.update_gap_end - self.update_gap) * progress
        elif self.mode == "exponential":
            progress = self.next_update / self.total_iters
            return self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
        elif self.mode == "exponential2":
            # More aggressive exponential growth by squaring the progress
            progress = (self.next_update / self.total_iters) ** 2
            return self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")
    
    def should_update(self, current_iter):
        """
        Check if we should update at the current iteration.
        
        This method is called during training to determine if it's time
        to update the projections.
        
        Args:
            current_iter (int): Current iteration number
            
        Returns:
            bool: True if it's time to update, False otherwise
        """
        if current_iter >= self.next_update:
            current_gap = max(1, int(self.compute_gap(current_iter)))
            self.next_update = current_iter + current_gap
            return True
        return False
            
    def print_update_steps(self):
        """
        Simulate and print the update schedule without affecting the scheduler's state.
        
        This method is useful for debugging and visualizing the update schedule
        before training begins.
        """
        list_of_updates = []
        next_update_sim = 0
        
        # For epoch statistics
        epoch_updates = [[] for _ in range(self.epochs)]
        
        for i in range(self.total_iters):
            if i >= next_update_sim:
                progress = next_update_sim / self.total_iters
                if self.mode == "fixed":
                    current_gap = self.update_gap
                elif self.mode == "linear":
                    current_gap = self.update_gap + (self.update_gap_end - self.update_gap) * progress
                elif self.mode == "exponential":
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                elif self.mode == "exponential2":
                    progress = progress ** 2  # Square the progress for more aggressive growth
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                
                current_gap = max(1, int(current_gap))
                list_of_updates.append((i, current_gap))
                
                # Track updates per epoch
                current_epoch = int(i / self.iter_per_epoch)
                if current_epoch < self.epochs:
                    epoch_updates[current_epoch].append(current_gap)
                
                next_update_sim = i + current_gap
        
        print(f"Update schedule simulation: {list_of_updates}")
        
        # Print epoch statistics
        print("\nEpoch-wise statistics:")
        for epoch, gaps in enumerate(epoch_updates):
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                print(f"Epoch {epoch}: average gap = {avg_gap:.2f} ({len(gaps)} updates)")
            
    def plot_update_schedule(self, save_path=None):
        """
        Plot the update schedule showing intervals over iterations.
        
        This method creates a visualization of how the update interval
        changes over the course of training.
        
        Args:
            save_path (str, optional): If provided, saves the plot to this path.
                                     If None, displays the plot.
        """
        # Simulate the schedule
        iterations = []
        gaps = []
        next_update_sim = 0
        
        for i in range(self.total_iters):
            if i >= next_update_sim:
                progress = next_update_sim / self.total_iters
                if self.mode == "fixed":
                    current_gap = self.update_gap
                elif self.mode == "linear":
                    current_gap = self.update_gap + (self.update_gap_end - self.update_gap) * progress
                elif self.mode == "exponential":
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                elif self.mode == "exponential2":
                    progress = progress ** 2
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                
                current_gap = max(1, int(current_gap))
                iterations.append(i)
                gaps.append(current_gap)
                next_update_sim = i + current_gap

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, gaps, 'b.-', label='Update Interval')
        
        plt.title(f'Update Interval Schedule ({self.mode} mode)')
        plt.xlabel('Iteration')
        plt.ylabel('Update Interval')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Either save or display the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            

