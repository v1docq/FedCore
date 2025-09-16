import math
import warnings
from typing import Callable, Iterable, Tuple, Union, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.autograd.profiler import record_function
import numpy as np

# Import your utility function that returns projectors.
from .projectors.projector_utils import get_projector

class TensorGRaD(Optimizer):
    """
    Implements AdamW with an optional composite projection branch.
    
    For each parameter:
      - If the parameter group's dictionary contains a "rank" key, then the optimizer
        uses two projectors (a sparse projector and a low-rank projector) as returned by
        `get_projector`. In that case, the optimizer:
          1. If use_sum=True:
             - Applies both projectors directly to the gradient
             - Updates separate Adam momentum buffers for each branch
             - Applies the inverse projections and sums the updates
          2. If use_sum=False:
             - Applies the first projector to the gradient
             - Computes residual by subtracting the first projection
             - Applies the second projector to the residual
             - Updates separate Adam momentum buffers for each branch
             - Applies the inverse projections and sums the updates
      - Otherwise, the parameter is updated with standard AdamW.
      
    The complex gradients are handled as in the original AdamW implementation.
    
    If enforce_full_complex_precision is True, then the optimizer will use full complex precision for states.
    
    Initialization parameters (lr, betas, eps, weight_decay, correct_bias, etc.) are provided
    in the usual way; additional flags such as "matrix_only", and
    projection-specific hyperparameters (update_proj_gap, scale, sparse_ratio, sparse_type, rank,
    proj_type, support_complex, etc.) are passed via the parameter group.
    """
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        matrix_only: bool = True,
        support_complex: bool = False,
        use_sum: bool = False,  # New parameter to control projection approach
        enforce_full_complex_precision: bool = False,  # New parameter for complex precision control
        run_name: str = None,
        verbose=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias
        }
        super().__init__(params, defaults)
        self.matrix_only = matrix_only
        self.enforce_full_complex_precision = enforce_full_complex_precision
        self.support_complex = support_complex
        self.id_counter = 0
        self.use_sum = use_sum
        self.run_name = run_name
        self.reset_sparse_optim_state = True
        self.verbose = verbose

        if self.verbose:
            if enforce_full_complex_precision:
                print("### Using TensorGRaD with full complex precision for states ###")
            else:
                print('#### Running with TensorGRaD ####')

    def _adam_update(self, grad, exp_avg, exp_avg_sq, beta1, beta2, eps, step):
        """Helper method to compute Adam update for a single gradient.
        
        Args:
            grad: The gradient tensor
            exp_avg: First moment estimate
            exp_avg_sq: Second moment estimate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            eps: Small constant for numerical stability
            step: Step count for bias correction
            
        Returns:
            tuple: (normalized gradient, step size factor)
        """
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # Handle ComplexHalf differently as it doesn't support addcmul_
        if grad.dtype == torch.complex32:
            # Use basic operations instead of addcmul_ for complex32
            exp_avg_sq.mul_(beta2)
            if torch.is_complex(grad):
                exp_avg_sq.add_((grad * grad.conj()) * (1.0 - beta2))
            else:
                exp_avg_sq.add_((grad * grad) * (1.0 - beta2))
        else:
            # Original logic for other dtypes
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj() if torch.is_complex(grad) else grad, value=1 - beta2)
            
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        denom = exp_avg_sq.sqrt().add_(eps)
        step_size = math.sqrt(bias_correction2) / bias_correction1
        return exp_avg / denom, step_size

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                grad_is_complex = torch.is_complex(grad)
                grad_dtype = grad.dtype

                # Initialize step counter and assign a unique id if needed.
                if "step" not in state:
                    state["step"] = 0
                    state["id"] = self.id_counter
                    state["total_params"] = p.numel()
                    self.id_counter += 1
                # Set default dimension if not provided.
                if "dim" not in group:
                    group["dim"] = 2
                    
                # Convert grad to full precision if needed
                if self.enforce_full_complex_precision and grad.dtype == torch.complex32:
                    grad = grad.to(torch.cfloat)

                # --- Composite Projection Branch ---
                if "rank" in group:
                    if "first_proj" not in state or "second_proj" not in state:
                        # These parameters should already be in the group from train_ns_repro_tensorgrad.py
                        first_proj, second_proj = get_projector(
                            group,
                            matrix_only=self.matrix_only,
                            support_complex=self.support_complex,
                        )
                        state["first_proj"] = first_proj
                        state["second_proj"] = second_proj
                        
                        # Identify which projector is sparse vs low-rank
                        # Sparse projector has sparse_ratio attribute, low-rank has rank attribute
                        if hasattr(first_proj, "sparse_ratio"):
                            state["sparse_is_first"] = True
                        else:
                            state["sparse_is_first"] = False
                        if self.verbose:
                            print(f"First projector is sparse: {state['sparse_is_first']} and second is low-rank: {not state['sparse_is_first']}")
                    
                    if grad.ndim == 5: # if complex tensor is stored as 2 real tensors
                        grad = torch.view_as_complex(grad)

                    with record_function("#### GRAD FORWARD PROJ ####"):                        
                        first_proj = state["first_proj"]
                        second_proj = state["second_proj"]
                        
                        if self.use_sum:
                            # Sum approach: apply both projections directly to the gradient
                            first_grad = first_proj.project(grad, state["step"])
                            second_grad = second_proj.project(grad, state["step"])
                        else:
                            # 1) project the gradient for the first branch
                            first_grad = state["first_proj"].project(grad, state["step"])

                            # 2) in‐place turn `grad` into the residual 
                            from torch.ao.quantization.utils import _normalize_kwargs
                            kwargs = _normalize_kwargs(state["first_proj"].project_back,
                                dict(output_buffer=grad, alpha=-1.0, accumulate=True)
                            )
                            state["first_proj"].project_back(
                                first_grad, **kwargs
                            )
                            # 3) now project that residual for the second branch
                            second_grad = state["second_proj"].project(grad, state["step"])
                        grad = None
                        torch.cuda.empty_cache()

                # --- Initialize Adam State Buffers ---
                if "rank" in group:
                    # Composite branch: separate buffers for first and second parts
                    dtype = torch.cfloat if self.enforce_full_complex_precision and grad_is_complex else grad_dtype                        
                    if "first_exp_avg" not in state or "second_exp_avg" not in state:
                        state["first_exp_avg"] = torch.zeros_like(first_grad, dtype=dtype)
                        state["first_exp_avg_sq"] = torch.zeros_like(first_grad, dtype=dtype)
                        state["second_exp_avg"] = torch.zeros_like(second_grad, dtype=dtype)
                        state["second_exp_avg_sq"] = torch.zeros_like(second_grad, dtype=dtype)
                        # total param in exp_avg vs the the the shape
                        if state["sparse_is_first"]:
                            state["sparse_ratio"] =  first_grad.numel() / state["total_params"]
                            state["rank"] = second_grad.numel() / state["total_params"]
                        else:
                            state["sparse_ratio"] = second_grad.numel() / state["total_params"]
                            state["rank"] = first_grad.numel() / state["total_params"]
                        if self.verbose:
                            print(f"Total params: {state['total_params']}")
                            print(f"Sparse ratio: {state['sparse_ratio']} and rank: {state['rank']}")
                    else:
                        if group.get('reset_sparse_optimizer_states', False):
                            # check if projector has attribute should_update and is True
                            if state["sparse_is_first"] and hasattr(state["first_proj"], "should_update") and state["first_proj"].should_update:
                                state["first_exp_avg"] = torch.zeros_like(first_grad, dtype=dtype)
                                state["first_exp_avg_sq"] = torch.zeros_like(first_grad, dtype=dtype)
                            elif not state["sparse_is_first"] and hasattr(state["second_proj"], "should_update") and state["second_proj"].should_update:
                                state["second_exp_avg"] = torch.zeros_like(second_grad, dtype=dtype)
                                state["second_exp_avg_sq"] = torch.zeros_like(second_grad, dtype=dtype)
                            
                else:
                    # Standard Adam buffers.
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(grad, dtype=grad.dtype)
                        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=grad.dtype)

                beta1, beta2 = group["betas"]
                state["step"] += 1

                # --- Update Moment Estimates ---
                if "rank" in group:
                    # Composite branch: use the helper method for both projections
                    norm_first, step_size = self._adam_update(
                        first_grad, state["first_exp_avg"], state["first_exp_avg_sq"], 
                        beta1, beta2, group["eps"], state["step"]
                    )
                    norm_second, _ = self._adam_update(
                        second_grad, state["second_exp_avg"], state["second_exp_avg_sq"], 
                        beta1, beta2, group["eps"], state["step"]
                    )
                        
                    # Back-project the momentum estimates using a single accumulator
                    with record_function("#### GRAD BACKWARD PROJ ####"):
                        # Compute lambda_sparse only once and store in state
                        if "lambda_sparse" not in state:
                            # First check if lambda_sparse is provided in group or defaults
                            lambda_sparse = group.get("lambda_sparse", None)
                            if lambda_sparse is None or lambda_sparse <= 0:                                
                                # Default: DoF ratio (sparse_ratio / rank_frac)
                                lambda_sparse = state["sparse_ratio"] / (state["rank"] + 1e-12)  # Add eps to avoid div by 0
                            
                            state["lambda_sparse"] = lambda_sparse
                            if self.verbose:
                                print(f"Initialized lambda_sparse={lambda_sparse:.4f} for parameter shape {p.shape} ({'provided' if group.get('lambda_sparse', None) is not None else 'computed'})")

                        if state["sparse_is_first"]:
                            # First back-project the part that doesn't need scaling
                            combined = state["second_proj"].project_back(norm_second)
                                
                            # Add the scaled sparse part to the same buffer
                            state["first_proj"].project_back(norm_first, 
                                                            output_buffer=combined,
                                                            alpha=state["lambda_sparse"],
                                                            accumulate=True)
                        else:
                            # First back-project the part that doesn't need scaling
                            combined = state["first_proj"].project_back(norm_first)
                            # Add the scaled sparse part to the same buffer
                            state["second_proj"].project_back(norm_second,
                                                            output_buffer=combined,
                                                            alpha=state["lambda_sparse"],
                                                            accumulate=True)
                        
                        # Handle special case when lambda_sparse is 5.5
                        if group.get("lambda_sparse", None) == 5.5:
                            combined.mul_(0.5)
                        
                        norm_grad = combined
                        del combined
                        torch.cuda.empty_cache()
                else:
                    # Standard Adam update: use the helper method
                    norm_grad, step_size = self._adam_update(
                        grad, state["exp_avg"], state["exp_avg_sq"], 
                        beta1, beta2, group["eps"], state["step"]
                    )
                    

                # Convert norm_grad back to parameter's dtype if needed
                if self.enforce_full_complex_precision and p.dtype == torch.complex32:
                    norm_grad = norm_grad.to(torch.complex32)

                # --- Apply Update ---
                step_size = -group["lr"] * step_size if group.get("correct_bias", True) else -group["lr"]
                torch.mul(norm_grad, step_size, out=norm_grad)
                p.add_(norm_grad)
                
                del norm_grad  # Free memory immediately
                torch.cuda.empty_cache()
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        
        return loss
    