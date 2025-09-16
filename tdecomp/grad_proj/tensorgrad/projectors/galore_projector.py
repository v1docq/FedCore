from functools import partial

import torch
from tensorly import tenalg
from torch.utils.checkpoint import checkpoint

class GaLoreProjector:
    def __init__(self, rank, verbose=False, svd_type=None, update_gap_scheduler=None, scale=1.0,
                 galore_2d_proj_type='left', activation_checkpoint=False, support_complex=False):
        self.rank = rank
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.scale = scale
        self.ortho_matrix = None
        self.galore_2d_proj_type = galore_2d_proj_type
        self.activation_checkpointing = activation_checkpoint
        self.support_complex = support_complex
        self._recon_buffer = None  # Pre-allocated reconstruction buffer
        self.svd_type = svd_type or partial(torch.linalg.svd, full_matrices=False)
        if verbose:
            print(f"rank={self.rank}, scale={self.scale}, galore_2d_proj_type={self.galore_2d_proj_type},"
                  f" activation_checkpointing={self.activation_checkpointing}, support_complex={self.support_complex}")
            print(f"GaLoreProjector initialized with rank={self.rank}, scale={self.scale}, "
                  f"galore_2d_proj_type={self.galore_2d_proj_type}, "
                  f"activation_checkpointing={self.activation_checkpointing}, support_complex={self.support_complex}")


    def _project_right(self, full_rank_grad):
        low_rank_grad = optional_checkpoint_matmul(full_rank_grad, self.ortho_matrix.t(), self.activation_checkpointing)
        return low_rank_grad

    def _project_left(self, full_rank_grad):
        low_rank_grad = optional_checkpoint_matmul(self.ortho_matrix.t(), full_rank_grad, self.activation_checkpointing)
        return low_rank_grad

    def _project_full(self, full_rank_grad):
        a = optional_checkpoint_matmul(self.ortho_matrix[0].t(), full_rank_grad, self.activation_checkpointing)
        low_rank_grad = optional_checkpoint_matmul(a, self.ortho_matrix[1].t(), self.activation_checkpointing)
        return low_rank_grad

    @torch.no_grad()
    def project(self, full_rank_grad, iter):
        type_ = self.galore_2d_proj_type
        if self.ortho_matrix is None or self.update_gap_scheduler.should_update(iter):
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, 
                                                           type=type_) 
        low_rank_grad = getattr(self, f'_project_{type_}')(full_rank_grad)                              
        return low_rank_grad

    def _project_back_right(self, low_rank_grad):
        if self._recon_buffer is None:
            self._recon_buffer = torch.zeros((low_rank_grad.shape[0], self.ortho_matrix.shape[1]), 
                                                   dtype=low_rank_grad.dtype, device=low_rank_grad.device)
        self._recon_buffer.zero_()
        torch.matmul(low_rank_grad, self.ortho_matrix, out=self._recon_buffer)
        return self._recon_buffer * self.scale

    def _project_back_left(self, low_rank_grad):
        if self._recon_buffer is None:
            self._recon_buffer = torch.zeros((self.ortho_matrix.shape[0], low_rank_grad.shape[1]), 
                                                   dtype=low_rank_grad.dtype, device=low_rank_grad.device)
        self._recon_buffer.zero_()
        torch.matmul(self.ortho_matrix, low_rank_grad, out=self._recon_buffer)
        return self._recon_buffer * self.scale
    

    def _project_back_full(self, low_rank_grad):
        if self._recon_buffer is None:
            self._recon_buffer = torch.zeros((self.ortho_matrix[0].shape[0], self.ortho_matrix[1].shape[1]), 
                                                   dtype=low_rank_grad.dtype, device=low_rank_grad.device)
        self._recon_buffer.zero_()
        intermediate = torch.matmul(self.ortho_matrix[0], low_rank_grad)
        torch.matmul(intermediate, self.ortho_matrix[1], out=self._recon_buffer)
        return self._recon_buffer * self.scale

    @torch.no_grad()
    def project_back(self, low_rank_grad):
        return getattr(self, f'_project_back_{self.galore_2d_proj_type}')(low_rank_grad)
    
    def get_orthogonal_matrix(self, weights, rank, type):
        with torch.no_grad():
            module_params = weights
            if torch.is_complex(module_params.data) and self.support_complex:
                float_data = False
                original_type = module_params.data.dtype
                original_device = module_params.data.device
                matrix = module_params.data.cfloat()
            elif module_params.data.dtype != torch.float:
                float_data = False
                original_type = module_params.data.dtype
                original_device = module_params.data.device
                matrix = module_params.data.float()
            else:
                float_data = True
                matrix = module_params.data

            full_n_params = matrix.shape[0] * matrix.shape[1]
            if isinstance(rank, float):
                low_rank_params = int(rank * full_n_params)
                int_rank = int(low_rank_params / matrix.shape[0])
            else:
                int_rank = rank

            #make the smaller matrix always to be orthogonal matrix
            if type == 'right':
                _, _, Vh = self.svd_type(matrix)
                B = Vh[:int_rank, :]
                if not float_data:
                    B = B.to(original_device).type(original_type)
                return B
            elif type == 'left':
                U, _, _ = self.svd_type(matrix)
                A = U[:, :int_rank]
                if not float_data:
                    A = A.to(original_device).type(original_type)
                return A
            elif type == 'full':
                U, _, Vh = self.svd_type(matrix)
                A = U[:, :rank]
                B = Vh[:rank, :]
                if not float_data:
                    A = A.to(original_device).type(original_type)
                    B = B.to(original_device).type(original_type)
                return [A, B]
            else:
                raise ValueError('type should be left, right or full')

def optional_checkpoint_matmul(a: torch.Tensor, b: torch.Tensor, activation_checkpoint=True):
    """optional_checkpoint_matmul performs torch.matmul and optionally performs
    activation checkpointing. Removed from code to modularize and remove redundant lines

    Parameters
    ----------
    a : torch.Tensor
        input 1 to matmul
    b : torch.Tensor
        input 2 to matmul
    computes torch.matmul(a,b)
    checkpoint : bool, optional
        whether to perform activation checkpointing, by default True
    """
    if activation_checkpoint:
        return checkpoint(torch.matmul, a, b)
    else:
        return torch.matmul(a, b)
