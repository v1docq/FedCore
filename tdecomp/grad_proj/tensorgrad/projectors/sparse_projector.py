import torch
from torch.utils.checkpoint import checkpoint

def optional_checkpoint_matmul(a, b, activation_checkpoint=False):
    if activation_checkpoint:
        return checkpoint(torch.matmul, a, b)
    else:
        return torch.matmul(a, b)

class GaLoreSparseProjector:
    """
    A sparse version of GaLore that uses row/column sampling
    instead of low-rank SVD. It follows the same signature
    and 'proj_type' logic as GaLoreProjector.

    proj_type can be:
        - 'right'        (always columns)
        - 'left'         (always rows)
        - 'full'         (rows AND columns)

    Additional parameters:
        sparse_ratio: fraction of rows (or columns) to keep
        sparse_type:  e.g. 'topk', 'randK', 'probability'
    """
    def __init__(
        self,
        sparse_ratio: float = 0.25,
        sparse_type: str = "topk",
        verbose: bool = False,
        update_gap_scheduler = None,
        scale: float = 1.0,
        proj_type: str = 'std',
        activation_checkpoint: bool = False,
    ):
        self.sparse_ratio = sparse_ratio
        self.sparse_type = sparse_type
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.scale = scale
        self.proj_type = proj_type
        self.activation_checkpoint = activation_checkpoint
        self._mask = None

        # Remember shape for project_back
        self._orig_shape = None
        # Keep track of iteration for update
        self._last_iter = -1

    def project(self, full_rank_grad: torch.Tensor, iteration: int):
        """
        Equivalent to GaLoreProjector.project, but uses row/column sampling
        to produce a smaller tensor from 'full_rank_grad'.
        """
        # store shape for reconstruction
        if self._orig_shape is None:
            self._orig_shape = full_rank_grad.shape

        # For 'std' logic, we decide whether to sample columns or rows
        # based on shape. If #rows >= #cols => columns, else rows.
        grad_2d = full_rank_grad  # we assume 2D here

        # Check if we need to update the mask(s)
        # (only do so every update_proj_gap steps)
        if self.update_gap_scheduler.should_update(iteration) or self._mask is None:
            self._update_masks(grad_2d)

        # Now apply the correct sub-selection
        if self.proj_type == 'std':
            if grad_2d.shape[0] >= grad_2d.shape[1]:
                # right: sample columns
                mask = self._mask
                return grad_2d[:, mask]
            else:
                # left: sample rows
                mask = self._mask
                return grad_2d[mask, :]

        elif self.proj_type == 'reverse_std':
            if grad_2d.shape[0] >= grad_2d.shape[1]:
                # left: sample rows
                mask = self._mask
                return grad_2d[mask, :]
            else:
                # right: sample columns
                mask = self._mask
                return grad_2d[:, mask]

        elif self.proj_type == 'right':
            # always columns
            mask = self._mask
            return grad_2d[:, mask]

        elif self.proj_type == 'left':
            # always rows
            mask = self._mask
            return grad_2d[mask, :]

        else:
            raise ValueError(f"Unknown proj_type={self.proj_type}")

    def project_back(self, low_rank_grad: torch.Tensor):
        """
        Re-inject the smaller tensor into the original shape, placing zeros
        in the unselected positions. Then multiply by scale.
        """
        full = torch.zeros(self._orig_shape, dtype=low_rank_grad.dtype, device=low_rank_grad.device)

        grad_2d = full  # we assume 2D
        if self.proj_type == 'std':
            if grad_2d.shape[0] >= grad_2d.shape[1]:
                # we used columns
                mask = self._mask
                grad_2d[:, mask] = low_rank_grad
            else:
                # we used rows
                mask = self._mask
                grad_2d[mask, :] = low_rank_grad

        elif self.proj_type == 'reverse_std':
            if grad_2d.shape[0] >= grad_2d.shape[1]:
                # we used rows
                mask = self._mask
                grad_2d[mask, :] = low_rank_grad
            else:
                # columns
                mask = self._mask
                grad_2d[:, mask] = low_rank_grad

        elif self.proj_type == 'right':
            # columns
            mask = self._mask
            grad_2d[:, mask] = low_rank_grad

        elif self.proj_type == 'left':
            # rows
            mask = self._mask
            grad_2d[mask, :] = low_rank_grad

        else:
            raise ValueError(f"Unknown proj_type={self.proj_type}")

        return full * self.scale

    def _update_masks(self, grad_2d):
        self._last_iter += 1
        # We'll produce a single mask if not 'full'
        if self.proj_type in ['std', 'reverse_std', 'left', 'right']:
            # Decide whether to do row or column sampling based on the shape + proj_type
            # We'll just store in self._mask
            do_rows = False
            if self.proj_type == 'left':
                do_rows = True
            elif self.proj_type == 'right':
                do_rows = False
            elif self.proj_type == 'std':
                do_rows = (grad_2d.shape[0] < grad_2d.shape[1])
            elif self.proj_type == 'reverse_std':
                do_rows = (grad_2d.shape[0] >= grad_2d.shape[1])

            self._mask = self._select_mask_1d(grad_2d, do_rows)

        else:
            raise ValueError(f"Unknown proj_type={self.proj_type}")

    def _select_mask_1d(self, grad_2d, do_rows=True):
        """
        Produce a 1D boolean mask along dimension 0 (rows) if do_rows=True,
        or dimension 1 (columns) if do_rows=False.
        """
        if do_rows:
            norms = torch.norm(grad_2d, dim=1)  # shape=(nrows,)
            dim_size = grad_2d.shape[0]
        else:
            norms = torch.norm(grad_2d, dim=0)  # shape=(ncols,)
            dim_size = grad_2d.shape[1]

        k = max(1, int(self.sparse_ratio * dim_size))
        
        # pick indices
        if self.sparse_type.lower() == 'topk':
            _, idxs = torch.topk(norms, k, largest=True)
            mask = torch.zeros_like(norms, dtype=torch.bool)
            mask[idxs] = True
            return mask

        elif self.sparse_type.lower() in ('randk', 'randomk'):
            perm = torch.randperm(dim_size, device=grad_2d.device)
            picked = perm[:k]
            mask = torch.zeros(dim_size, dtype=torch.bool, device=grad_2d.device)
            mask[picked] = True
            return mask

        elif self.sparse_type.lower() == 'probability':
            denom = norms.sum() + 1e-12
            probs = norms / denom
            picked = torch.multinomial(probs, k, replacement=False)
            mask = torch.zeros(dim_size, dtype=torch.bool, device=grad_2d.device)
            mask[picked] = True
            return mask

        else:
            raise ValueError(f"Unknown sparse_type={self.sparse_type}")