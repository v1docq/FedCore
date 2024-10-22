import torch
from torch.linalg import matrix_norm, vector_norm
from torch.nn.modules import Module
from torch import Tensor


class SVDLoss(Module):
    """Base class for singular value decomposition losses.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``1``).
    """

    def __init__(self, factor: float = 1.0) -> None:
        super().__init__()
        self.factor = factor


class OrthogonalLoss(SVDLoss):
    """Orthogonality regularizer for unitary matrices obtained by SVD decomposition.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``1``).
    """

    def __init__(self, factor: float = 1.0) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module) -> Tensor:
        """Calculates orthogonality loss.

        Args:
            model: Optimizable module containing SVD decomposed layers.
        """
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split(".")[-1] == "U":
                if len(parameter.size()) < 3:
                    U = parameter
                else:
                    n, c, w, h = parameter.size()
                    decompose_shape = (n, c * w * h)
                    U = parameter.reshape(decompose_shape)
                n += 1
                r = U.size()[1]
                E = torch.eye(r, device=U.device)
                loss += matrix_norm(U.transpose(0, 1) @ U - E) ** 2 / r

            elif name.split(".")[-1] == "Vh":
                if len(parameter.size()) < 3:
                    Vh = parameter
                else:
                    n, c, w, h = parameter.size()
                    decompose_shape = (n, c * w * h)
                    Vh = parameter.reshape(decompose_shape)
                r = Vh.size()[0]
                E = torch.eye(r, device=Vh.device)
                loss += matrix_norm(Vh @ Vh.transpose(0, 1) - E) ** 2 / r
        return self.factor * loss / n


class HoyerLoss(SVDLoss):
    """Hoyer regularizer for matrix with singular values obtained by SVD decomposition.

    Args:
    factor: The hyperparameter by which the calculated loss function is multiplied
        (default: ``1``).
    """

    def __init__(self, factor: float = 1.0) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module) -> Tensor:
        """Calculates Hoyer loss.

        Args:
            model: Optimizable module containing SVD decomposed layers.
        """
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split(".")[-1] == "S":
                n += 1
                S = parameter
                loss += vector_norm(S, ord=1) / vector_norm(S, ord=2)
        return self.factor * loss / n
