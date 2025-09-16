import os

from torch.ao.quantization.utils import _normalize_kwargs
from typing import *

from tdecomp.grad_proj.tensorgrad.config import TensorGRaDConfig, DataConfig, OptimizerConfig
from tdecomp.grad_proj.tensorgrad.setup_optimizer import setup_optimizer_and_scheduler
from tdecomp.grad_proj.tensorgrad.tensorgrad import TensorGRaD

os.environ['WANDB_MODE'] = 'offline'

class ParallelTG(TensorGRaD):
    _defaults = dict(
        proj_type='low_rank', 
        galore_2d_proj_type='left', 
        second_proj_type='unstructured_sparse'
    )

    def __new__(cls, model, svd_type, rank: Union[int, float, Tuple[int]], *, scheduler='StepLR', learning_rate=1e-4, **kwargs):
        if not isinstance(rank, (tuple, list)):
            rank = (rank,)
        parameters = cls._defaults | dict(
            svd_type=svd_type, 
            learning_rate=learning_rate,
            scheduler=scheduler,
            rank=rank[0],
            optimizer_type='tensorgrad_sum',
            second_rank = rank[-1]
        ) | kwargs
        config = TensorGRaDConfig(
            DataConfig(**_normalize_kwargs(DataConfig.__init__, parameters)),
            OptimizerConfig(**_normalize_kwargs(OptimizerConfig.__init__, parameters))
        )
        opt, sch = setup_optimizer_and_scheduler(config, model, None)
        return opt, sch
    

class ULTG(TensorGRaD):
    _defaults = dict(
        proj_type='unstructured_sparse', 
        galore_2d_proj_type='left', 
        second_proj_type='low_rank'
    )

    def __new__(cls, model, svd_type, rank: Union[int, float, Tuple[int]], *, 
                scheduler='StepLR', learning_rate=1e-4, **kwargs):
        if not isinstance(rank, (tuple, list)):
            rank = (rank,)
        parameters = cls._defaults | dict(
            svd_type=svd_type, 
            learning_rate=learning_rate,
            scheduler=scheduler,
            rank=rank[0],
            optimizer_type='tensorgrad',
            second_rank = rank[-1]
        ) | kwargs
        config = TensorGRaDConfig(
            DataConfig(**_normalize_kwargs(DataConfig.__init__, parameters)),
            OptimizerConfig(**_normalize_kwargs(OptimizerConfig.__init__, parameters))
        )
        opt, sch = setup_optimizer_and_scheduler(config, model, None)
        return opt, sch
