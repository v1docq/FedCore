import math 

import torch

from tdecomp.grad_proj.tensorgrad.tensorgrad import TensorGRaD


SMALLNESS_THRESHOLD = 1000


def is_small(p: torch.Tensor) -> float:
    n = p.ndim
    return n <= 2 or p.numel() < SMALLNESS_THRESHOLD


def setup_optimizer_and_scheduler(config, model: torch.nn.Module, logging_name):
    """Set up optimizer based on whether galore is enabled"""
    tensorgrad_params = [
        p for p in model.parameters() if not is_small(p)
    ]     
    # First parameter is usually the lifting layer, we don't want to apply tensorgrad to it
    if len(tensorgrad_params) > 0:
        tensorgrad_params.pop(0)
    ndim2group = {}
    for i in range(2, 6):
        ndim2group[i] = [
        p for p in model.parameters() if p.ndim == i and p.numel() > SMALLNESS_THRESHOLD
        ]
    ndim2id = {
        i: {id(p) for p in ps} for i, ps in ndim2group.items()
    }

    regular_parameters = []
    for p in model.parameters():
        idp = id(p)
        registered = any(idp in group for group in ndim2id.values())
        if not registered:
            regular_parameters.append(p)

    del ndim2id

    tg_gl_groups = {
        i: {
        'params': ndim2group[i],
            'type': "tucker" if i > 2 else 'galore',
            'rank': config.opt.rank,
            'dim': i,
            'optimizer_type': config.opt.optimizer_type,
            'scale': config.opt.scale,
            'proj_type': config.opt.proj_type,
            'galore_2d_proj_type': config.opt.galore_2d_proj_type,
            'sparse_ratio': config.opt.sparse_ratio,
            'sparse_type': config.opt.sparse_type,
            'scale_by_mask_ratio': config.opt.scale_by_mask_ratio,
            'reset_sparse_optimizer_states': config.opt.reset_sparse_optimizer_states,
            'enforce_full_complex_precision': getattr(config.opt, 'enforce_full_complex_precision', False),
            'svd_type': getattr(config.opt, 'svd_type', 'truncated_svd'),
            
            # Second projector parameters
            'second_proj_type': config.opt.second_proj_type,
            'second_sparse_ratio': config.opt.second_sparse_ratio,
            'second_sparse_type': config.opt.second_sparse_type,
            'second_scale': config.opt.second_scale,
            'second_rank': config.opt.second_rank,
            'second_scale_by_mask_ratio': config.opt.second_scale_by_mask_ratio,
            
            # Scheduler update gap parameters
            'update_proj_gap': config.opt.update_proj_gap,
            'update_proj_gap_end': config.opt.update_proj_gap_end,
            'update_proj_gap_mode': config.opt.update_proj_gap_mode,
            'batch_size': config.data.batch_size,
            'epochs': config.opt.n_epochs,
            'scheduler_T_max': config.opt.scheduler_T_max,
            'training_samples': config.data.n_train,
            
            # Tucker parameters
            'n_iter_max_tucker': config.opt.n_iter_max_tucker,
            'tucker_warm_restart': config.opt.tucker_warm_restart,
            
            # 'log_ranks_interval': config.wandb.log_ranks_interval,
            'lambda_sparse': config.opt.tensorgrad_sum_lambda_sparse,
        } for i, g in ndim2group.items() if len(g)
    }
        
    param_groups = [
        {'params': regular_parameters},
        *tg_gl_groups.values()
    ]

    # Common optimizer arguments
    optimizer_args = {
            'lr': config.opt.learning_rate,
            'matrix_only': config.opt.naive_galore,
            'support_complex': config.opt.adamw_support_complex,
            'run_name': logging_name,
            'enforce_full_complex_precision': config.opt.enforce_full_complex_precision,
    }

    optimizer_args['use_sum'] = (config.opt.optimizer_type == "tensorgrad_sum")
    optimizer = TensorGRaD(param_groups, **optimizer_args)
    
    # Set up scheduler for non-per-layer optimization
    scheduler = get_scheduler(
        scheduler_name=config.opt.scheduler,
        optimizer=optimizer,
        gamma=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        T_max=config.opt.scheduler_T_max,
        step_size=config.opt.step_size
    )
    
    return optimizer, scheduler


def get_scheduler(scheduler_name: str,
                  optimizer: torch.optim.Optimizer,
                  gamma: float,
                  patience: int,
                  T_max: int,
                  step_size: int,):
    '''
    Returns LR scheduler of choice from available options
    '''
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=gamma,
            patience=patience,
            mode="min",
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name == "constant":
        def constant_scheduler(step):
            return 1.0
        return constant_scheduler
    elif scheduler_name == "step":
        def step_scheduler(step):
            return gamma ** (step // step_size)
        return step_scheduler
    elif scheduler_name == "exponential":
        def exp_scheduler(step):
            return gamma ** step
        return exp_scheduler
    elif scheduler_name == "cosine":
        def cosine_scheduler(step):
            return 0.5 * (1 + math.cos(math.pi * step / T_max))
        return cosine_scheduler
    else:
        raise ValueError(f"Got scheduler={scheduler_name}")

    return scheduler