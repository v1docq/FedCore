"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from typing import Union
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer. https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
    """

    def __init__(self, model: nn.Module, optimizer: nn.Module = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state, opt_state = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state,
            "optimizer": opt_state
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"]
        )

def load_training_state(save_dir: Union[str, Path], 
                        save_name: str,
                        model: nn.Module,
                        optimizer: nn.Module = None,
                        scheduler: nn.Module = None,
                        regularizer: nn.Module = None,
                        map_location: dict = None,
                        distributed: bool = False) -> dict:
    """
    Load training state saved using either `torch.save` or `dcp.async_save`.
    Supports resuming from both distributed and non-distributed formats.

    Parameters
    ----------
    save_dir : Union[str, Path]
        Directory to load training state from.
    save_name : str
        Name of the model checkpoint to load.
    model : nn.Module
        Model to load state into.
    optimizer : nn.Module, optional
        Optimizer to load state into, by default None.
    scheduler : nn.Module, optional
        Scheduler to load state into, by default None.
    regularizer : nn.Module, optional
        Regularizer to load state into, by default None.
    map_location : dict, optional
        Mapping for device placement, by default None.
    distributed : bool, optional
        If true, wraps the model with DistributedDataParallel, by default False.

    Returns
    -------
    dict
        loaded model, optimizer, scheduler, regularizer, and epoch.
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    if not map_location and dist.is_initialized():
        map_location = {"cuda:0": f"cuda:{dist.get_rank()}"}

    torch.cuda.empty_cache()

    # Define paths for distributed and non-distributed formats
    save_pth = save_dir / f"{save_name}_state_dict.pt"
    ckpt_id = save_dir / f"{save_name}_dcp_async"
    torch_checkpoint_path = save_dir / f"{save_name}_torch_checkpoint.pth"

    # Check which format to load
    if save_pth.exists():
        print(f"Loading non-distributed checkpoint from {save_pth}...")
        checkpoint = torch.load(save_pth, map_location=map_location)
    elif torch_checkpoint_path.exists():
        print(f"Loading converted Torch checkpoint from {torch_checkpoint_path}...")
        checkpoint = torch.load(torch_checkpoint_path, map_location=map_location)
    elif ckpt_id.exists(): # NOTE - we experienced problems loading from this path, thus we convert it to torch format
        print(f"No Torch checkpoint found. Converting DCP checkpoint at {ckpt_id} to Torch format...")
        dcp_to_torch_save(str(ckpt_id), str(torch_checkpoint_path))
        checkpoint = torch.load(torch_checkpoint_path, map_location=map_location)
    else:
        raise FileNotFoundError(
            f"No valid checkpoint found in {save_dir}. Expected {save_pth} or {ckpt_id}."
        )

    # Check if `app_state` exists (saved with DCP)
    if "app" in checkpoint:
        app_state = checkpoint["app"]

        # Reconstruct AppState if needed
        if isinstance(app_state, dict):
            print("Reconstructing `AppState` from dictionary...")
            app_state = AppState(
                model=model,
                optimizer=optimizer
            )
            app_state.load_state_dict(checkpoint["app"])

        # Load model and optimizer states from AppState
        model.load_state_dict(app_state.model.state_dict())
        if optimizer and app_state.optimizer is not None:
            optimizer.load_state_dict(app_state.optimizer.state_dict())
        epoch = checkpoint["manifest"].get("epoch", 0)

        # Load scheduler and regularizer from manifest if present
        if scheduler and "scheduler" in checkpoint["manifest"]:
            scheduler.load_state_dict(checkpoint["manifest"]["scheduler"])
        if regularizer and "regularizer" in checkpoint["manifest"]:
            regularizer.load_state_dict(checkpoint["manifest"]["regularizer"])
    else:  # Non-DCP format (torch.save)
        print("Loading checkpoint saved without DCP...")
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint.get("epoch", 0)

        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if regularizer and "regularizer_state" in checkpoint:
            regularizer.load_state_dict(checkpoint["regularizer_state"])

    # Move model to the correct device
    if map_location:
        if isinstance(map_location, dict):  # Handle map_location as a dict
            model.to(next(iter(map_location.values())))
        elif isinstance(map_location, torch.device):  # Handle map_location as a single torch.device
            model.to(map_location)
        else:
            raise TypeError(f"Unsupported map_location type: {type(map_location)}. Must be dict or torch.device.")

    if distributed and dist.is_initialized():
        rank = dist.get_rank()
        model = DDP(model, device_ids=[rank], output_device=rank)
        
    return model, optimizer, scheduler, regularizer, epoch


def save_training_state(
    save_dir: Union[str, Path],
    save_name: str,
    model: nn.Module,
    optimizer: nn.Module = None,
    scheduler: nn.Module = None,
    regularizer: nn.Module = None,
    epoch: int=None,
    ) -> None:

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    local_rank = dist.get_rank() if dist.is_initialized() else 0

    if local_rank == 0:
        app_state = AppState(
            model.module if hasattr(model, "module") else model,
            optimizer
        )

        manifest = {}
        if epoch is not None:
            manifest["epoch"] = epoch
        if scheduler is not None:
            manifest["scheduler"] = scheduler.state_dict()
        if regularizer is not None:
            manifest["regularizer"] = regularizer.state_dict()

        checkpoint_data = {
            "app": app_state,
            "manifest": manifest
        }

        ckpt_id = (save_dir / f"{save_name}_dcp_async").as_posix()
        future = dcp.async_save(checkpoint_data, checkpoint_id=ckpt_id)
        print(f"[Rank {local_rank}] Initiated async checkpoint to {ckpt_id}")
        # Optionally block:
        future.result()