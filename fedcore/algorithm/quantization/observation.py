"""Owned source observation shared by catalogue and graph execution."""
import torch
from fedcore.algorithm.low_rank.candidate_storage import clone_preserving_storages
from fedcore.algorithm.low_rank.topology import module_paths


def observe_operator_rows(model,calibration,paths,max_peak_bytes):
    work = clone_preserving_storages(model,max_peak_bytes=max_peak_bytes).eval()
    rows,hooks = {},[]
    for path in paths:
        def hook(_module,inputs,path=path):
            rows.setdefault(path,[]).append(inputs[0].detach().reshape(-1,inputs[0].shape[-1]).clone())
        hooks.append(module_paths(work)[path].register_forward_pre_hook(hook))
    try:
        with torch.no_grad():work(calibration)
    finally:
        for hook in hooks:hook.remove()
    return {path:torch.cat(values) for path,values in rows.items()}
