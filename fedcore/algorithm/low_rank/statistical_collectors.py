"""Bounded individual-gradient shell for the CPU Linear FWSVD profile.

The original module is never forwarded or differentiated: a private deepcopy
protects its mixed train/eval modes, parameters, buffers and existing .grad
objects even on failure. Collection uses one example graph at a time and
O(sum target_weight.numel()) float64 accumulation. The caller must budget the
private model copy and forward/backward workspace separately.
"""
from __future__ import annotations

from dataclasses import replace
import copy
from itertools import islice
import random

import numpy as np
import torch
from torch import nn

from .approximation import NumericalDomainViolation, _finite
from .statistics import StatisticsPassport, _passport
from .statistical_profiles import (GradientCollectionMetadata,
                                   fwsvd_statistics, validate_gradient_metadata)


def collect_empirical_gradient_squares(model, target_paths, examples, loss_fn,
                                       passport, metadata, *, max_examples,
                                       max_batches):
    """Collect E[(d loss(example)/d W)**2] using observed labels.

    ``examples`` is an iterable of batches ``(inputs, labels)`` or
    ``(inputs, labels, observation_mask)``. Inputs and labels are CPU tensors
    with matching first dimensions; one slice [i:i+1] is one example.
    A mask is one boolean per batch item, excluding it before forward/backward.
    ``loss_fn(prediction, labels)`` returns an unreduced scalar/tensor; this
    collector applies metadata.reduction (mean or sum) *within each example*.
    Token/element masking belongs to loss_fn and is identified by
    metadata.masking; observation-mask identity belongs to passport.mask_id.

    The result maps each target path to EmpiricalGradientSquares with [out,in]
    mean_square. Exactly one backward traversal per included example is made,
    regardless of target count. Both max_examples and max_batches are hard
    limits; no iterator is materialized and no extra batch is read afterwards.
    Checkpoint/data fingerprints are supplied by the caller in the passport.
    """
    _passport(passport)
    validate_gradient_metadata(metadata)
    if (not isinstance(model, nn.Module) or not callable(loss_fn)
            or not isinstance(target_paths, (tuple, list)) or not target_paths
            or any(not isinstance(path, str) for path in target_paths)
            or len(set(target_paths)) != len(target_paths)):
        raise NumericalDomainViolation("A model, loss callable and unique target paths are required")
    if type(max_examples) is not int or max_examples <= 0 or type(max_batches) is not int or max_batches <= 0:
        raise NumericalDomainViolation("Positive explicit example and batch collection limits required")
    if any(tensor.device.type != "cpu" for tensor in (*model.parameters(), *model.buffers())):
        raise NumericalDomainViolation("The initial empirical gradient collector requires a CPU model")
    # Capture all three CPU RNGs before deepcopy as custom copy methods may use
    # them too. No CUDA state is touched because this profile rejects CUDA.
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        try:
            work = copy.deepcopy(model)
        except Exception as error:
            raise NumericalDomainViolation("A private model copy is required for isolated gradient collection") from error
        work.train(metadata.model_mode == "train")
        targets = []
        for path in target_paths:
            try:
                layer = work.get_submodule(path)
            except (AttributeError, KeyError) as error:
                raise NumericalDomainViolation(f"Unknown Linear target path: {path}") from error
            if type(layer) is not nn.Linear or layer.weight.dtype not in (torch.float32, torch.float64):
                raise NumericalDomainViolation("Initial gradient collection supports standard float32/float64 Linear targets")
            layer.weight.requires_grad_(True)
            targets.append(layer.weight)
        if len({id(parameter) for parameter in targets}) != len(targets):
            raise NumericalDomainViolation("Shared target weights require an explicit coupled collection plan")
        totals = [torch.zeros_like(parameter, dtype=torch.float64) for parameter in targets]
        count = batches = masked = 0
        for batch in islice(iter(examples), max_batches):
            if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                raise NumericalDomainViolation("Each gradient batch is (inputs, observed_labels[, observation_mask])")
            inputs, labels = batch[:2]
            if (not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor)
                    or inputs.device.type != "cpu" or labels.device.type != "cpu"
                    or inputs.ndim == 0 or labels.ndim == 0 or len(inputs) != len(labels)):
                raise NumericalDomainViolation("CPU inputs and observed labels require matching batch axes")
            observation_mask = batch[2] if len(batch) == 3 else None
            if observation_mask is not None and (not isinstance(observation_mask, torch.Tensor)
                    or observation_mask.dtype != torch.bool or observation_mask.shape != (len(inputs),)
                    or observation_mask.device.type != "cpu" or not passport.mask_id):
                raise NumericalDomainViolation("Observation masks require CPU boolean rows and a passport mask_id")
            batches += 1
            for index in range(len(inputs)):
                if observation_mask is not None and not bool(observation_mask[index]):
                    masked += 1
                    continue
                with torch.enable_grad():
                    prediction = work(inputs[index:index + 1].detach())
                    raw_loss = loss_fn(prediction, labels[index:index + 1].detach())
                    if (not isinstance(raw_loss, torch.Tensor) or not raw_loss.numel()
                            or raw_loss.device.type != "cpu" or not bool(torch.isfinite(raw_loss).all())
                            or raw_loss.is_complex() or not raw_loss.requires_grad):
                        raise NumericalDomainViolation("Per-example loss must be finite, real and differentiable")
                    loss = raw_loss.mean() if metadata.reduction == "mean" else raw_loss.sum()
                    gradients = torch.autograd.grad(loss, targets, allow_unused=True)
                for total, gradient in zip(totals, gradients):
                    if gradient is not None:
                        _finite(gradient, "individual gradient")
                        total.add_(gradient.detach().double().square())
                        _finite(total, "individual gradient-square accumulation")
                count += 1
                if count == max_examples:
                    break
            if count == max_examples:
                break
        if count == 0:
            raise NumericalDomainViolation("No unmasked calibration examples were collected within the explicit limits")
        finished = replace(passport, count=count, weight_sum=float(count))
        collection = {
            "individual_gradient_squares": True, "backward_passes": count,
            "observed_examples": count, "observed_batches": batches, "masked_examples": masked,
            "max_examples": max_examples, "max_batches": max_batches,
            "model_copy": "private_deepcopy", "original_state": "not_executed_or_differentiated",
            "rng_policy": "restore_python_numpy_torch_cpu", "installed_hooks": 0,
            "loss_reduction_scope": "within_individual_example", "labels": "observed",
            "full_fisher": False, "hessian": False,
        }
        return {path: fwsvd_statistics(total / count, finished, metadata,
                                      collection_manifest={**collection, "target_path": path})
                for path, total in zip(target_paths, totals)}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
