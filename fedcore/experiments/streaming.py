"""Batch reductions for ablations; bounds exclude the model and its activations."""
from __future__ import annotations

import math

import torch
from torch import nn

from .math_checks import activation_rows, layer_matrices


class RelativeNorm:
    """Accumulate aligned finite tensor norms without retaining predictions."""
    def __init__(self):
        self.error_squared = 0.
        self.reference_squared = 0.
        self.elements = 0

    def add(self, reference, approximation):
        reference, approximation = reference.detach().cpu(), approximation.detach().cpu()
        if reference.shape != approximation.shape or not torch.isfinite(reference).all() or not torch.isfinite(approximation).all():
            raise ValueError("Aligned finite tensors are required")
        reference, approximation = reference.double(), approximation.double()
        self.error_squared += float((approximation - reference).square().sum())
        self.reference_squared += float(reference.square().sum())
        self.elements += reference.numel()

    def report(self):
        if not self.elements:
            raise ValueError("Layer was not executed on the supplied role")
        numerator, denominator = math.sqrt(self.error_squared), math.sqrt(self.reference_squared)
        return {"absolute": numerator, "denominator": denominator,
                "relative": numerator / denominator if denominator else (0. if numerator == 0 else None),
                "status": "measured" if denominator else ("zero_reference_exact" if numerator == 0 else "undefined_zero_reference")}


class ActivationMoments:
    """Exact uncentered moments per group, in float64 and original row order.

    Retains O(groups * input_dimension**2) numbers; processes convolution patches
    one sample at a time and Gram updates in bounded row chunks. The explicit
    limit covers the moments and patch/reduction workspace, not forward memory.
    A very wide layer or single image can therefore be explicitly unsupported.
    """
    def __init__(self, layer, *, max_bytes=256 * 1024 * 1024, row_chunk_size=4096):
        if type(max_bytes) is not int or max_bytes <= 0 or type(row_chunk_size) is not int or row_chunk_size <= 0:
            raise ValueError("Moment memory and row chunk limits must be positive integers")
        matrices = layer_matrices(layer)
        groups, _, dimension = matrices.shape
        self.layer, self.max_bytes = layer, max_bytes
        self.moment_bytes = groups * dimension * dimension * 8
        # Gram matrix update and final division need at most one additional copy.
        if 2 * self.moment_bytes + dimension * 32 > max_bytes:
            raise ValueError("unsupported: layer second moment exceeds the declared memory limit")
        self.row_chunk_size = min(row_chunk_size, max(1, (max_bytes - 2 * self.moment_bytes) // (dimension * 32)))
        self.sums = [torch.zeros(dimension, dimension, dtype=torch.float64) for _ in range(groups)]
        self.counts = [0] * groups
        self.forwards = 0
        self.largest_patch_bytes = 0

    def _patch_bytes(self, sample):
        if isinstance(self.layer, nn.Linear):
            return 0
        layer = self.layer
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d)) or isinstance(layer.padding, str):
            raise ValueError("Convolution requires explicit numeric padding")
        positions = 1
        for size, kernel, dilation, padding, stride in zip(sample.shape[2:], layer.kernel_size, layer.dilation, layer.padding, layer.stride):
            positions *= (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        return positions * layer.in_channels * math.prod(layer.kernel_size) * sample.element_size()

    def add(self, inputs):
        self.forwards += 1
        inputs = inputs.detach()
        for sample in inputs.split(1):
            patch_bytes = self._patch_bytes(sample)
            # Unfold plus the group row views/copies can coexist. Do not silently
            # allocate an image's patches beyond the chosen reduction workspace.
            if 2 * self.moment_bytes + 3 * patch_bytes + self.row_chunk_size * self.sums[0].shape[0] * 32 > self.max_bytes:
                raise ValueError("unsupported: one convolution patch workspace exceeds the declared memory limit")
            self.largest_patch_bytes = max(self.largest_patch_bytes, patch_bytes)
            rows = activation_rows(self.layer, sample)
            for group, observations in enumerate(rows):
                for chunk in observations.split(self.row_chunk_size):
                    if not torch.isfinite(chunk).all():
                        raise ValueError("Calibration activations must be finite")
                    x = chunk.to(device="cpu", dtype=torch.float64)
                    self.sums[group].add_(x.T @ x)
                    self.counts[group] += len(x)

    def finish(self):
        if not all(self.counts):
            raise ValueError("Layer was not executed on calibration")
        for total, count in zip(self.sums, self.counts):
            total.div_(count)
        return tuple(self.sums), {"observations_per_group": self.counts, "forwards": self.forwards,
                                  "moment_tensor_bytes": self.moment_bytes, "largest_patch_tensor_bytes": self.largest_patch_bytes,
                                  "workspace_limit_bytes": self.max_bytes, "row_chunk_size": self.row_chunk_size,
                                  "scope": "reduction workspace only; model forward, SVD/eigensolve, allocator and libraries excluded"}


def capture_moments(model, layer, split, batch_size, *, max_bytes=256 * 1024 * 1024):
    accumulator = ActivationMoments(layer, max_bytes=max_bytes)
    handle = layer.register_forward_pre_hook(lambda _module, values: accumulator.add(values[0]))
    try:
        with torch.inference_mode():
            for start in range(0, len(split.x), batch_size):
                model(split.x[start:start + batch_size])
    finally:
        handle.remove()
    return accumulator.finish()


def layer_error(model, layer, replacement, split, batch_size):
    accumulator = RelativeNorm()
    device = next(replacement.parameters()).device
    def consume(_module, values, output):
        approximation = replacement(values[0].to(device))
        accumulator.add(output, approximation)
    handle = layer.register_forward_hook(consume)
    try:
        with torch.inference_mode():
            for start in range(0, len(split.x), batch_size):
                model(split.x[start:start + batch_size])
    finally:
        handle.remove()
    return accumulator.report()


def model_error(reference, approximation, split, batch_size):
    accumulator = RelativeNorm()
    left_device = next(reference.parameters(), split.x).device
    right_device = next(approximation.parameters(), split.x).device
    with torch.inference_mode():
        for start in range(0, len(split.x), batch_size):
            x = split.x[start:start + batch_size]
            accumulator.add(reference(x.to(left_device)), approximation(x.to(right_device)))
    return accumulator.report()
