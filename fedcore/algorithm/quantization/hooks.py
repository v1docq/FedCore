"""
Quantization hooks for FedCore.

This module defines hook classes that plug into the training/serving pipeline
to perform post-training quantization (PTQ) and quantization-aware training (QAT)
actions at well-defined moments. Hooks follow FedCore's hook protocol
(:class:`fedcore.models.network_impl.hooks.BaseHook`) and are selected by
``_SUMMON_KEY = 'quantization'``.

Hooks
-----
- DynamicQuantizationHook
    Lightweight post-training quantization preparation step (no calibration).
- StaticQuantizationHook
    Runs a calibration loop over a validation dataloader to prepare static PTQ.
- QATHook
    Performs a short quantization-aware fine-tuning loop.

Integration
-----------
These hooks assume the caller passes a ``kws`` dictionary containing:
    - ``'model'``: the torch model being quantized,
    - and that ``params`` provided to the hook include:
        * ``'device'``: target torch device string,
        * optionally ``'input_data'`` with ``features.val_dataloader`` (for static PTQ),
        * optionally training hyperparameters (for QAT).
"""

import torch
from torch import nn, optim
from enum import Enum

from fedcore.models.network_impl.hooks import BaseHook


class DynamicQuantizationHook(BaseHook):
    """Hook for dynamic post-training quantization (PTQ).

    This hook prepares the model for dynamic quantization by switching to eval
    mode and moving it to the requested device. Actual quantization (e.g.,
    ``torch.quantization.quantize_dynamic``) is typically performed elsewhere;
    this hook focuses on the orchestration step.

    Hook metadata
    -------------
    _SUMMON_KEY : 'quantization'
    _hook_place : 'post'  (run after the main training step)

    Parameters
    ----------
    params : dict-like
        Expected keys:
          - ``'device'``: target device string (e.g., 'cpu', 'cuda:0').
          - ``'dtype'`` (optional): quantized dtype preference.
    model : nn.Module
        Model reference (not modified directly here; the instance is provided
        again via ``kws['model']`` at call time).

    Attributes
    ----------
    dtype : Any | None
        Optional dtype preference passed via params (not applied here).
    """
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.dtype = params.get('dtype', None)

    def trigger(self, quant_type, **kwargs):
        """Return True when the requested quantization type is ``'dynamic'``."""
        return quant_type == 'dynamic'

    def action(self, quant_type, kws):
        """Put the model in eval mode and move it to the configured device.

        Parameters
        ----------
        quant_type : str
            The selected quantization mode (expected: 'dynamic').
        kws : dict
            Must contain ``'model'`` (nn.Module).
        """
        print("[HOOK] Performing Dynamic PTQ hook operations.")
        model = kws['model']
        model.eval()
        model.to(self.params['device'])
        print("[HOOK] Dynamic PTQ setup completed.")


class StaticQuantizationHook(BaseHook):
    """Hook for static post-training quantization (PTQ) calibration.

    Runs a simple calibration pass over the validation dataloader to collect
    activation statistics, with gradients disabled.

    Hook metadata
    -------------
    _SUMMON_KEY : 'quantization'
    _hook_place : 'post'

    Expected ``params`` keys
    ------------------------
    - ``'device'``: torch device string.
    - ``'input_data'``: an object exposing ``features.val_dataloader``.
    """
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def trigger(self, quant_type, **kwargs):
        """Return True when the requested quantization type is ``'static'``."""
        return quant_type == 'static'

    def action(self, quant_type, kws):
        """Run a no-grad calibration loop over the validation dataloader.

        Parameters
        ----------
        quant_type : str
            The selected quantization mode (expected: 'static').
        kws : dict
            Must contain ``'model'`` (nn.Module). ``params`` must provide
            ``'input_data'.features.val_dataloader`` and ``'device'``.
        """
        model = kws['model'].eval().to(self.params['device'])
        with torch.no_grad():
            for data, _ in self.params['input_data'].features.val_dataloader:
                model(data.to(self.params['device']))


class QATHook(BaseHook):
    """Hook for quantization-aware training (QAT).

    Executes a short fine-tuning loop (epochs and optimizer configurable) to
    adapt the model to fake-quantized operations.

    Hook metadata
    -------------
    _SUMMON_KEY : 'quantization'
    _hook_place : 'post'

    Parameters
    ----------
    params : dict-like
        Recognized keys:
          - ``'epochs'`` (int, default=2): number of QAT epochs,
          - ``'optimizer'`` (torch.optim class, default=Adam),
          - ``'criterion'`` (loss, default=CrossEntropyLoss()),
          - ``'lr'`` (float, default=0.001),
          - ``'device'`` (str): target device,
          - ``'input_data'``: provides ``features.train_dataloader``.
        If ``'criterion'`` is passed as a tuple, the first element is used.
    model : nn.Module
        Model reference (actual instance passed again via ``kws['model']``).

    Attributes
    ----------
    epochs : int
        Number of epochs for the QAT loop.
    optimizer : torch.optim.Optimizer | type
        Optimizer class used to construct an instance on demand.
    criterion : nn.Module
        Loss function used during QAT.
    learning_rate : float
        Learning rate for the optimizer.
    train_dataloader : DataLoader
        Training dataloader used for QAT updates.
    """
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.epochs = params.get("epochs", 2)
        self.optimizer = params.get("optimizer", optim.Adam)
        self.criterion = params.get("criterion", nn.CrossEntropyLoss())
        self.learning_rate = params.get("lr", 0.001)
        self.train_dataloader = params['input_data'].features.train_dataloader
        if isinstance(self.criterion, tuple):
            self.criterion = self.criterion[0]
        
    def trigger(self, quant_type, **kwargs):
        """Return True when the requested quantization type is ``'qat'``."""
        return quant_type == 'qat'

    def action(self, quant_type, kws):
        """Run a short QAT fine-tuning loop on the training dataloader.

        Parameters
        ----------
        quant_type : str
            The selected quantization mode (expected: 'qat').
        kws : dict
            Must contain ``'model'`` (nn.Module). ``params`` must provide
            ``'device'`` and ``'input_data'.features.train_dataloader``.
        """
        print("[HOOK] Performing QAT training hook.")
        model = kws['model']
        model.train()
        model.to(self.params['device'])
        optimizer = self.optimizer(model.parameters())
        criterion = self.criterion

        for epoch in range(self.epochs):
            print(f"[HOOK] QAT epoch {epoch+1}/{self.epochs} started.")
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.params['device']), target.to(self.params['device'])
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print(f"[HOOK][Epoch: {epoch+1}] Batch {batch_idx+1} Loss: {loss.item():.4f}")
        print("[HOOK] QAT training completed.")


class QuantizationHooks(Enum):
    """Registry of available quantization hooks."""
    DYNAMIC_QUANTIZATION_HOOK = DynamicQuantizationHook
    STATIC_QUANTIZATION_HOOK = StaticQuantizationHook
    QAT_HOOK = QATHook
