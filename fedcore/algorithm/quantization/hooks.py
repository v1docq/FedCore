"""Historical trainer helpers, outside the supported quantization operation.

These hooks do not own prepare/convert or prove a QAT conversion. Use
BaseQuantizer.fit for the supported prepare -> train/calibrate -> convert
contract and its explicit QuantizationResult. Kept for import compatibility.
"""

import torch
from torch import nn, optim
from enum import Enum

from fedcore.models.network_impl.utils.hooks import BaseHook


class DynamicQuantizationHook(BaseHook):
    """Legacy eval/device preparation helper; it does not convert a model."""
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def __init__(self, trainer):
        super().__init__(trainer)
        self.dtype = self.hookable_trainer.params.get('dtype', None)

    def trigger(self, quant_type, **kwargs):
        return quant_type == 'dynamic'

    def action(self, quant_type, kws):
        print("[HOOK] Performing Dynamic PTQ hook operations.")
        model = kws['model']
        model.eval()
        model.to(self.params['device'])
        print("[HOOK] Dynamic PTQ setup completed.")


class StaticQuantizationHook(BaseHook):
    """Legacy calibration helper; it does not prepare or convert a model."""
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def trigger(self, quant_type, **kwargs):
        """Check whether static quantization should run at this epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Returns
        -------
        bool
            ``True`` if ``epoch`` matches the schedule defined by
            ``self.quant_each``, otherwise ``False``.
        """
        return quant_type == 'static'

    def action(self, quant_type, kws):
        """Run prepare–calibrate–convert static PTQ pipeline.

        Parameters
        ----------
        epoch : int
            Current epoch number (unused).
        kws : dict
            Dictionary expected to contain ``"val_loader"`` with a validation
            dataloader for calibration.
        """
        model = kws['model'].eval().to(self.params['device'])
        with torch.no_grad():
            for data, _ in self.params['input_data'].features.val_dataloader:
                model(data.to(self.params['device']))


class QATHook(BaseHook):
    """Legacy training helper; use BaseQuantizer for prepared QAT conversion."""
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def __init__(self, trainer):
        super().__init__(trainer)
        params = self.hookable_trainer.params
        self.epochs = params.get("epochs", 2)
        self.optimizer = params.get("optimizer", optim.Adam)
        self.criterion = params.get("criterion", nn.CrossEntropyLoss())
        self.learning_rate = params.get("lr", 0.001)
        #print('@@@@@@@@@@@@@@@@ PARAMS', params.keys())

        self.train_dataloader = params['input_data'].train_dataloader
        if isinstance(self.criterion, tuple):
            self.criterion = self.criterion[0]

    def trigger(self, quant_type, **kwargs):
        """Determine whether to prepare QAT or convert at the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Returns
        -------
        bool
            ``True`` if either QAT preparation or final conversion should
            occur at this epoch, otherwise ``False``.
        """
        return quant_type == 'qat'

    def action(self, quant_type, kws):
        """Run QAT preparation or final conversion depending on the epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Notes
        -----
        * At ``epoch == prepare_qat_after_epoch``:
          :func:`torch.quantization.prepare_qat` is called in-place.
        * At ``epoch == quant_each`` or if ``quant_each == -1`` and this is
          the last epoch:
          the model is converted to a quantized version via
          :func:`torch.quantization.convert`.
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
    DYNAMIC_QUANTIZATION_HOOK = DynamicQuantizationHook
    STATIC_QUANTIZATION_HOOK = StaticQuantizationHook
    QAT_HOOK = QATHook
