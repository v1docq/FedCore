from abc import abstractmethod
import torch
from torch import nn

from fedcore.models.network_impl.utils.hooks import BaseHook
from torch.ao.quantization import (quantize_dynamic)


class AbstractQuantizationHook(BaseHook):
    """Base class for quantization hooks.

    This abstract hook configures common quantization settings and provides a
    unified :meth:`action` implementation that:

    * sets the global quantized engine backend;
    * delegates to a subclass-specific :meth:`_child_action`;
    * marks the model as quantized via ``model._is_quantized = True``.
    """
    _QUANT_TYPE = 'AbstractQuantizationHook'

    def __init__(self, 
                 quant_each: int, 
                 dtype, 
                 allowed_quant_module_mapping: set[nn.Module], 
                 prepare_qat_after_epoch: int,
                 quantized_engine_backend: str
        ):
        super().__init__()
        self.dtype = dtype
        self.quant_each = quant_each
        self.allowed_quant_module_mapping = allowed_quant_module_mapping
        self.prepare_qat_after_epoch = prepare_qat_after_epoch
        self.quantized_engine_backend = quantized_engine_backend

    @classmethod
    def check_init(cls, d: dict):
        if (not super().check_init(d)):
            return False
        if d["quant_type"] != cls._QUANT_TYPE:
            return False
        return True
    
    @abstractmethod
    def _child_action(self, epoch, kws):
        """Subclass-specific quantization logic.

        This method must be implemented by concrete quantization hooks
        (:class:`DynamicQuantizationHook`, :class:`StaticQuantizationHook`,
        :class:`QATHook`). It is called from :meth:`action` after the
        quantized backend is set.
        """
        pass
    
    def action(self, epoch, kws):
        """Perform quantization step and mark the model as quantized.
        """
        torch.backends.quantized.engine = self.quantized_engine_backend
        self._child_action(epoch, kws)
        self.model._is_quantized = True


class DynamicQuantizationHook(AbstractQuantizationHook):
    """Hook for dynamic post-training quantization (PTQ).

    At the scheduled epochs, this hook calls
    :func:`torch.ao.quantization.quantize_dynamic` on the model using the
    provided module mapping and dtype. The model is temporarily put into
    eval mode during quantization and then switched back to train mode.

    Quantization type identifier: ``"dynamic"``.
    """
    _SUMMON_KEY = 'quant_type'
    HOOK_PLACE = 40
    _QUANT_TYPE = 'dynamic'

    def trigger(self, epoch, kws) -> bool:
        return self.is_epoch_arrived_default(epoch, self.quant_each)

    def _child_action(self, epoch, kws):
        print("[HOOK] Performing Dynamic PTQ hook operations.")
        self.model.eval()
        quantize_dynamic(
                    self.model,
                    qconfig_spec=self.allowed_quant_module_mapping,
                    dtype=self.dtype,
                    inplace=True)
        self.model.train()
        print("[HOOK] Dynamic PTQ setup completed.")


class StaticQuantizationHook(AbstractQuantizationHook):
    """Hook for static post-training quantization with calibration.

    At the scheduled epochs, this hook:

    1. Puts the model into eval mode.
    2. Calls :func:`torch.quantization.prepare` to insert observers.
    3. Runs a calibration pass over ``kws["val_loader"]`` under
       ``torch.no_grad()``.
    4. Calls :func:`torch.quantization.convert` to produce the quantized
       model.
    5. Switches the model back to train mode.

    Quantization type identifier: ``"static"``.
    """
    _SUMMON_KEY = 'quant_type'
    HOOK_PLACE = 40
    _QUANT_TYPE = 'static'

    def trigger(self, epoch, kws):
        return self.is_epoch_arrived_default(epoch, self.quant_each)

    def _child_action(self, epoch, kws):
        self.model.eval()
        torch.quantization.prepare(self.model, inplace=True)
        
        with torch.no_grad():
            for data, _ in kws['val_loader']:
                self.model(data.to(self.hookable_trainer.device))
        
        torch.quantization.convert(self.model, inplace=True)
        self.model.train()


class QATHook(AbstractQuantizationHook):
    """Hook for quantization-aware training (QAT).

    This hook has two distinct phases controlled by epochs:

    * At ``epoch == prepare_qat_after_epoch`` it prepares the model for QAT
      by calling :func:`torch.quantization.prepare_qat` in-place.
    * At ``epoch == quant_each`` (or at the last epoch if ``quant_each == -1``),
      it converts the QAT-prepared model into a quantized model via
      :func:`torch.quantization.convert`.

    Between these two phases, the trainer can perform regular training steps
    on the QAT-prepared model.

    Quantization type identifier: ``"qat"``.
    """
    _SUMMON_KEY = 'quant_type'
    HOOK_PLACE = 40
    _QUANT_TYPE = 'qat'
        
    def trigger(self, epoch, kws) -> bool:
        if (epoch == self.prepare_qat_after_epoch):
            return True
        elif (epoch == self.quant_each or (self.quant_each == -1 and epoch == self.hookable_trainer.epochs)):
            return True
        return False

    def _child_action(self, epoch, kws):
        if (epoch == self.prepare_qat_after_epoch):
            torch.quantization.prepare_qat(self.model, inplace=True)
        elif (epoch == self.quant_each or (self.quant_each == -1 and epoch == self.hookable_trainer.epochs)):
            self.model.eval()
            torch.quantization.convert(self.model, inplace=True)
            self.model.train()
