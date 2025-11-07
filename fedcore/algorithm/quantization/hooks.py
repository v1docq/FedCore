from abc import abstractmethod
import torch
from torch import nn

from fedcore.models.network_impl.hooks import BaseHook
from torch.ao.quantization import (quantize_dynamic)

class AbstractQuantizationHook(BaseHook):
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
        pass
    
    def action(self, epoch, kws):
        torch.backends.quantized.engine = self.quantized_engine_backend
        self._child_action(epoch, kws)
        self.model._is_quantized = True

class DynamicQuantizationHook(AbstractQuantizationHook):
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