import torch
from torch import nn, optim
from enum import Enum

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.hooks import BaseHook


class DynamicQuantizationHook(BaseHook):
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.dtype = params.get('dtype', None)

    def trigger(self, quant_type, **kwargs):
        return quant_type == 'dynamic'

    def action(self, quant_type, kws):
        print("[HOOK] Performing Dynamic PTQ hook operations.")
        model = kws['model']
        model.eval()
        model.to(default_device())
        print("[HOOK] Dynamic PTQ setup completed.")


class StaticQuantizationHook(BaseHook):
    _SUMMON_KEY = 'quantization'
    _hook_place = 'post'

    def trigger(self, quant_type, **kwargs):
        return quant_type == 'static'

    def action(self, quant_type, kws):
        model = kws['model'].eval().to(default_device())
        with torch.no_grad():
            for data, _ in self.params['input_data'].features.val_dataloader:
                model(data.to(default_device()))


class QATHook(BaseHook):
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
        return quant_type == 'qat'

    def action(self, quant_type, kws):
        print("[HOOK] Performing QAT training hook.")
        model = kws['model']
        model.train()
        model.to(self.params['device'])
        optimizer = self.optimizer(model.parameters())
        criterion = self.criterion

        for epoch in range(self.epochs):
            print(f"[HOOK] QAT epoch {epoch+1}/{self.epochs} started.")
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(default_device()), target.to(default_device())
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