import os

import onnxruntime as ort
import torch
from torch import nn


class ONNXInferenceModel(nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model
        self.providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self.ort_session = ort.InferenceSession(model, providers=self.providers)

    def forward(self, inputs):
        inputs = inputs.cpu()
        return torch.Tensor(self.ort_session.run(None, {"input": inputs.numpy()}))

    def to(self, device):
        # onnx runtime chooses it's own way
        return self

    def size(self):
        return os.path.getsize(self.model_name)
