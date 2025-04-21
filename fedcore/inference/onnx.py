from __future__ import annotations
from pathlib import Path
from typing import Sequence

import torch
import onnxruntime as ort


__all__: Sequence[str] = ("ONNXInferenceModel", "load")


def load(path: str | Path) -> "ONNXInferenceModel":
    return ONNXInferenceModel(str(path))


class ONNXInferenceModel(torch.nn.Module):
    _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, onnx_path: str):
        super().__init__()
        self.session = ort.InferenceSession(
            onnx_path,
            providers=self._providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.session.run(
            [self.output_name],
            {self.input_name: x.detach().cpu().numpy()},
        )[0]
        return torch.as_tensor(out)

    def to(self, *_, **__) -> "ONNXInferenceModel":  # «no‑op»
        return self