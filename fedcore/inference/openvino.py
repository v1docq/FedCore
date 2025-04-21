from __future__ import annotations
from pathlib import Path
from typing import Sequence

import torch
from openvino.runtime import Core


__all__: Sequence[str] = ("OpenVINOInferenceModel", "load")


def load(xml_path: str | Path) -> "OpenVINOInferenceModel":
    return OpenVINOInferenceModel(str(xml_path))


class OpenVINOInferenceModel(torch.nn.Module):
    def __init__(self, xml_path: str, device: str = "AUTO"):
        super().__init__()
        core = Core()
        self.compiled_model = core.compile_model(xml_path, device)
        self.input_port = self.compiled_model.inputs[0]
        self.output_port = self.compiled_model.outputs[0]

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.compiled_model([x.detach().cpu().numpy()])[self.output_port]
        return torch.as_tensor(result)

    def to(self, *_, **__) -> "OpenVINOInferenceModel":
        return self
