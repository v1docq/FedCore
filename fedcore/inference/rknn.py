from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None

__all__: Sequence[str] = ("RKNNInferenceModel", "load")


def load(path: str | Path, perf_debug: bool = False) -> "RKNNInferenceModel":
    return RKNNInferenceModel(str(path), perf_debug=perf_debug)


class RKNNInferenceModel(torch.nn.Module):
    def __init__(self, rknn_path: str, perf_debug: bool = False):
        super().__init__()
        self.rknn = RKNNLite()
        target = None # "rk3588"
        self.rknn.load_rknn(rknn_path)
        self.perf_debug = perf_debug

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = [x.detach().cpu().numpy().astype(np.float32)]
        outputs, _ = self.rknn.inference(inputs=inputs, data_format="nhwc",
                                         perf_debug=self.perf_debug)
        return torch.as_tensor(outputs[0])

    def to(self, *_, **__) -> "RKNNInferenceModel":
        return self
