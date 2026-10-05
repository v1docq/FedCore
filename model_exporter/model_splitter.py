"""Independent partitions for the explicitly supported sequential graph subset."""
from __future__ import annotations
import copy
import torch
from torch import nn
from fedcore.external_runtime.contracts import ContractError
try:
    from .model_analyzer import ModelAnalyzer
except ImportError:
    from model_analyzer import ModelAnalyzer


class ModelSplitter:
    def __init__(self, device_arch):
        self.analyzer = ModelAnalyzer(device_arch)
        self.device_arch = self.analyzer.device_arch

    def get_parts_info(self, model, example_input=None):
        return self.analyzer.get_model_parts_info(model, example_input)

    def split_model(self, model, parts_info, example_input=None):
        expected = self.get_parts_info(model, example_input)
        supplied = parts_info.get("parts_info", [])
        if parts_info.get("profile") != expected["profile"] or len(supplied) != len(expected["parts_info"]):
            raise ContractError("invalid_partition", "Partition was planned for a different profile or graph")
        for supplied_part, planned in zip(supplied, expected["parts_info"]):
            for field in ("start_layer", "end_layer", "is_npu_part"):
                if supplied_part.get(field) != planned[field]:
                    raise ContractError("invalid_partition", "Partition boundaries or support flags differ from actual execution")
        parts = []
        value = example_input.detach().clone() if example_input is not None else None
        for part in expected["parts_info"]:
            module = self._create_part_model(model, expected["model_layers"], part["start_layer"], part["end_layer"])
            example = value.detach().clone() if value is not None else None
            if value is not None:
                with torch.inference_mode():
                    value = module(value)
            parts.append({"part_index": part["part_index"], "model": module,
                          "is_npu_part": part["is_npu_part"], "layers_info": part,
                          "example_input": example, "profile": expected["profile"]})
        if example_input is not None:
            local = copy.deepcopy(model).eval()
            with torch.inference_mode():
                torch.testing.assert_close(value, local(example_input.detach().clone()))
        return parts

    def _create_part_model(self, model, layers_info, start_idx, end_idx):
        return nn.Sequential(*(copy.deepcopy(layer["module"]) for layer in layers_info[start_idx:end_idx])).eval()
