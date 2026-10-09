"""Sequential partition support is validated against actual FX execution edges."""
from __future__ import annotations
import copy
from dataclasses import asdict
from typing import Any
import torch
from torch import nn
from fedcore.external_runtime.contracts import ContractError, DeviceProfile


class ModelAnalyzer:
    op_mapping = {"Linear": "Gemm", "Conv1d": "Conv", "Conv2d": "Conv", "Conv3d": "Conv",
                  "ReLU": "Relu", "Identity": "Identity", "Flatten": "Flatten",
                  "BatchNorm1d": "BatchNormalization", "BatchNorm2d": "BatchNormalization",
                  "Sigmoid": "Sigmoid", "Tanh": "Tanh", "Dropout": "Dropout",
                  "MaxPool1d": "MaxPool", "MaxPool2d": "MaxPool", "AvgPool1d": "AvgPool", "AvgPool2d": "AvgPool"}

    def __init__(self, device_arch):
        self.profile = device_arch if isinstance(device_arch, DeviceProfile) else DeviceProfile.parse(device_arch)
        self.device_arch = asdict(self.profile)

    def get_layer_type(self, layer):
        return self.op_mapping.get(type(layer).__name__, type(layer).__name__)

    def analyze_model_structure(self, model, example_input=None):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")
        local = copy.deepcopy(model).eval()
        # Wrap a root torch leaf so FX records a call_module rather than its internals.
        source = nn.Sequential(local) if not list(local.children()) and type(local).__module__.startswith("torch.nn") else local
        try:
            graph = torch.fx.symbolic_trace(source)
        except Exception as error:
            raise ContractError("unsupported_graph", "Model cannot be proven to be a sequential FX chain") from error
        nodes = list(graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        if len(placeholders) != 1:
            raise ContractError("unsupported_graph", "Partitioning supports exactly one tensor input")
        previous = placeholders[0]
        layers = []
        value = example_input.detach().clone() if isinstance(example_input, torch.Tensor) else example_input
        if value is not None and not isinstance(value, torch.Tensor):
            raise ContractError("unsupported_input", "Partitioning requires one tensor input")
        for node in nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                if node.args != (previous,):
                    raise ContractError("unsupported_graph", "Output must be the final sequential value")
                continue
            if node.op != "call_module" or node.args != (previous,) or node.kwargs:
                raise ContractError("unsupported_graph", "Branching, functional links and multiple inputs cannot be partitioned as a chain")
            module = graph.get_submodule(str(node.target))
            operation = self.get_layer_type(module)
            if operation not in set(self.op_mapping.values()) or not type(module).__module__.startswith("torch.nn"):
                raise ContractError("unsupported_graph", "Only declared torch.nn leaf operations may be partitioned")
            input_shape = list(value.shape) if isinstance(value, torch.Tensor) else None
            if value is not None:
                with torch.inference_mode():
                    value = module(value)
                if not isinstance(value, torch.Tensor):
                    raise ContractError("unsupported_graph", "Intermediate values must be tensors")
            layers.append({"name": str(node.target), "node": node.name, "input_node": previous.name,
                           "type": operation, "module": module, "module_object": module,
                           "supported": self.profile.supports(operation), "input_shape": input_shape,
                           "output_shape": list(value.shape) if isinstance(value, torch.Tensor) else None})
            previous = node
        if not layers or len(previous.users) != 1:
            raise ContractError("unsupported_graph", "Graph must be a nonempty chain")
        if example_input is not None:
            with torch.inference_mode():
                torch.testing.assert_close(value, local(example_input.detach().clone()))
        return layers

    def find_split_points(self, layers_info):
        return [i for i in range(1, len(layers_info)) if layers_info[i-1]["supported"] != layers_info[i]["supported"]]

    def get_model_parts_info(self, model, example_input=None):
        layers = self.analyze_model_structure(model, example_input)
        points = self.find_split_points(layers)
        parts = []
        start = 0
        for end in points + [len(layers)]:
            selected = layers[start:end]
            supported = all(layer["supported"] for layer in selected)
            count = sum(layer["supported"] for layer in selected)
            parts.append({"part_index": len(parts), "start_layer": start, "end_layer": end,
                          "layers_count": len(selected), "supported_layers": count,
                          "unsupported_layers": len(selected)-count, "layers": selected,
                          "is_npu_part": supported, "input_shape": selected[0]["input_shape"],
                          "output_shape": selected[-1]["output_shape"]})
            start = end
        return {"model_layers": layers, "split_points": points, "parts_info": parts,
                "total_layers": len(layers), "supported_layers": sum(layer["supported"] for layer in layers),
                "unsupported_layers": sum(not layer["supported"] for layer in layers), "profile": asdict(self.profile)}
