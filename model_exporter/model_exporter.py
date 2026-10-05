"""Device-specific exports use one strict backend implementation."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from fedcore.external_runtime.contracts import ContractError, DeviceProfile
from fedcore.tools.export import export_model, normalize_framework


class ModelExporter:
    def __init__(self, device_arch):
        self.profile = device_arch if isinstance(device_arch, DeviceProfile) else DeviceProfile.parse(device_arch)
        self.device_arch = asdict(self.profile)

    def export_parts(self, parts, export_dir, format_type=None):
        # Validate every part and requested backend before writing the first artifact.
        decisions = []
        for part in parts:
            if part.get("profile") != asdict(self.profile):
                raise ContractError("profile_mismatch", "Splitter and exporter must use the same job profile")
            if part["is_npu_part"] and any(not self.profile.supports(layer["type"]) for layer in part["layers_info"]["layers"]):
                raise ContractError("unsupported_operation", "NPU part contains an unsupported operation")
            backend = normalize_framework(format_type or (self.profile.npu_framework if part["is_npu_part"] else self.profile.cpu_framework))
            if part.get("example_input") is None:
                raise ContractError("missing_input_spec", "Actual intermediate tensors are required to export parts")
            decisions.append((part, backend))
        files = []
        for part, backend in decisions:
            suffix = {"torchscript": ".pt", "onnx": ".onnx", "tensorrt": ".engine"}[backend]
            device = "npu" if part["is_npu_part"] else "cpu"
            path = Path(export_dir) / f"model_part_{part['part_index']}_{device}{suffix}"
            files.append(str(export_model(part["model"], backend, path, part["example_input"])))
        return files

    def _export_torchscript(self, model, path, example_input=None):
        return export_model(model, "torchscript", path, example_input)

    def _export_onnx_with_versions(self, model, path, example_input=None):
        return export_model(model, "onnx", path, example_input)

    def _get_example_input(self, model):
        raise ContractError("missing_input_spec", "Provide an actual example tensor; image dimensions are never inferred")
