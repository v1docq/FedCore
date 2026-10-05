"""Compatibility facade. Every operation constructs its own immutable profile."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from fedcore.external_runtime.contracts import DeviceProfile
from fedcore.external_runtime.models import load_model_bundle, save_model_bundle
from fedcore.tools.export import export_model
try:
    from .model_analyzer import ModelAnalyzer
    from .model_splitter import ModelSplitter
    from .model_exporter import ModelExporter
except ImportError:
    from model_analyzer import ModelAnalyzer
    from model_splitter import ModelSplitter
    from model_exporter import ModelExporter


class ModelManager:
    def __init__(self, profile=None):
        self.profile = profile if isinstance(profile, DeviceProfile) else DeviceProfile.parse(profile or {})
        self.device_arch = asdict(self.profile)
        self.exporter = ModelExporter(self.profile)
        self.splitter = ModelSplitter(self.profile)
        self.analyzer = ModelAnalyzer(self.profile)

    def load_default_architecture(self):
        return asdict(DeviceProfile())

    def load_architecture(self, arch_file):
        path = Path(arch_file).resolve()
        root = Path(__file__).parent.joinpath("device_architectures").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profiles must be selected from packaged device_architectures")
        return asdict(DeviceProfile.parse(json.loads(path.read_text(encoding="utf-8"))))

    def save_model(self, model, model_path):
        save_model_bundle(model, model_path)
        return True

    def load_model(self, model_path):
        return load_model_bundle(model_path)

    def export_model(self, model, export_format, export_dir, model_name, example_input=None):
        try:
            if Path(model_name).name != model_name or model_name in ("", ".", ".."):
                raise ValueError("model_name must be a filename stem")
            path = export_model(model, export_format, Path(export_dir) / model_name, example_input)
            return {"message": "Artifact exported and loader validated", "file": str(path)}
        except Exception as error:
            return {"error": error.to_dict() if hasattr(error, "to_dict") else str(error)}

    def export_parts(self, model, export_dir, architecture_file=None, example_input=None, profile=None):
        try:
            selected = profile or (self.load_architecture(architecture_file) if architecture_file else self.profile)
            # One profile value feeds all collaborators for this call.
            splitter = ModelSplitter(selected)
            exporter = ModelExporter(selected)
            info = splitter.get_parts_info(model, example_input)
            parts = splitter.split_model(model, info, example_input)
            files = exporter.export_parts(parts, export_dir)
            return {"message": "Partitions exported and loader validated", "exported_files": files, "parts_count": len(files)}
        except Exception as error:
            return {"error": error.to_dict() if hasattr(error, "to_dict") else str(error)}

    def analyze_model(self, model, example_input=None, profile=None):
        try:
            info = ModelAnalyzer(profile or self.profile).get_model_parts_info(model, example_input)
            info["model_layers"] = [{k: v for k, v in layer.items() if k not in ("module", "module_object")} for layer in info["model_layers"]]
            for part in info["parts_info"]:
                part.pop("layers")
            return {"model_analysis": info, **{key: info[key] for key in ("total_layers", "supported_layers", "unsupported_layers")}}
        except Exception as error:
            return {"error": error.to_dict() if hasattr(error, "to_dict") else str(error)}

    def get_supported_ops(self):
        ops = [op for op in self.profile.supported_ops if self.profile.supports(op)]
        return {"supported_operations": ops, "count": len(ops)}

    def get_architectures(self):
        root = Path(__file__).parent / "device_architectures"
        values = []
        for path in sorted(root.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            values.append({"name": raw.get("name", path.stem), "alias": raw.get("name", path.stem),
                           "filename": path.name, "file": path.name, "cpu_framework": raw.get("cpu_framework"),
                           "npu_framework": raw.get("npu_framework")})
        return {"architectures": values}

    def set_device_architecture(self, arch_file):
        # GUI-only compatibility: replace the complete manager configuration.
        try:
            name = Path(arch_file).name
            profile = DeviceProfile.parse(self.load_architecture(Path(__file__).parent / "device_architectures" / name))
            self.__init__(profile)
            return True
        except (ValueError, OSError):
            return False

    def analyze_log(self, path):
        try:
            from .log_analizer import LogAnalyzer
        except ImportError:
            from log_analizer import LogAnalyzer
        analyzer = LogAnalyzer()
        return {"analysis": analyzer.analyze_problems(path), "problems": analyzer.find_problematic_layers(path),
                "report": analyzer.generate_detailed_report(path)}


model_manager = ModelManager()
