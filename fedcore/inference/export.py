from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
from copy import deepcopy
import os
import itertools
import subprocess
import tempfile
import json, time
import numpy as np

import torch
from torch import nn
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from fedcore.api.api_configs import ExportTemplate
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.inference.openvino import OpenVINOInferenceModel
from fedcore.inference.rknn import RKNNInferenceModel
from fedcore.inference.trt import TensorRTInferenceModel

# Optional
try:
    from openvino.tools.mo import convert as openvino_convert
except ImportError:
    openvino_convert = None

# Optional, NPU only!
try:
    from rknn.api import RKNN
except ImportError:
    RKNN = None


@dataclass
class ExportResult:
    model_path: Path
    inference_model: torch.nn.Module
    framework: str
    classes: List[str]


class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches, cache_file):
        super().__init__()
        self.batches, self.idx = batches, 0
        self.cache_file = str(cache_file)
        self.device_mem = cuda.mem_alloc(batches[0].nbytes)

    def get_batch_size(self):
        return self.batches[0].shape[0]

    def get_batch(self, names):
        if self.idx >= len(self.batches):
            return None
        cuda.memcpy_htod(self.device_mem, np.ascontiguousarray(self.batches[self.idx]))
        self.idx += 1
        return [int(self.device_mem)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class BaseExporter:
    DEVICE2FRAMEWORK = {"cpu": "onnx", "cuda": "tensorrt", "npu": "rknn"}
    def __init__(self, params: ExportTemplate):
        self.p = params
        self.device = self._resolve_device()
        print(self.device)
        self.framework = self._resolve_framework()
        print(self.framework)
        self._resolve_path()
        
        model_src = self.p.model_to_export
        if isinstance(model_src, str):
            self.model = torch.load(model_src, map_location=self.device)
        else:
            self.model = model_src.to(self.device)
        self.model.eval()

    def export(self) -> ExportResult:
        if self.framework == "onnx":
            return self._export_onnx()
        if self.framework == "tensorrt":
            return self._export_tensorrt()
        if self.framework == "openvino":
            return self._export_openvino()
        if self.framework == "rknn":
            return self._export_rknn()

    def _resolve_device(self) -> torch.device:
        if self.p.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.p.device)

    def _resolve_framework(self) -> str:
        if self.p.framework != "auto":
            return self.p.framework.lower()
        return self.DEVICE2FRAMEWORK[self.device.type]
    
    def _resolve_path(self):
        framework_ext = {"onnx": ".onnx", "tensorrt": ".engine",
                         "openvino": ".xml", "rknn": ".rknn"}[self.framework]
        print(self.p.output_path)
        self.p.output_path = Path(self.p.output_path, f"exported_model{framework_ext}")
        print(self.p.output_path)
        self.p.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_model(self, model):
        if hasattr(model, "engine") or hasattr(model, "session"):
            torch.save(model, self.p.output_path)
        else:
            torch.save(model.state_dict(), self.p.output_path)

    def _dummy_input(self) -> torch.Tensor:
        c, h, w = self.p.image_size
        return torch.randn(self.p.batch_size, c, h, w, device=self.device)

    def _replace_submodule(self, root: nn.Module, qualified_name: str, new_module: nn.Module):
        names = qualified_name.split(".")
        parent = root
        for n in names[:-1]:
            parent = getattr(parent, n)
        setattr(parent, names[-1], new_module)

    def _check_model(self, model: nn.Module, dummy_input: torch.Tensor) -> nn.Module:       
        feats: Dict[str, Tuple[int, int]] = {}
        def _hook(mod, inp, out):
            if out.dim() == 4:
                feats[id(mod)] = out.shape[-2:]
        handles = [m.register_forward_hook(_hook) for m in model.modules()]
        model.eval()
        with torch.no_grad():
            model(dummy_input)
        for h in handles:
            h.remove()
        for name, module in model.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                h, w = feats.get(id(module), (1, 1))
                new_pool = nn.AvgPool2d(kernel_size=(h, w), stride=1)
                self._replace_submodule(model, name, new_pool)
        return model

    def _export_onnx(self) -> ExportResult:
        onnx_path = self.p.output_path
        model = self._check_model(self.model, self._dummy_input())
        print(model)
        def _export(model, dummy):
            torch.onnx.export(
                model,
                dummy,
                onnx_path.as_posix(),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
                if self.p.dynamic_axes else None,
                opset_version=18,
            )
        try:
            _export(model, self._dummy_input())
        except:
            cpu_model = model.to("cpu")
            cpu_dummy = self._dummy_input().to("cpu")
            _export(cpu_model, cpu_dummy)
        inf = ONNXInferenceModel(onnx_path.as_posix())
        return ExportResult(onnx_path, inf, "onnx", self.p.classes)

    def _export_tensorrt(self) -> ExportResult:
        onnx_res = self._export_onnx()
        trt_path = self.p.output_path
        logger_trt = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     self.p.trt_workspace_gb * (1 << 30))
        parser = trt.OnnxParser(network, logger_trt)
        with open(onnx_res.model_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("TensorRT: ONNX parse failed")
        prec = self.p.tensorrt_precision.lower()
        if prec == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif prec == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calib_batches = self._load_calib_images()
            calibrator = _EntropyCalibrator(calib_batches, "calib.cache")
            try:
                config.int8_calibrator = calibrator
            except AttributeError:
                config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_int8_calibrator(calibrator)
        engine_bytes = builder.build_serialized_network(network, config)
        meta = dict(description="TensorRT model",
                    created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    version="1.0",
                    batch=self.p.batch_size,
                    imgsz=list(self.p.image_size[1:]),
                    names=self.p.classes)
        with open(trt_path, "wb") as f:
            meta_json = json.dumps(meta).encode()
            f.write(len(meta_json).to_bytes(4, "little", signed=True))
            f.write(meta_json)
            f.write(engine_bytes)
        inf = TensorRTInferenceModel(trt_path.as_posix())
        self.save_model(inf)
        return ExportResult(trt_path, inf, "tensorrt", self.p.classes)
    
    def _load_calib_images(self):
        if self.p.tensorrt_precision != "int8":
            return []
        dl = self.p.data
        c, h, w = self.p.image_size
        collected = []
        for batch in itertools.islice(dl, self.p.trt_calib_max_imgs // self.p.batch_size):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.permute(0, 3, 1, 2) if imgs.shape[-1] == 3 else imgs  # NHWC → NCHW
                if list(imgs.shape[1:]) != [c, h, w]:
                    imgs = torch.nn.functional.interpolate(imgs, size=(h, w), mode="bilinear")
                imgs = imgs.float().div(255).cpu().numpy()
            collected.append(imgs)
        return collected
    
    def _export_openvino(self) -> ExportResult:
        if openvino_convert is None:
            raise RuntimeError("OpenVINO not installed (`pip install openvino-dev`)")
        onnx_res = self._export_onnx()
        ir_model = openvino_convert(onnx_res.model_path.as_posix(),
                                    output_dir=self.output_dir.as_posix())
        xml_path = Path(ir_model.xml_path)
        inf = OpenVINOInferenceModel(xml_path.as_posix())
        self.save_model(inf)
        return ExportResult(xml_path, inf, "openvino", self.p.classes)

    def _export_rknn(self) -> ExportResult:
        if RKNN is None:
            raise RuntimeError("rknn-toolkit2 not installed")
        onnx_res = self._export_onnx()
        rknn_path = self.p.output_path
        rknn = RKNN()
        rknn.config(target_platform="rk3588")
        rknn.load_onnx(model=onnx_res.model_path.as_posix())
        rknn.build(do_quantization=self.p.rknn_quantize)
        rknn.export_rknn(rknn_path.as_posix())
        rknn.release()
        inf = RKNNInferenceModel(rknn_path.as_posix())
        self.save_model(inf)
        return ExportResult(rknn_path, inf, "rknn", self.p.classes)