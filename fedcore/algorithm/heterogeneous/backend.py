"""Real ONNX Q/DQ export and lazy TensorRT deployment for closed CV recipes.

CPU ONNX Runtime validation is deliberately not an A100 performance claim.
Optional dependencies and CUDA are touched only by the requested backend.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import (
    CandidateStorageError, capture_inventory, inventory_fingerprint, inventory_cost, storage_guard)
from fedcore.algorithm.low_rank.cv_execution import _calibration
from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid, NumericalFailure, ResourceFailure
from fedcore.algorithm.low_rank.plans import Violation
from fedcore.algorithm.pruning.sparse.cv_channels import BLOCK_PATHS, validate_cv_model
from .contracts import fingerprint
from .cv import validate_workflow


@dataclass(frozen=True)
class BackendSpec:
    runtime: str = "onnxruntime"
    device_uuid: str | None = None
    opset: int = 17
    calibration_batch_size: int = 16
    max_batch_size: int = 128
    opt_batch_size: int = 128
    workspace_bytes: int = 1024**3
    float_rtol: float = 2e-4
    float_atol: float = 2e-5
    backend_rtol: float = 2e-2
    backend_atol: float = 2e-2

    def __post_init__(self):
        if self.device_uuid is not None:
            from fedcore.experiments.heterogeneous_device import validate_gpu_uuid
            validate_gpu_uuid(self.device_uuid)
        if (self.runtime not in ("onnxruntime", "tensorrt") or self.opset != 17
                or any(type(v) is not int or v < 1 for v in
                       (self.calibration_batch_size, self.max_batch_size, self.opt_batch_size, self.workspace_bytes))
                or self.opt_batch_size > self.max_batch_size
                or any(type(v) not in (float, int) or not np.isfinite(v) or v < 0
                       for v in (self.float_rtol, self.float_atol, self.backend_rtol, self.backend_atol))):
            raise ValueError("Explicit bounded backend profile and finite export tolerances required")


class BackendContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DeploymentArtifact:
    model_path: str
    manifest_path: str
    backend: str
    storage_bytes: int
    evidence: dict

    def load(self):
        return load_deployment(self.manifest_path)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def deployment_size(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _check_manifest(path, manifest)
    return sum((path.parent / name).stat().st_size for name in manifest["required_files"])


def _check_manifest(path, manifest):
    if (type(manifest) is not dict or manifest.get("schema") != "heterogeneous_deployment_v1"
            or manifest.get("backend") not in ("onnxruntime", "tensorrt")):
        raise BackendContractError("InvalidManifest", "Unknown self-contained deployment schema")
    primary = "model.onnx" if manifest["backend"] == "onnxruntime" else "model.engine"
    required = [primary, "manifest.json"]
    if manifest.get("required_files") != required or path.name != "manifest.json":
        raise BackendContractError("InvalidManifest", "Exact complete deployment file list required")
    if manifest.get("model_sha256") != _sha(path.parent / primary):
        raise BackendContractError("ArtifactChanged", "Deployment bytes changed after validation")


def _fold_bn(model):
    """Owned evaluation-only folding; the original CV constructor stays untouched."""
    result = deepcopy(model).eval()
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    result.conv1, result.bn1 = fuse(result.conv1, result.bn1), nn.Identity()
    for path in BLOCK_PATHS:
        block = result.get_submodule(path)
        block.conv1, block.bn1 = fuse(block.conv1, block.bn1), nn.Identity()
        if type(block.conv2) is nn.Sequential:
            block.conv2[1] = fuse(block.conv2[1], block.bn2)
        else:
            block.conv2 = fuse(block.conv2, block.bn2)
        block.bn2 = nn.Identity()
        if block.downsample is not None:
            block.downsample[0] = fuse(block.downsample[0], block.downsample[1])
            block.downsample[1] = nn.Identity()
    return result


def _selected_convs(model, workflow):
    # PTQ applies to the residual transform Conv1 and Conv2 (both factors if
    # present). Downsample, stem and classifier stay float in the first G1.
    paths = []
    for region in workflow.ptq_regions:
        paths.append(region + ".conv1")
        block = model.get_submodule(region)
        paths.extend((region + ".conv2.0", region + ".conv2.1")
                     if type(block.conv2) is nn.Sequential else (region + ".conv2",))
    return tuple(paths)


def _conv_nodes(graph, selected_paths):
    """Bind exported nodes to exact named weight initializers, never substring scopes."""
    initializers = {value.name for value in graph.graph.initializer}
    matches = {}
    for node in graph.graph.node:
        if node.op_type != "Conv":
            continue
        for path in selected_paths:
            if len(node.input) > 1 and node.input[1] == path + ".weight" and node.input[1] in initializers:
                if path in matches:
                    raise BackendContractError("AmbiguousConv", "One selected Conv initializer has multiple consumers")
                matches[path] = node.name
    if set(matches) != set(selected_paths) or len(set(matches.values())) != len(matches):
        raise BackendContractError("MissingConv", "Every selected Conv must retain its exact exporter weight boundary")
    return matches


def inspect_qdq(graph, selected_nodes):
    """Reject hidden float Conv inputs or unsigned/nonzero-point substitutions."""
    import onnx
    nodes = {node.name: node for node in graph.graph.node}
    producers = {output: node for node in graph.graph.node for output in node.output}
    initializers = {item.name: item for item in graph.graph.initializer}
    checked = []
    for name in selected_nodes:
        conv = nodes.get(name)
        if conv is None or conv.op_type != "Conv":
            raise BackendContractError("MissingQuantizedConv", name)
        if len(conv.input) > 2:
            bias_producer = producers.get(conv.input[2])
            if bias_producer is not None and bias_producer.op_type == "DequantizeLinear":
                raise BackendContractError("QuantizedBias", "Conv bias must retain its float32 representation")
        for index in (0, 1):
            dq = producers.get(conv.input[index])
            if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) != 3:
                raise BackendContractError("HiddenFloatFallback", name + ": declared INT8 Conv lacks Q/DQ")
            zp = initializers.get(dq.input[2])
            scale = initializers.get(dq.input[1])
            if zp is None or zp.data_type != onnx.TensorProto.INT8 or scale is None:
                raise BackendContractError("QuantizationFormat", "Symmetric signed INT8 Q/DQ required")
            if np.any(onnx.numpy_helper.to_array(zp) != 0):
                raise BackendContractError("QuantizationFormat", "TensorRT symmetric zero points required")
            scales = onnx.numpy_helper.to_array(scale)
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise BackendContractError("QuantizationScale", "Positive finite calibration scales required")
            if index == 1:
                weight = initializers.get(dq.input[0])
                if weight is None or weight.data_type != onnx.TensorProto.INT8:
                    raise BackendContractError("FloatWeight", "Selected Conv must persist signed INT8 weights")
                axis = next((attribute.i for attribute in dq.attribute if attribute.name == "axis"), 1)
                if axis != 0 or scales.ndim != 1 or scales.shape[0] != weight.dims[0]:
                    raise BackendContractError("QuantizationAxis", "Conv weights require output-channel scales on axis zero")
            else:
                q = producers.get(dq.input[0])
                if q is None or q.op_type != "QuantizeLinear":
                    raise BackendContractError("FloatActivation", "Selected activation must cross explicit quantization")
                if tuple(q.input[1:]) != tuple(dq.input[1:]):
                    raise BackendContractError("QuantizationCoordinates", "Activation Q/DQ scale and zero point must agree")
        checked.append(name)
    return tuple(checked)


def _ort_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxRuntimeDeployment:
    def __init__(self, path, manifest):
        self.session = _ort_session(path)
        self.manifest = manifest

    def __call__(self, inputs):
        tensor = isinstance(inputs, torch.Tensor)
        value = inputs.detach().cpu().numpy() if tensor else np.asarray(inputs)
        _check_inputs(value, self.manifest["backend_spec"]["max_batch_size"])
        result = self.session.run(None, {"images": np.ascontiguousarray(value)})[0]
        if not np.isfinite(result).all():
            raise BackendContractError("OutputOverflow", "Loaded deployment produced nonfinite outputs")
        return torch.from_numpy(result) if tensor else result


def _check_inputs(value, max_batch):
    if (value.dtype != np.float32 or value.ndim != 4 or tuple(value.shape[1:]) != (3, 32, 32)
            or not 1 <= value.shape[0] <= max_batch or not np.isfinite(value).all()):
        raise BackendContractError("InputContract", "Finite bounded float32 CIFAR NCHW input required")


def _check_recovered(model, calibration, workflow):
    failure = validate_workflow(workflow)
    if failure is not None:
        raise BackendContractError("UncheckedWorkflow", str(failure))
    validate_cv_model(model, cpu=True)
    if any(layer.training for layer in model.modules()):
        raise BackendContractError("UnfrozenBatchNorm", "Freeze all recovered evaluation modes before export")
    if _calibration(calibration) != workflow.space.structural_space.calibration.fingerprint:
        raise BackendContractError("StaleCalibration", "Root calibration data changed after workflow compilation")
    evidence = getattr(model, "_fedcore_method_evidence", {})
    if (evidence.get("heterogeneous_workflow_fingerprint") != workflow.fingerprint
            or evidence.get("source_fingerprint") != workflow.space.structural_space.source.fingerprint
            or evidence.get("structural_representation_fingerprint") != fingerprint(model.representation_config())):
        raise BackendContractError("StaleCandidate", "Recovered float model must retain checked source and channel provenance")
    for block in workflow.structural_workflow.blocks:
        index = BLOCK_PATHS.index(block.gene.path)
        if model.block_widths[index] != block.internal_width or model.conv2_ranks[index] != block.rank:
            raise BackendContractError("RepresentationMismatch", "Recovered shape/rank differs from checked CV workflow")


def export_onnx_qdq(model, calibration, workflow, destination, *, spec=BackendSpec(), recovery_evidence=None):
    """Export the actual recovered candidate; training remains caller-owned.

    A separate ONNX and independent CPU session verify folding/export. INT8 is
    calibrated on this candidate, with exact selected Conv boundaries. TensorRT
    additionally requires inspectable INT8 kernels and independent engine load.
    """
    try:
        if type(spec) is not BackendSpec:
            raise BackendContractError("InvalidBackend", "A typed backend profile is required")
        cuda_binding = None
        if spec.runtime == 'tensorrt':
            from fedcore.experiments.heterogeneous_device import require_cuda_binding
            cuda_binding = require_cuda_binding(spec.device_uuid)
        # Own the exact calibration bytes before checking their workflow binding.
        # A caller mutation between validation and PTQ must never become a new
        # implicitly accepted calibration identity.
        _calibration(calibration)
        calibration_source = calibration
        calibration = calibration.detach().clone(memory_format=torch.contiguous_format)
        _check_recovered(model, calibration, workflow)
        calibration_identity = workflow.space.structural_space.calibration.fingerprint
        import onnx
        import onnxruntime as ort
        output = Path(destination).resolve()
        if output.exists() and any(output.iterdir()):
            raise BackendContractError("DestinationExists", "Export destination must be empty; existing artifacts are immutable")
        output.mkdir(parents=True, exist_ok=True)
        source_inventory = capture_inventory(model)
        recovered_identity = inventory_fingerprint(source_inventory)
        peak = workflow.space.structural_space.options.max_peak_bytes
        storage_guard(6 * inventory_cost(source_inventory, "bytes")
                      + calibration.numel() * calibration.element_size(), peak)
        folded = _fold_bn(model)
        selected_paths = _selected_convs(folded, workflow)
        with tempfile.TemporaryDirectory(prefix="fedcore-heterogeneous-") as temporary:
            float_path = Path(temporary) / "float.onnx"
            probe = calibration[:min(2, calibration.shape[0], spec.max_batch_size)].contiguous()
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.onnx.export(folded, probe, str(float_path), opset_version=spec.opset,
                                  input_names=["images"], output_names=["logits"],
                                  dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                                  do_constant_folding=False, keep_initializers_as_inputs=False)
                reference = model(probe).detach().numpy()
            graph = onnx.load(str(float_path))
            onnx.checker.check_model(graph)
            selected = _conv_nodes(graph, selected_paths)
            session = _ort_session(float_path)
            actual = session.run(None, {"images": probe.numpy()})[0]
            if not np.allclose(reference, actual, rtol=spec.float_rtol, atol=spec.float_atol):
                raise BackendContractError("FloatExportMismatch", "BN-folded float ONNX differs from the recovered model")
            model_path = output / "model.onnx"
            if selected:
                from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
                class Reader(CalibrationDataReader):
                    def __init__(self):
                        self.index = 0
                    def get_next(self):
                        if self.index >= calibration.shape[0]:
                            return None
                        start = self.index
                        self.index += min(spec.calibration_batch_size, spec.max_batch_size)
                        return {"images": calibration[start:self.index].contiguous().numpy()}
                quantize_static(str(float_path), str(model_path), Reader(), quant_format=QuantFormat.QDQ,
                                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                                per_channel=True, nodes_to_quantize=list(selected.values()), op_types_to_quantize=["Conv"],
                                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                               "CalibTensorRangeSymmetric": True, "QDQKeepRemovableActivations": True,
                                               "QuantizeBias": False})
            else:
                model_path.write_bytes(float_path.read_bytes())
        graph = onnx.load(str(model_path), load_external_data=False)
        storage_guard(0, peak)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise BackendContractError("ExternalWeights", "First deployment profile requires self-contained ONNX")
        onnx.checker.check_model(graph)
        coverage = inspect_qdq(graph, selected.values())
        loaded = _ort_session(model_path)
        qoutput = loaded.run(None, {"images": probe.numpy()})[0]
        if qoutput.shape != reference.shape or not np.isfinite(qoutput).all():
            raise BackendContractError("OutputContract", "Independent Q/DQ session must produce finite classifier outputs")
        manifest = {"schema": "heterogeneous_deployment_v1", "backend": spec.runtime,
                    "cuda_binding": cuda_binding,
                    "backend_spec": asdict(spec), "workflow_fingerprint": workflow.fingerprint,
                    "source_M0": workflow.space.structural_space.source.fingerprint,
                    "recovered_float_fingerprint": recovered_identity,
                    "calibration_fingerprint": calibration_identity,
                    "calibration_semantics": "current_recovered_candidate_after_BN_freeze_and_folding",
                    "representation": model.representation_config(),
                    "selected_conv_paths": list(selected_paths), "selected_conv_nodes": selected,
                    "qdq_verified_nodes": list(coverage), "stem_head_downsample_policy": "float32",
                    "conv_bias_policy": "float32_no_INT32_dequantize",
                    "float_export_max_abs_error": float(np.max(np.abs(reference - actual))),
                    "quantized_probe_max_abs_error": float(np.max(np.abs(reference - qoutput))),
                    "float_tensor_bytes": inventory_cost(source_inventory, "bytes"),
                    "recovery": recovery_evidence, "versions": {"torch": str(torch.__version__), "onnx": onnx.__version__, "onnxruntime": ort.__version__},
                    "hardware_validation": "CPU_ONNX_Runtime_only"}
        primary = model_path
        if spec.runtime == "tensorrt":
            primary, details = _compile_tensorrt(model_path, output, spec, tuple(selected.values()))
            manifest.update(details)
        manifest["required_files"] = [primary.name, "manifest.json"]
        manifest["model_sha256"] = _sha(primary)
        manifest_path = output / "manifest.json"
        # Load the self-contained runtime afresh. CPU output equality checks
        # serialization, not agreement between quantized and float arithmetic.
        runtime = (OnnxRuntimeDeployment(primary, manifest) if spec.runtime == 'onnxruntime'
                   else TensorRTDeployment(primary, manifest))
        if spec.runtime == "onnxruntime":
            if not np.array_equal(runtime(probe.numpy()), qoutput):
                raise BackendContractError("ReloadMismatch", "Fresh deployment load changed Q/DQ outputs")
        else:
            hardware_output = runtime(probe.numpy())
            if not np.allclose(hardware_output, qoutput, rtol=spec.backend_rtol, atol=spec.backend_atol):
                raise BackendContractError("TensorRTOutputMismatch", "Reloaded TensorRT engine differs from CPU Q/DQ reference")
            manifest["hardware_validation"] = "TensorRT_engine_reloaded_kernel_inspected_probe_verified_not_benchmarked"
            manifest["backend_probe_max_abs_error"] = float(np.max(np.abs(hardware_output - qoutput)))
        if inventory_fingerprint(capture_inventory(model)) != recovered_identity:
            raise BackendContractError("SourceMutation", "Export changed the caller-owned recovered candidate")
        if (_calibration(calibration) != calibration_identity
                or _calibration(calibration_source) != calibration_identity):
            raise BackendContractError("StaleCalibration", "Calibration changed during export or independent reload")
        # Publish the self-contained loader entry only after every postcondition.
        # Failed exports can retain diagnostic files, but are not loadable packages.
        _json(manifest_path, manifest)
        return DeploymentArtifact(str(primary), str(manifest_path), spec.runtime, deployment_size(manifest_path), manifest)
    except BackendContractError as error:
        return StructuralInvalid((Violation(error.code, "heterogeneous.export", str(error)),))
    except ImportError as error:
        return StructuralInvalid((Violation("MissingOptionalDependency", "heterogeneous.export", str(error)),))
    except MemoryError as error:
        return ResourceFailure("ExportMemory", str(error))
    except CandidateStorageError as error:
        return ResourceFailure(error.violations[0].code, str(error))
    except (ValueError, RuntimeError, TypeError, KeyError, OSError) as error:
        return NumericalFailure(type(error).__name__, str(error))


def _compile_tensorrt(model_path, output, spec, selected_nodes):
    """Lazy optional hardware compilation. Ambiguous kernel coverage fails closed."""
    import tensorrt as trt
    _check_tensorrt_version(trt)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model_path.read_bytes()):
        raise BackendContractError("TensorRTParse", "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, spec.workspace_bytes)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 32, 32), (spec.opt_batch_size, 3, 32, 32), (spec.max_batch_size, 3, 32, 32))
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise BackendContractError("TensorRTBuild", "TensorRT did not produce an engine")
    engine_path = output / "model.engine"
    engine_path.write_bytes(bytes(engine_bytes))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise BackendContractError("TensorRTReload", "Independent serialized engine load failed")
    inspection_context = engine.create_execution_context()
    if inspection_context is None or not inspection_context.set_input_shape(
            "images", (spec.opt_batch_size, 3, 32, 32)):
        raise BackendContractError("TensorRTInspector", "A concrete input shape is required for kernel inspection")
    inspector = engine.create_engine_inspector()
    inspector.execution_context = inspection_context
    description = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    # Keep rejected inspection evidence for diagnosis. Only the deployment
    # manifest marks an accepted independently validated artifact.
    (output / "engine-inspector.json").write_text(description, encoding="utf-8")
    information = json.loads(description)
    layers = information if type(information) is list else information.get("Layers", [])
    if type(layers) is not list:
        raise BackendContractError("TensorRTInspector", "Inspectable kernel entries are required")
    for selected in selected_nodes:
        if not _confirmed_int8_layer(layers, selected):
            raise BackendContractError("TensorRTFloatFallback", "No confirmed INT8 kernel for " + selected)
    return engine_path, {"versions_tensorrt": trt.__version__, "hardware_validation": "TensorRT_engine_reloaded_kernel_inspected_not_benchmarked",
                         "tensorrt_int8_verified_nodes": list(selected_nodes), "inspector_sha256": _sha(output / "engine-inspector.json")}


def _check_tensorrt_version(trt):
    if str(trt.__version__).split(".", 1)[0] != "10":
        raise BackendContractError("TensorRTVersion", "This named-I/O backend supports TensorRT 10 only")


def _confirmed_int8_layer(layers, selected):
    """Exact ONNX provenance plus integer Conv input and kernel/weight precision.

    Inspector versions which omit precision/constant information fail closed;
    a quantized neighbour or a matching substring is never kernel evidence.
    """
    def has_int8_type(value):
        if type(value) is str:
            return bool(re.search(r"(?<![A-Za-z0-9])int8(?![A-Za-z0-9])", value, re.IGNORECASE))
        if type(value) is dict:
            return any(has_int8_type(v) for k, v in value.items()
                       if k.lower() in ("type", "datatype", "format/datatype", "precision", "weightstype"))
        if type(value) is list:
            return any(has_int8_type(item) for item in value)
        return False
    for item in layers:
        if type(item) is not dict:
            continue
        metadata = str(item.get("Metadata", ""))
        origins = re.findall(r"\[ONNX Layer:\s*([^\]]+)\]", metadata)
        if selected not in tuple(origin.strip() for origin in origins):
            continue
        if "conv" not in str(item.get("LayerType", "")).lower():
            continue
        inputs = item.get("Inputs", [])
        # Explicit floating computation contradicts integer-kernel evidence;
        # quantized storage alone cannot override a reported float precision.
        if "Precision" in item and not has_int8_type(item["Precision"]):
            continue
        activation_int8 = type(inputs) is list and bool(inputs) and has_int8_type(inputs[0])
        kernel_int8 = has_int8_type(item.get("Precision", "")) or has_int8_type(item.get("Weights", {}))
        constant_int8 = type(inputs) is list and len(inputs) > 1 and has_int8_type(inputs[1])
        if activation_int8 and (kernel_int8 or constant_int8):
            return True
    return False


class TensorRTDeployment:
    """TensorRT 10 named-I/O runner; CUDA work occurs only when explicitly called."""
    def __init__(self, path, manifest):
        from fedcore.experiments.heterogeneous_device import require_cuda_binding
        require_cuda_binding(manifest['backend_spec']['device_uuid'])
        import tensorrt as trt
        _check_tensorrt_version(trt)
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise BackendContractError("TensorRTReload", "Serialized engine cannot load")
        self.context = self.engine.create_execution_context()
        if (self.context is None or self.engine.num_io_tensors != 2
                or self.engine.get_tensor_dtype("images") != trt.float32
                or self.engine.get_tensor_dtype("logits") != trt.float32):
            raise BackendContractError("TensorRTBoundary", "Exactly float32 images/logits boundaries required")
        self.manifest = manifest

    def __call__(self, inputs):
        if not torch.cuda.is_available():
            raise BackendContractError("MissingCUDA", "TensorRT inference requires an authorized CUDA device")
        value = inputs.detach().cpu().numpy() if isinstance(inputs, torch.Tensor) else np.asarray(inputs)
        _check_inputs(value, self.manifest["backend_spec"]["max_batch_size"])
        x = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        result = self.infer_device(x).cpu()
        if not bool(torch.isfinite(result).all()):
            raise BackendContractError("OutputOverflow", "Engine output is nonfinite")
        return result if isinstance(inputs, torch.Tensor) else result.numpy()

    def infer_device(self, x):
        """CUDA-resident path for authorized latency sessions; no CPU/NumPy copies.

        The caller owns synchronization, warmup and finite-input validation
        outside timed regions. Shape binding/allocation and execution remain
        visible work. This context is serial and must not be shared by threads.
        """
        if (not isinstance(x, torch.Tensor) or x.device.type != 'cuda'
                or x.dtype != torch.float32 or not x.is_contiguous() or x.ndim != 4
                or tuple(x.shape[1:]) != (3,32,32)
                or not 1 <= x.shape[0] <= self.manifest['backend_spec']['max_batch_size']):
            raise BackendContractError('InputContract', 'Contiguous bounded CUDA float32 CIFAR inputs required')
        if not self.context.set_input_shape("images", tuple(x.shape)):
            raise BackendContractError("TensorRTShape", "Input shape is outside the engine profile")
        shape = tuple(self.context.get_tensor_shape("logits"))
        if any(n < 1 for n in shape):
            raise BackendContractError("TensorRTShape", "Output dimensions remain unresolved")
        y = torch.empty(shape, dtype=torch.float32, device=x.device)
        self.context.set_tensor_address("images", x.data_ptr())
        self.context.set_tensor_address("logits", y.data_ptr())
        stream = torch.cuda.current_stream(x.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise BackendContractError("TensorRTExecution", "Engine execution failed")
        return y


def load_deployment(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _check_manifest(path, manifest)
    if manifest["backend"] == "onnxruntime":
        import onnx
        graph = onnx.load(str(path.parent / "model.onnx"), load_external_data=False)
        if manifest["qdq_verified_nodes"] != list(manifest["selected_conv_nodes"].values()):
            raise BackendContractError("InvalidCoverage", "Manifest must preserve complete selected Conv coverage")
        inspect_qdq(graph, manifest["qdq_verified_nodes"])
        return OnnxRuntimeDeployment(path.parent / "model.onnx", manifest)
    return TensorRTDeployment(path.parent / "model.engine", manifest)
