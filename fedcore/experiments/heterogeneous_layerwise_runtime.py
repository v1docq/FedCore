"""Prepared one-build layerwise worker; importing this module never uses CUDA.

This module does not launch itself. The fixed container driver must admit the
sole UUID-bound worker under its original OS deadline before calling it. Every
tap belongs to a rebuilt diagnostic graph, even if final-logit controls agree.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time

from . import heterogeneous_container as container
from . import heterogeneous_qonly_diagnostic as qonly
from . import heterogeneous_layerwise as layerwise
from .heterogeneous_pilot import PilotError


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _write_bytes(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush(); os.fsync(stream.fileno())


@dataclass
class Accounting:
    calls_attempted: int = 0
    calls_completed: int = 0
    row_visits_attempted: int = 0
    row_visits_completed: int = 0
    engine_builds_attempted: int = 0
    engine_builds_completed: int = 0
    tap_payload_bytes: int = 0
    calls_by_runtime: dict = field(default_factory=dict)


def load_prior_pairs(path, calibration_indices, *, expected_sha256):
    """Read exactly the frozen prior 2000-row diagnostic, with no new inference."""
    import numpy as np
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2 or qonly._sha(path) != expected_sha256:
        raise PilotError("Frozen prior numerical pairs hash or bounded size differs")
    positions, indices, references, actuals = [], [], [], []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream):
            if number >= 17 or len(line) > 1024**2:
                raise PilotError("Frozen prior batch sequence exceeds its bound")
            row = json.loads(line)
            start, stop = qonly.batch_ranges(2000, 128)[number]
            if (row.get("schema") != "heterogeneous-qonly-logit-pairs-v1"
                    or row.get("batch_number") != number
                    or row.get("calibration_positions") != list(range(start, stop))
                    or row.get("train_indices") != list(calibration_indices[start:stop])):
                raise PilotError("Frozen prior pairs have changed batch positions or calibration IDs")
            reference = np.asarray(row["cpu_qdq_logits"], dtype=np.float32)
            actual = np.asarray(row["tensorrt_logits"], dtype=np.float32)
            qonly.comparison(reference, actual)
            if len(reference) != stop - start:
                raise PilotError("Frozen prior logits have changed cardinality")
            positions.extend(row["calibration_positions"]); indices.extend(row["train_indices"])
            references.append(reference); actuals.append(actual)
    if positions != list(range(2000)) or tuple(indices) != calibration_indices:
        raise PilotError("Frozen prior pairs do not contain exactly the original calibration corpus")
    return np.concatenate(references), np.concatenate(actuals)


def replay_spans(case):
    if case == "partitioned":
        return ((0, 2), (2, 8))
    if case == "singletons":
        return tuple((i, i + 1) for i in range(8))
    if case == "whole":
        return ((0, 8),)
    raise ValueError("Unknown fixed replay case")


def _save_selected_inputs(root, inputs, selected_rows):
    """Preserve exact normalized input bytes before the first numerical call."""
    import numpy as np
    if (not isinstance(inputs, np.ndarray) or inputs.dtype != np.float32
            or inputs.shape != (8, 3, 32, 32) or len(selected_rows) != 8 or not np.isfinite(inputs).all()):
        raise PilotError("Exactly eight finite normalized float32 CIFAR inputs required")
    target = Path(root) / "selected-inputs.npy"
    with target.open("xb") as stream:
        np.save(stream, inputs, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    identity = dict(path=str(target), sha256=qonly._sha(target), bytes=target.stat().st_size,
        array_sha256=qonly._array_sha(inputs), shape=list(inputs.shape), dtype=str(inputs.dtype),
        selected_rows=[asdict(row) for row in selected_rows], source_calibration_fingerprint=qonly.CALIBRATION_SHA256,
        hash_encoding="array_sha256 uses NumPy shape/dtype prefix plus contiguous bytes; sha256 hashes the complete NPY file")
    container._publish(Path(root) / "selected-inputs.json", identity)
    return identity


def run_logit_replays(inputs, selected_rows, *, ort_runner, trt_runner, record):
    """Fixed 22 calls / 48 row visits, using eight distinct already selected rows."""
    import numpy as np
    if inputs.shape != (8, 3, 32, 32) or len(selected_rows) != 8:
        raise PilotError("The frozen eight-row CIFAR replay is required")
    result = {}
    for runtime_name, runner in (("original_ORT", ort_runner), ("original_TensorRT", trt_runner)):
        result[runtime_name] = {}
        for case in ("partitioned", "singletons", "whole"):
            outputs = [record(runtime_name, case, inputs[start:stop], selected_rows[start:stop], runner)
                       for start, stop in replay_spans(case)]
            result[runtime_name][case] = np.concatenate(outputs)
    return result


class _Recorder:
    def __init__(self, root, accounting, deadline, clock=time.monotonic):
        self.root, self.accounting, self.deadline, self.clock = Path(root), accounting, deadline, clock
        if any((self.root / name).exists() for name in ("layerwise-logits.jsonl", "layerwise-progress.jsonl")):
            raise PilotError("Diagnostic logits must start in a fresh run directory")

    def check(self):
        if self.clock() >= self.deadline:
            raise TimeoutError("Original layerwise deadline exhausted; no retry")

    def call(self, runtime_name, case, inputs, selected_rows, runner):
        import numpy as np
        self.check()
        inputs = np.ascontiguousarray(inputs)
        if (inputs.dtype != np.float32 or inputs.shape != (len(selected_rows), 3, 32, 32)
                or not 1 <= len(selected_rows) <= 8 or not np.isfinite(inputs).all()):
            raise PilotError("Finite bounded selected calibration batch required")
        if self.accounting.calls_attempted >= 28 or self.accounting.row_visits_attempted + len(inputs) > 72:
            raise PilotError("Fixed layerwise inference budget exceeded")
        digest = qonly._array_sha(inputs)
        call_id = self.accounting.calls_attempted
        self.accounting.calls_attempted += 1
        self.accounting.row_visits_attempted += len(inputs)
        began = self.clock()
        _append(self.root / "layerwise-progress.jsonl", dict(event="inference_started", at=began,
            call_id=call_id, runtime=runtime_name, replay_case=case,
            selected_rows=[asdict(row) for row in selected_rows], input_sha256=digest,
            input_shape=list(inputs.shape), original_deadline=self.deadline, accounting=asdict(self.accounting)))
        logits = runner(inputs)
        qonly.comparison(logits, logits)  # finite float32 N-by-10 boundary only
        if logits.shape != (len(inputs), 10) or qonly._array_sha(inputs) != digest:
            raise PilotError("Runtime changed the input or logit batch cardinality")
        _append(self.root / "layerwise-logits.jsonl", dict(call_id=call_id, runtime=runtime_name,
            replay_case=case, selected_rows=[asdict(row) for row in selected_rows],
            input_shape=list(inputs.shape), input_sha256=digest, input_dtype=str(inputs.dtype),
            logits=logits.tolist(), logits_sha256=qonly._array_sha(logits),
            began_at=began, ended_at=self.clock()))
        self.accounting.calls_completed += 1
        self.accounting.row_visits_completed += len(inputs)
        self.accounting.calls_by_runtime[runtime_name] = self.accounting.calls_by_runtime.get(runtime_name, 0) + 1
        _append(self.root / "layerwise-progress.jsonl", dict(event="inference_completed", at=self.clock(),
            call_id=call_id, runtime=runtime_name, logits_sha256=qonly._array_sha(logits),
            accounting=asdict(self.accounting)))
        self.check()
        return logits


def _ort_session_with_optimized_graph(model_path, optimized_path):
    """Same ORT settings as the original path; export only its optimized graph."""
    import onnxruntime as ort
    optimized_path = Path(optimized_path)
    if optimized_path.exists():
        raise PilotError("Optimized ORT graph output must be fresh")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.optimized_model_filepath = str(optimized_path)
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def optimized_graph_facts(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    return dict(path=str(path), sha256=qonly._sha(path), bytes=Path(path).stat().st_size,
        operators=dict(sorted(Counter((node.domain + "::" if node.domain else "") + node.op_type
            for node in model.graph.node).items())),
        interpretation="actual ORT optimized graph; Conv storage/operator counts alone do not prove executed kernel precision")


def quantization_parameter_facts(model):
    """Preserve existing Q/DQ scales/zero points, without quantizing or calibrating."""
    import numpy as np
    from onnx import numpy_helper
    initializers = {value.name: value for value in model.graph.initializer}
    parameters, nodes, elements = {}, [], 0
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if len(node.input) != 3 or any(name not in initializers for name in node.input[1:]):
            raise PilotError("Saved Q/DQ parameters must be explicit constant scale/zero-point initializers")
        for name in node.input[1:]:
            if name not in parameters:
                value = numpy_helper.to_array(initializers[name])
                if value.size > 512 or value.ndim > 1 or not np.isfinite(value).all():
                    raise PilotError("Saved Q/DQ parameter dimensions exceed the diagnostic bound")
                elements += value.size
                if elements > 65536:
                    raise PilotError("Saved Q/DQ parameter export exceeds its total bound")
                parameters[name] = dict(shape=list(value.shape), dtype=str(value.dtype), values=value.reshape(-1).tolist(),
                    initializer_sha256=layerwise._digest(initializers[name].SerializeToString()))
        scale = numpy_helper.to_array(initializers[node.input[1]])
        zero = numpy_helper.to_array(initializers[node.input[2]])
        if scale.dtype != np.float32 or not bool((scale > 0).all()) or zero.dtype not in (np.dtype("int8"), np.dtype("uint8")):
            raise PilotError("Saved Q/DQ must retain positive float32 scales and integer zero points")
        nodes.append(dict(name=node.name, op_type=node.op_type, input_tensor=node.input[0],
            scale=node.input[1], zero_point=node.input[2], output_tensors=list(node.output),
            axis=next((a.i for a in node.attribute if a.name == "axis"), 1)))
    return dict(nodes=nodes, parameters=parameters, elements=elements, quantization_calibrations=0)


class NamedTensorRTRunner:
    """Fresh context, nondefault stream and explicit serial named-I/O binding.

    This path deliberately bypasses TensorRTDeployment.infer_device. No shared
    context, inferred physical buffer layout, or default-stream dependency is
    used. Output tensors must be float32 LINEAR graph boundaries.
    """
    def __init__(self, engine, trt, gpu_uuid, output_shapes):
        from .heterogeneous_device import require_cuda_binding
        require_cuda_binding(gpu_uuid)
        import torch
        self.engine, self.trt, self.output_shapes = engine, trt, dict(output_shapes)
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.Stream(device="cuda:0")
        names = tuple(engine.get_tensor_name(i) for i in range(engine.num_io_tensors))
        if self.context is None or set(names) != {"images", *self.output_shapes} or len(names) != len(self.output_shapes) + 1:
            raise PilotError("Exactly the planned named TensorRT I/O tensors are required")
        for name in names:
            expected_mode = trt.TensorIOMode.INPUT if name == "images" else trt.TensorIOMode.OUTPUT
            if (engine.get_tensor_mode(name) != expected_mode or engine.get_tensor_dtype(name) != trt.float32
                    or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                    or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE):
                raise PilotError("Independent TensorRT boundary must be DEVICE float32 LINEAR with exact I/O modes")

    def __call__(self, inputs):
        import numpy as np
        import torch
        value = np.ascontiguousarray(inputs)
        if value.dtype != np.float32 or value.ndim != 4 or tuple(value.shape[1:]) != (3, 32, 32) or not 1 <= len(value) <= 8:
            raise PilotError("Independent binding requires bounded float32 CIFAR inputs")
        if not np.isfinite(value).all():
            raise PilotError("Independent binding input is nonfinite")
        with torch.cuda.stream(self.stream):
            x = torch.from_numpy(value).to("cuda:0", non_blocking=False)
            self.stream.synchronize()
            if not self.context.set_input_shape("images", tuple(value.shape)):
                raise PilotError("Independent binding input shape rejected")
            buffers = {"images": x}
            for name, sample_shape in self.output_shapes.items():
                shape = tuple(self.context.get_tensor_shape(name))
                if shape != (len(value), *sample_shape):
                    raise PilotError("Independent binding output shape differs from plan")
                buffers[name] = torch.empty(shape, dtype=torch.float32, device="cuda:0")
            for name, tensor in buffers.items():
                address = tensor.data_ptr()
                if (not self.context.set_tensor_address(name, address)
                        or self.context.get_tensor_address(name) != address):
                    raise PilotError("Independent binding tensor address was not retained")
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise PilotError("Independent TensorRT execution failed")
            self.stream.synchronize()
            result = {name: buffers[name].cpu().numpy() for name in self.output_shapes}
        if any(not np.isfinite(array).all() for array in result.values()):
            raise PilotError("Independent TensorRT output is nonfinite")
        return result


def _build_instrumented_engine(onnx_path, root, accounting, check):
    """Exactly one separately saved diagnostic engine with the frozen build flags."""
    import tensorrt as trt
    if accounting.engine_builds_attempted or trt.__version__ != "10.0.1":
        raise PilotError("Only one TensorRT 10.0.1 diagnostic build is permitted")
    check()
    accounting.engine_builds_attempted += 1
    _append(Path(root) / "layerwise-progress.jsonl", dict(event="engine_build_started", at=time.monotonic(),
        onnx_sha256=qonly._sha(onnx_path), accounting=asdict(accounting), max_builds=1,
        profile_shapes=[[1, 3, 32, 32], [128, 3, 32, 32], [128, 3, 32, 32]],
        tf32=False, workspace_bytes=1024**3))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise PilotError("Instrumented ONNX parser failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 32, 32), (128, 3, 32, 32), (128, 3, 32, 32))
    config.add_optimization_profile(profile)
    raw = builder.build_serialized_network(network, config)
    if raw is None:
        raise PilotError("Instrumented TensorRT build did not produce an engine")
    _write_bytes(Path(root) / "instrumented.engine", bytes(raw))
    accounting.engine_builds_completed += 1
    _append(Path(root) / "layerwise-progress.jsonl", dict(event="engine_build_completed", at=time.monotonic(),
        engine_sha256=qonly._sha(Path(root) / "instrumented.engine"),
        engine_bytes=(Path(root) / "instrumented.engine").stat().st_size, accounting=asdict(accounting)))
    check()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(raw))
    if engine is None:
        raise PilotError("Instrumented TensorRT engine reload failed")
    context = engine.create_execution_context()
    if context is None or not context.set_input_shape("images", (128, 3, 32, 32)):
        raise PilotError("Instrumented inspector needs the original opt-batch shape")
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    _write_bytes(Path(root) / "instrumented-engine-inspector.json",
                 inspector.get_engine_information(trt.LayerInformationFormat.JSON).encode("utf-8"))
    provenance = {name: dict(path=str(Path(root) / name), sha256=qonly._sha(Path(root) / name),
                             bytes=(Path(root) / name).stat().st_size)
                  for name in ("instrumented.onnx", "instrumented.engine", "instrumented-engine-inspector.json")}
    container._publish(Path(root) / "instrumented-artifacts.json", provenance)
    _append(Path(root) / "layerwise-progress.jsonl", dict(event="instrumented_artifacts_verified", at=time.monotonic(),
        artifacts=provenance, accounting=asdict(accounting)))
    return engine, trt, (logger, runtime)


def _original_controls_gate(root, controls):
    """Preserve all original controls before deciding whether to build a copy."""
    container._publish(Path(root) / "original-controls.json", controls)
    if controls["independent_binding_vs_existing_TensorRT"]["within_original_tolerance"] is not True:
        raise PilotError("Independent binding disagrees under the unchanged original tolerance; no diagnostic engine built")


def _save_taps(root, batch_number, rows, ort_outputs, trt_outputs, plan, accounting):
    import numpy as np
    if set(ort_outputs) != {item.tensor for item in plan.landmarks} or set(trt_outputs) != set(ort_outputs):
        raise PilotError("Every planned tap must be captured from both diagnostic runtimes")
    arrays, summaries = {}, []
    for item in plan.landmarks:
        reference, actual = ort_outputs[item.tensor], trt_outputs[item.tensor]
        expected_shape = (len(rows), *item.shape[1:])
        if reference.shape != expected_shape or actual.shape != expected_shape:
            raise PilotError("Captured tap shape differs from the bounded plan")
        arrays["ort__" + item.name] = reference
        arrays["trt__" + item.name] = actual
        summaries.append(dict(batch_number=batch_number, landmark=item.name, tensor=item.tensor,
            selected_rows=[asdict(row) for row in rows], graph_provenance="rebuilt_diagnostic_graph",
            **layerwise.compare_tensors(reference, actual)))
    payload_bytes = sum(array.nbytes for array in arrays.values())
    if accounting.tap_payload_bytes + payload_bytes > layerwise.MAX_TAP_BYTES:
        raise PilotError("Combined ORT/TensorRT tap array payload exceeds 64 MiB")
    target = Path(root) / ("tap-batch-" + str(batch_number) + ".npz")
    with target.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    accounting.tap_payload_bytes += payload_bytes
    for row in summaries:
        _append(Path(root) / "tap-summaries.jsonl", row)
    return dict(path=str(target), sha256=qonly._sha(target), bytes=target.stat().st_size,
                array_payload_bytes=payload_bytes, batch_number=batch_number)


def run_layerwise_worker(plan, cfg, binding, boundary, config_sha):
    """Called only after the existing fixed driver's sole-worker admission."""
    import numpy as np
    root = Path(plan.run_directory)
    accounting = Accounting()
    deadline = plan.started_at + cfg.control_config().gpu_seconds - cfg.control_config().cleanup_seconds
    recorder = _Recorder(root, accounting, deadline)
    status, failure, before, after, static_plan, controls, optimized, tap_files = "failed", None, None, None, None, {}, {}, []
    versions, selected, instrumented_hash, parameter_export, instrumented_artifacts, input_artifact = {}, (), None, None, None, None
    try:
        if getattr(cfg, "profile", None) != "layerwise_v1" or cfg.rows != 8 or cfg.batch_size != 8:
            raise PilotError("Fixed layerwise_v1 eight-row profile required")
        container._boundary(root, cfg.control_config(), plan.started_at, "worker")
        before = qonly.artifact_snapshot(cfg)
        recorder.check()
        from .heterogeneous_data import CifarTrainingData
        from fedcore.algorithm.low_rank.cv_execution import _calibration
        from fedcore.algorithm.heterogeneous.backend import TensorRTDeployment
        import onnx
        data = CifarTrainingData(plan.data_root, split_seed=20261102)
        calibration = data.calibration_tensor()
        if (data.fingerprint != qonly.DATA_SHA256 or data.roles.sha256 != qonly.ROLES_SHA256
                or _calibration(calibration) != qonly.CALIBRATION_SHA256):
            raise PilotError("Original calibration corpus or frozen split differs")
        artifact = Path(cfg.artifact_directory)
        prior = artifact.parents[5] / "diagnostic-01" / "logit-pairs.jsonl"
        prior_ort, prior_trt = load_prior_pairs(prior, data.roles.calibration, expected_sha256=qonly.PRIOR_PAIRS_SHA256)
        selected = layerwise.select_calibration_rows(data.roles.calibration, prior_ort, prior_trt)
        if len(selected) != 8:
            raise PilotError("Frozen prior pairs did not yield first two plus six disagreement rows")
        inputs = np.ascontiguousarray(calibration.numpy()[[row.calibration_position for row in selected]])
        input_artifact = _save_selected_inputs(root, inputs, selected)
        original_model = onnx.load(str(artifact / "model.onnx"), load_external_data=False)
        static_plan = layerwise.build_layerwise_plan(original_model, (artifact / "engine-inspector.json").read_bytes(), selected,
            source_onnx_sha256=qonly.SAVED_SHA256["model.onnx"], source_engine_sha256=qonly.SAVED_SHA256["model.engine"],
            source_inspector_sha256=qonly.SAVED_SHA256["engine-inspector.json"])
        container._publish(root / "layerwise-plan.json", asdict(static_plan))
        container._publish(root / "quantization-parameters.json", quantization_parameter_facts(original_model))
        parameter_export = dict(path=str(root / "quantization-parameters.json"), sha256=qonly._sha(root / "quantization-parameters.json"))
        enriched = layerwise.instrument_onnx_copy(original_model, static_plan)
        _write_bytes(root / "instrumented.onnx", enriched.SerializeToString())
        instrumented_hash = qonly._sha(root / "instrumented.onnx")
        original_session = _ort_session_with_optimized_graph(artifact / "model.onnx", root / "original-ort-optimized.onnx")
        original_runtime = TensorRTDeployment(artifact / "model.engine", dict(diagnostic_only=True,
            backend_spec=dict(device_uuid=binding["declared_uuid"], max_batch_size=128)))
        versions = qonly._runtime_versions(original_runtime)
        original = run_logit_replays(inputs, selected,
            ort_runner=lambda x: original_session.run(["logits"], {"images": x})[0],
            trt_runner=original_runtime, record=recorder.call)
        independent = NamedTensorRTRunner(original_runtime.engine, original_runtime.trt, binding["declared_uuid"], {"logits": (10,)})
        low_outputs = [recorder.call("independent_binding_TensorRT", "partitioned", inputs[start:stop], selected[start:stop],
            lambda x: independent(x)["logits"]) for start, stop in replay_spans("partitioned")]
        low_logits = np.concatenate(low_outputs)
        controls["independent_binding_vs_existing_TensorRT"] = qonly.comparison(original["original_TensorRT"]["partitioned"], low_logits)
        for name, results in original.items():
            for case in ("singletons", "whole"):
                controls[name + "_" + case + "_vs_partitioned"] = qonly.comparison(results["partitioned"], results[case])
        _original_controls_gate(root, controls)
        recorder.check()
        enriched_session = _ort_session_with_optimized_graph(root / "instrumented.onnx", root / "instrumented-ort-optimized.onnx")
        optimized = {"original": optimized_graph_facts(root / "original-ort-optimized.onnx"),
                     "instrumented": optimized_graph_facts(root / "instrumented-ort-optimized.onnx")}
        engine, trt, engine_owners = _build_instrumented_engine(root / "instrumented.onnx", root, accounting, recorder.check)
        instrumented_artifacts = json.loads(container._read(root / "instrumented-artifacts.json"))
        runner = NamedTensorRTRunner(engine, trt, binding["declared_uuid"], {item.tensor: item.shape[1:] for item in static_plan.landmarks})
        output_names = [value.name for value in enriched.graph.output]
        enriched_ort_logits, enriched_trt_logits = [], []
        for number, (start, stop) in enumerate(replay_spans("partitioned")):
            captured = {}
            def ort_call(x):
                captured["ort"] = dict(zip(output_names, enriched_session.run(output_names, {"images": x})))
                return captured["ort"]["logits"]
            def trt_call(x):
                captured["trt"] = runner(x)
                return captured["trt"]["logits"]
            enriched_ort_logits.append(recorder.call("instrumented_ORT", "partitioned", inputs[start:stop], selected[start:stop], ort_call))
            enriched_trt_logits.append(recorder.call("instrumented_TensorRT", "partitioned", inputs[start:stop], selected[start:stop], trt_call))
            tap_files.append(_save_taps(root, number, selected[start:stop], captured["ort"], captured["trt"], static_plan, accounting))
        enriched_ort_logits, enriched_trt_logits = np.concatenate(enriched_ort_logits), np.concatenate(enriched_trt_logits)
        controls["enriched_ORT_vs_original_ORT"] = qonly.comparison(original["original_ORT"]["partitioned"], enriched_ort_logits)
        controls["instrumented_TensorRT_vs_original_TensorRT"] = qonly.comparison(original["original_TensorRT"]["partitioned"], enriched_trt_logits)
        controls["instrumented_ORT_vs_TensorRT"] = qonly.comparison(enriched_ort_logits, enriched_trt_logits)
        container._publish(root / "instrumentation-controls.json", {name: value for name, value in controls.items()
            if name in ("enriched_ORT_vs_original_ORT", "instrumented_TensorRT_vs_original_TensorRT", "instrumented_ORT_vs_TensorRT")})
        if (accounting.calls_completed != 28 or accounting.row_visits_completed != 72
                or accounting.engine_builds_completed != 1):
            raise PilotError("Fixed layerwise replay/build accounting differs")
        if _calibration(calibration) != qonly.CALIBRATION_SHA256:
            raise PilotError("Original calibration changed during diagnostics")
        after = qonly.artifact_snapshot(cfg)
        if before != after:
            raise PilotError("Frozen rejected artifacts or receipts changed")
        recorder.check()
        container._boundary(root, cfg.control_config(), plan.started_at, "worker")
        status = "complete"
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    receipt = dict(schema="heterogeneous-layerwise-diagnostic-v1", status=status, failure=failure,
        config=asdict(cfg), diagnostic_config_sha256=config_sha, started_at=plan.started_at,
        worker_deadline=deadline, hard_deadline=plan.started_at + cfg.control_config().gpu_seconds,
        ended_at=time.monotonic(), cuda_binding=binding, boundary=boundary, accounting=asdict(accounting),
        selected_rows=[asdict(row) for row in selected], unique_calibration_rows=len(selected), versions=versions,
        selected_input_artifact=input_artifact,
        original_artifact_status="rejected_no_manifest", deployment_accepted=False,
        artifacts_before=before, artifacts_after=after, controls=controls, tap_files=tap_files,
        optimized_graph_exports=optimized, onnx_instrumentation=dict(copies=1 if instrumented_hash else 0,
            sha256=instrumented_hash, source_weights_changed=False),
        instrumented_artifacts=instrumented_artifacts,
        quantization_parameter_export=parameter_export,
        prior_pairs_sha256=qonly.PRIOR_PAIRS_SHA256, data_sha256=qonly.DATA_SHA256,
        roles_sha256=qonly.ROLES_SHA256, calibration_fingerprint=qonly.CALIBRATION_SHA256,
        split_seed=20261102, backend_rtol=qonly.RTOL, backend_atol=qonly.ATOL,
        tap_provenance="rebuilt diagnostic graph; final-logit agreement cannot prove original internal equivalence",
        required_controls=list(layerwise.REQUIRED_CONTROLS), sgd_attempted=0, sgd_completed=0,
        trained_model_exports=0, quantization_calibrations=0, searches=0, Vsearch_opened=False,
        Vreport_opened=False, official_test_opened=False, labels_evaluated=False, automatic_retries=0,
        host_pid_identity_proven=False, module_sha256=qonly._sha(__file__),
        static_plan_module_sha256=qonly._sha(layerwise.__file__),
        interpretation="calibration numerical diagnostics only; no deployment acceptance, quality or latency claim")
    container._publish(root / "diagnostic-receipt.json", receipt)
    return 0 if status == "complete" else 1
