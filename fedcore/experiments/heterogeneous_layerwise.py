"""Bounded CPU planning for a separate diagnostic copy of the Q-only graph.

No runtime, engine builder, data provider or GPU is imported here. Prefix cuts
are dependency metadata, not a request to build multiple engines. Instrumenting
outputs can change fusion/tactics, so final-logit controls are mandatory before
attributing an observed difference to the saved original engine.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re


MAX_ROWS = 8
MAX_LANDMARKS = 12
MAX_TAP_BYTES = 64 * 1024**2
MAX_TENSOR_ELEMENTS = 524288
MAX_MODEL_BYTES = 32 * 1024**2
MAX_GRAPH_ITEMS = 512
RTOL, ATOL = .02, .02
LANDMARK_NODES = (("stem", "/relu/Relu"),) + tuple(
    (f"layer{stage}.{block}", f"/layer{stage}/layer{stage}.{block}/relu_1/Relu")
    for stage in range(1, 5) for block in range(2)
) + (("pool", "/avgpool/GlobalAveragePool"), ("flatten", "/Flatten"), ("logits", "/fc/Gemm"))
REQUIRED_CONTROLS = (
    "enriched_ORT_logits_vs_original_ORT_logits_same_inputs_and_batch",
    "instrumented_TensorRT_logits_vs_original_saved_engine_logits_same_inputs_and_batch",
)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _hash_value(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("An explicit SHA-256 identity is required")
    return value


@dataclass(frozen=True)
class SelectedRow:
    calibration_position: int
    train_index: int
    reason: str
    ort_top2_margin: float
    ort_argmax: int
    trt_argmax: int

    def __post_init__(self):
        if (type(self.calibration_position) is not int or not 0 <= self.calibration_position < 2000
                or type(self.train_index) is not int or not 0 <= self.train_index < 50000
                or self.reason not in ("original_first_two", "largest_ORT_margin_disagreement")
                or type(self.ort_top2_margin) not in (int, float)
                or not math.isfinite(self.ort_top2_margin) or self.ort_top2_margin < 0
                or any(type(v) is not int or not 0 <= v < 10 for v in (self.ort_argmax, self.trt_argmax))
                or self.reason == "largest_ORT_margin_disagreement" and self.ort_argmax == self.trt_argmax):
            raise ValueError("A calibration row with finite prior logit evidence is required")


def select_calibration_rows(calibration_indices, ort_logits, trt_logits, *, max_rows=MAX_ROWS):
    """Select first two plus descending ORT top-two margins among disagreements.

    This interprets already saved logits only. It consumes no labels and runs
    no model. Ties use calibration position, and the first two are never repeated.
    """
    import numpy as np
    if (type(max_rows) is not int or not 2 <= max_rows <= MAX_ROWS
            or type(calibration_indices) is not tuple or len(calibration_indices) != 2000
            or any(type(i) is not int or not 0 <= i < 50000 for i in calibration_indices)
            or calibration_indices != tuple(sorted(set(calibration_indices)))
            or any(not isinstance(a, np.ndarray) or a.shape != (2000, 10)
                   or a.dtype != np.float32 or not np.isfinite(a).all() for a in (ort_logits, trt_logits))):
        raise ValueError("Ordered calibration IDs and finite saved float32 2000-by-10 logits required")
    top = np.partition(ort_logits.astype(np.float64), -2, axis=1)[:, -2:]
    margins = top[:, 1] - top[:, 0]
    ort_argmax, trt_argmax = ort_logits.argmax(1), trt_logits.argmax(1)
    candidates = sorted((i for i in range(2, 2000) if ort_argmax[i] != trt_argmax[i]),
                        key=lambda i: (-margins[i], i))
    positions = (0, 1) + tuple(candidates[:max_rows - 2])
    return tuple(SelectedRow(i, calibration_indices[i],
        "original_first_two" if i < 2 else "largest_ORT_margin_disagreement",
        float(margins[i]), int(ort_argmax[i]), int(trt_argmax[i])) for i in positions)


@dataclass(frozen=True)
class InspectorMatch:
    layer_index: int
    layer_name: str
    layer_type: str
    onnx_origins: tuple[str, ...]
    relations: tuple[str, ...]
    output_tensors: tuple[str, ...]
    output_formats: tuple[str, ...]


@dataclass(frozen=True)
class Landmark:
    name: str
    producer: str
    tensor: str
    shape: tuple[int, ...]
    dtype: str
    tap_bytes: int
    ancestor_node_indices: tuple[int, ...]
    ancestor_node_names: tuple[str, ...]
    required_initializers: tuple[str, ...]
    cut_inputs: tuple[str, ...]
    quantized_companion_tensors: tuple[str, ...]
    inspector_status: str
    inspector_matches: tuple[InspectorMatch, ...]


@dataclass(frozen=True)
class LayerwisePlan:
    model_proto_sha256: str
    source_engine_sha256: str
    source_inspector_sha256: str
    selected_rows: tuple[SelectedRow, ...]
    landmarks: tuple[Landmark, ...]
    total_tap_bytes: int
    required_controls: tuple[str, ...] = REQUIRED_CONTROLS
    engine_bytes_verified: bool = False
    original_engine_internal_tensors_available: bool = False
    prefix_cuts_are_metadata_only: bool = True

    def __post_init__(self):
        for value in (self.model_proto_sha256, self.source_engine_sha256, self.source_inspector_sha256):
            _hash_value(value)
        _validate_selection(self.selected_rows)
        if (len(self.landmarks) != MAX_LANDMARKS
                or tuple(item.name for item in self.landmarks) != tuple(name for name, _ in LANDMARK_NODES)
                or len({item.tensor for item in self.landmarks}) != MAX_LANDMARKS
                or self.total_tap_bytes != sum(item.tap_bytes for item in self.landmarks)
                or not 0 < self.total_tap_bytes <= MAX_TAP_BYTES
                or any(item.dtype != "float32" or item.shape[0] != len(self.selected_rows)
                       or any(type(d) is not int or d < 1 for d in item.shape)
                       or math.prod(item.shape) > MAX_TENSOR_ELEMENTS
                       or item.tap_bytes != 4 * math.prod(item.shape) for item in self.landmarks)
                or self.required_controls != REQUIRED_CONTROLS
                or self.engine_bytes_verified is not False
                or self.original_engine_internal_tensors_available is not False
                or self.prefix_cuts_are_metadata_only is not True):
            raise ValueError("Twelve bounded float32 landmarks and explicit original-output controls required")


def _validate_selection(selected_rows):
    if (type(selected_rows) is not tuple or not 2 <= len(selected_rows) <= MAX_ROWS
            or any(type(row) is not SelectedRow for row in selected_rows)
            or tuple(row.calibration_position for row in selected_rows[:2]) != (0, 1)
            or any(row.reason != "original_first_two" for row in selected_rows[:2])
            or any(row.reason != "largest_ORT_margin_disagreement" for row in selected_rows[2:])
            or len({row.calibration_position for row in selected_rows}) != len(selected_rows)
            or len({row.train_index for row in selected_rows}) != len(selected_rows)):
        raise ValueError("First two plus at most six distinct calibration disagreement rows required")


def _inspector_layers(raw, expected_sha256):
    if type(raw) is str:
        raw = raw.encode("utf-8")
    if type(raw) is not bytes or len(raw) > 1024**2 or _digest(raw) != expected_sha256:
        raise ValueError("Bounded original inspector bytes with matching SHA-256 required")
    payload = json.loads(raw)
    layers = payload if type(payload) is list else payload.get("Layers") if type(payload) is dict else None
    if type(layers) is not list or not 1 <= len(layers) <= MAX_GRAPH_ITEMS:
        raise ValueError("Bounded inspector layer array required")
    for layer in layers:
        if (type(layer) is not dict or any(type(layer.get(key, "")) is not str
                or len(layer.get(key, "")) > 16384 for key in ("Name", "LayerType", "Metadata"))
                or type(layer.get("Outputs")) is not list or len(layer["Outputs"]) > 8
                or any(type(out) is not dict or type(out.get("Name")) is not str
                       or len(out["Name"]) > 4096
                       or type(out.get("Format/Datatype", "")) is not str for out in layer["Outputs"])):
            raise ValueError("Inspector names, metadata and outputs must retain their exact bounded structure")
    return layers


def _mapping(node_name, tensor, companions, layers):
    matches = []
    for index, layer in enumerate(layers):
        outputs = tuple(out["Name"] for out in layer["Outputs"])
        origins = tuple(re.findall(r"\[ONNX Layer:\s*([^\]]+)\]", layer.get("Metadata", "")))
        relations = tuple(label for label, present in (
            ("exact_tensor_output", tensor in outputs),
            ("quantized_companion_output", any(name in outputs for name in companions)),
            ("producer_in_fused_region", node_name in origins)) if present)
        if relations:
            matches.append(InspectorMatch(index, layer.get("Name", ""), layer.get("LayerType", ""),
                origins, relations, outputs, tuple(out.get("Format/Datatype", "") for out in layer["Outputs"])))
    status = ("unmapped" if not matches else "ambiguous_multiple_layers" if len(matches) > 1
              else "exact_tensor_output" if "exact_tensor_output" in matches[0].relations
              else "fused_or_quantized_region_only")
    return status, tuple(matches)


def _checked_graph(model):
    import onnx
    if (not isinstance(model, onnx.ModelProto) or len(model.SerializeToString()) > MAX_MODEL_BYTES
            or not 1 <= len(model.graph.node) <= MAX_GRAPH_ITEMS
            or len(model.graph.initializer) > MAX_GRAPH_ITEMS
            or tuple(value.name for value in model.graph.input) != ("images",)
            or tuple(value.name for value in model.graph.output) != ("logits",)
            or any(item.data_location == onnx.TensorProto.EXTERNAL for item in model.graph.initializer)
            or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for node in model.graph.node for a in node.attribute)):
        raise ValueError("Bounded self-contained flat images-to-logits ONNX graph required")
    onnx.checker.check_model(model)
    nodes, producers, available = {}, {}, {value.name for value in model.graph.input}
    available.update(item.name for item in model.graph.initializer)
    if len(available) != len(model.graph.input) + len(model.graph.initializer):
        raise ValueError("Graph inputs and initializers must have distinct names")
    for index, node in enumerate(model.graph.node):
        if (not node.name or node.name in nodes or any(name and name not in available for name in node.input)
                or not node.output or any(not name or name in available for name in node.output)):
            raise ValueError("Unique topologically ordered nodes and tensor producers required")
        nodes[node.name] = index
        for name in node.output:
            producers[name] = index
            available.add(name)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    values = {v.name: v for v in (*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output)}
    return nodes, producers, values


def _ancestors(model, tensor, producers):
    initializers = {item.name for item in model.graph.initializer}
    inputs = {item.name for item in model.graph.input}
    nodes, needed_initializers, needed_inputs = set(), set(), set()
    def visit(name):
        if not name:
            return
        if name in initializers:
            needed_initializers.add(name)
        elif name in inputs:
            needed_inputs.add(name)
        else:
            index = producers[name]
            if index not in nodes:
                nodes.add(index)
                for value in model.graph.node[index].input:
                    visit(value)
    visit(tensor)
    ordered = tuple(sorted(nodes))
    return ordered, tuple(model.graph.node[i].name for i in ordered), tuple(sorted(needed_initializers)), tuple(sorted(needed_inputs))


def build_layerwise_plan(model, inspector, selected_rows, *, source_onnx_sha256,
                         source_engine_sha256, source_inspector_sha256):
    """Plan twelve taps from original ModelProto and raw inspector bytes.

    ONNX/inspector hashes are verified here. The engine is not loaded or read;
    its supplied hash must be verified separately by any future owned launcher.
    Selected rows come from saved-logit selection, never quality evaluation.
    """
    for value in (source_onnx_sha256, source_engine_sha256, source_inspector_sha256):
        _hash_value(value)
    _validate_selection(selected_rows)
    if _digest(model.SerializeToString()) != source_onnx_sha256:
        raise ValueError("Original ONNX ModelProto SHA-256 differs")
    layers = _inspector_layers(inspector, source_inspector_sha256)
    nodes, producers, values = _checked_graph(model)
    landmarks = []
    for name, producer in LANDMARK_NODES:
        if producer not in nodes or len(model.graph.node[nodes[producer]].output) != 1:
            raise ValueError("A fixed coarse landmark producer is missing or has ambiguous outputs")
        node = model.graph.node[nodes[producer]]
        tensor = node.output[0]
        value = values.get(tensor)
        if value is None or value.type.tensor_type.elem_type != 1:
            raise ValueError("Every coarse landmark must have inferred float32 shape")
        dims = value.type.tensor_type.shape.dim
        if len(dims) < 2 or not dims[0].dim_param or any(d.dim_value < 1 for d in dims[1:]):
            raise ValueError("Landmark batch must be symbolic and all remaining dimensions fixed")
        shape = (len(selected_rows),) + tuple(d.dim_value for d in dims[1:])
        companions = tuple(out for other in model.graph.node if other.op_type == "QuantizeLinear"
                           and other.input[0] == tensor for out in other.output)
        status, matches = _mapping(producer, tensor, companions, layers)
        prefix_indices, prefix_names, weights, cut_inputs = _ancestors(model, tensor, producers)
        landmarks.append(Landmark(name, producer, tensor, shape, "float32", math.prod(shape) * 4,
            prefix_indices, prefix_names, weights, cut_inputs, companions, status, matches))
    return LayerwisePlan(source_onnx_sha256, source_engine_sha256, source_inspector_sha256,
        selected_rows, tuple(landmarks), sum(item.tap_bytes for item in landmarks))


def instrument_onnx_copy(model, plan):
    """Return a separate ONNX copy with twelve planned outputs, without file I/O.

    The original logits output is kept first and deduplicated. No runtime is
    loaded. A future runner must perform both required final-logit controls.
    """
    import onnx
    if type(plan) is not LayerwisePlan or _digest(model.SerializeToString()) != plan.model_proto_sha256:
        raise ValueError("Instrumentation requires the exact original graph and bounded plan")
    _, _, values = _checked_graph(model)
    result = deepcopy(model)
    existing = {value.name for value in result.graph.output}
    for item in plan.landmarks:
        if item.tensor not in existing:
            result.graph.output.add().CopyFrom(values[item.tensor])
            existing.add(item.tensor)
    onnx.checker.check_model(result)
    return result


def summarize_tensor(array):
    """Bounded finite float32 activation summary, no activation dump or runtime."""
    import numpy as np
    if (not isinstance(array, np.ndarray) or array.dtype != np.float32 or not array.size
            or array.size > MAX_TENSOR_ELEMENTS or not np.isfinite(array).all()):
        raise ValueError("Bounded finite float32 tensor required")
    contiguous = np.ascontiguousarray(array)
    data = contiguous.astype(np.float64)
    digest = hashlib.sha256(str((tuple(array.shape), str(array.dtype))).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return dict(shape=list(array.shape), dtype="float32", elements=array.size, bytes=array.nbytes,
        sha256=digest.hexdigest(), min=float(data.min()), max=float(data.max()), mean=float(data.mean()),
        std=float(data.std()), l2_norm=float(np.linalg.norm(data.reshape(-1))),
        zero_count=int(np.count_nonzero(data == 0)),
        hash_encoding="SHA256(str((tuple(shape), str(numpy_dtype))).encode() + contiguous_array_bytes)")


def compare_tensors(reference, actual):
    """Summarize errors; intermediate-tensor thresholds are descriptive only."""
    import numpy as np
    ref, observed = summarize_tensor(reference), summarize_tensor(actual)
    if reference.shape != actual.shape:
        raise ValueError("Matching landmark tensor shapes required")
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    normalized = difference / (ATOL + RTOL * np.abs(reference.astype(np.float64)))
    worst = np.argsort(-difference.reshape(-1), kind="stable")[:16]
    return dict(reference=ref, actual=observed, max_abs_error=float(difference.max()),
        mean_abs_error=float(difference.mean()), max_normalized_error=float(normalized.max()),
        violating_count=int(np.count_nonzero(~np.isclose(actual, reference, rtol=RTOL, atol=ATOL))),
        backend_rtol=RTOL, backend_atol=ATOL, tolerance_semantics="original logit tolerance; intermediate description only",
        worst_coordinates=[list(map(int, np.unravel_index(int(i), difference.shape))) for i in worst],
        worst_abs_errors=[float(difference.reshape(-1)[i]) for i in worst],
        deployment_accepted=False)
