"""Pure, bounded CPU arithmetic for the saved layer1.0 residual block only.

No files, CUDA, training, searches or data providers are used. The owner supplies
the eight saved ORT stem activations and runs the prepared functions under an
external wall/RSS limit with CPU thread environment variables set before import.
The binary64 Conv epilogue is a mathematical reference, not an ORT/TensorRT
kernel implementation. A literal ORT control must match the saved ORT block tap
bitwise before drawing causal conclusions from any counterfactual.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import numpy as np


SOURCE_ONNX_SHA256 = "fe2a9052f9ea093339bfbb7a24f796961440dc655591bc50ea5455ca74b19f2a"
STEM = "/relu/Relu_output_0"
PREFIX = "/layer1/layer1.0/"
STAGE_NODES = (
    ("stem_codes", STEM + "_QuantizeLinear"),
    ("stem_dq", STEM + "_DequantizeLinear"),
    ("conv1_pre_q", PREFIX + "conv1/Conv"),
    ("conv1_codes", PREFIX + "conv1/Conv_output_0_QuantizeLinear"),
    ("conv1_dq", PREFIX + "conv1/Conv_output_0_DequantizeLinear"),
    ("relu1", PREFIX + "relu/Relu"),
    ("conv2_input_codes", PREFIX + "relu/Relu_output_0_QuantizeLinear"),
    ("conv2_input_dq", PREFIX + "relu/Relu_output_0_DequantizeLinear"),
    ("conv2_pre_q", PREFIX + "conv2/Conv"),
    ("conv2_codes", PREFIX + "conv2/Conv_output_0_QuantizeLinear"),
    ("conv2_dq", PREFIX + "conv2/Conv_output_0_DequantizeLinear"),
    ("add", PREFIX + "Add"),
    ("block_out", PREFIX + "relu_1/Relu"),
)
VARIANTS = ("exact_acc_unrounded_bias", "exact_acc_int32_rounded_bias", "remove_only_conv1_output_qdq")
PRECISION_CONTRACT = (
    "source activation Q: FP32 divide, round-to-nearest-even, add zero point, clamp INT8; "
    "DQ/Relu/residual Add: FP32; Conv: exact int64 A -> FP64, multiply sx64 THEN sw64, "
    "then add stored FP32 bias converted to FP64; output Q: FP64 divide and round-even, DQ returns FP32; "
    "rounded-bias variant: RNE(b64/(sx64*sw64)) checked INT32, added to int64 A before scaling; "
    "removed-QDQ variant: only Conv1 output Q/DQ omitted, Conv1 result cast FP32 before Relu; "
    "no fused multiply-add; no downstream block-output Q/DQ; not bitexact ORT or TensorRT"
)


def _readonly(value):
    result = np.ascontiguousarray(value).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Quantization:
    scale: np.float32
    zero_point: np.int8

    def __post_init__(self):
        if (type(self.scale) is not np.float32 or not np.isfinite(self.scale) or self.scale <= 0
                or type(self.zero_point) is not np.int8):
            raise ValueError("Original scalar FP32 scale and INT8 zero point required")


@dataclass(frozen=True)
class ConvParameters:
    weights: np.ndarray
    weight_scale: np.ndarray
    weight_zero_point: np.ndarray
    bias: np.ndarray

    def __post_init__(self):
        w, scale, zero, bias = self.weights, self.weight_scale, self.weight_zero_point, self.bias
        if (not isinstance(w, np.ndarray) or w.dtype != np.int8 or w.ndim != 4
                or not 1 <= w.shape[0] <= 64 or not 1 <= w.shape[1] <= 64 or w.shape[2:] != (3, 3)
                or any(not isinstance(a, np.ndarray) or a.shape != (w.shape[0],) for a in (scale, zero, bias))
                or scale.dtype != np.float32 or zero.dtype != np.int8 or bias.dtype != np.float32
                or not np.isfinite(scale).all() or not (scale > 0).all() or not np.isfinite(bias).all()):
            raise ValueError("Bounded INT8 3x3 weights, per-output FP32 scales/INT8 zero points and FP32 bias required")
        for field in ("weights", "weight_scale", "weight_zero_point", "bias"):
            object.__setattr__(self, field, _readonly(getattr(self, field)))


@dataclass(frozen=True)
class BlockParameters:
    stem_q: Quantization
    conv1_q: Quantization
    conv2_input_q: Quantization
    conv2_q: Quantization
    conv1: ConvParameters
    conv2: ConvParameters
    source_sha256: str

    def __post_init__(self):
        if (any(type(q) is not Quantization for q in (self.stem_q, self.conv1_q, self.conv2_input_q, self.conv2_q))
                or any(type(c) is not ConvParameters for c in (self.conv1, self.conv2))
                or len(set(self.conv1.weights.shape[:2] + self.conv2.weights.shape[:2])) != 1
                or type(self.source_sha256) is not str or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("One equal-width residual block with an explicit source SHA-256 required")


@dataclass(frozen=True)
class VariantResult:
    name: str
    stages: dict[str, np.ndarray]
    proofs: dict
    precision_contract: str = PRECISION_CONTRACT
    actual_tensorrt_graph: bool = False


def quantize_fp32(value, quantization):
    """Literal declared FP32 activation-Q arithmetic, including signed ties."""
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Finite FP32 quantizer input required")
    with np.errstate(over="ignore"):
        divided = np.divide(value, quantization.scale, dtype=np.float32)
    rounded = np.rint(divided)
    shifted = np.add(rounded, np.float32(quantization.zero_point), dtype=np.float32)
    return np.clip(shifted, -128, 127).astype(np.int8)


def quantize_conv_real(value, quantization):
    """Declared mathematical Conv-output Q route; never silently casts to FP32."""
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("Finite FP64 Conv epilogue required for mathematical output Q")
    with np.errstate(over="ignore"):
        divided = np.divide(value, np.float64(quantization.scale), dtype=np.float64)
    return np.clip(np.rint(divided) + np.float64(quantization.zero_point), -128, 127).astype(np.int8)


def dequantize_fp32(codes, quantization):
    if not isinstance(codes, np.ndarray) or codes.dtype != np.int8:
        raise ValueError("INT8 source codes required")
    centered = codes.astype(np.int32) - int(quantization.zero_point)
    return np.multiply(centered.astype(np.float32), quantization.scale, dtype=np.float32)


def accumulator_proof(channels, input_zero_point, weight_zero_points):
    """Data-independent absolute bound, including every padded 3x3 term."""
    if (type(channels) is not int or not 1 <= channels <= 64 or type(input_zero_point) is not np.int8
            or not isinstance(weight_zero_points, np.ndarray) or weight_zero_points.dtype != np.int8
            or weight_zero_points.ndim != 1 or not 1 <= len(weight_zero_points) <= 64):
        raise ValueError("Bounded channels and original INT8 zero points required")
    input_bound = max(abs(-128 - int(input_zero_point)), abs(127 - int(input_zero_point)))
    weight_bounds = np.maximum(np.abs(-128 - weight_zero_points.astype(np.int64)),
                               np.abs(127 - weight_zero_points.astype(np.int64)))
    bounds = channels * 9 * input_bound * weight_bounds
    if bool((bounds >= np.iinfo(np.int64).max).any()):
        raise ValueError("Integer dot product lacks an int64 overflow proof")
    return dict(terms=channels * 9, max_abs_centered_input=input_bound,
        max_abs_centered_weights=weight_bounds.tolist(), per_channel_abs_bound=bounds.tolist(),
        max_abs_bound=int(bounds.max()), fits_int32=bool((bounds <= np.iinfo(np.int32).max).all()),
        int64_to_float32_exact_by_bound=bool((bounds <= 2**24).all()),
        int64_to_float64_exact_by_bound=bool((bounds <= 2**53).all()),
        padding="centered zero, equivalent to padding raw quantized values with input zero point")


def exact_int64_conv2d(codes, parameters, input_zero_point, *, tile_positions=64, tile_outputs=8, check=None):
    """Stride1/pad1 3x3 cross-correlation with exact bounded int64 dot products.

    At maximum geometry a patch tile is 64x576 int64 (288 KiB), an output
    tile is 8x64 int64 (4 KiB), and one centered padded image is 578 KiB.
    No output-channel x input-channel x kernel x image broadcast is created.
    The complete eight-row int64 result is 4 MiB. Temporary storage is bounded
    by declared geometry and tile sizes, independently of activation values.
    """
    if (not isinstance(codes, np.ndarray) or codes.dtype != np.int8 or codes.ndim != 4
            or not 1 <= codes.shape[0] <= 8 or codes.shape[1] != parameters.weights.shape[1]
            or any(not 1 <= d <= 32 for d in codes.shape[2:])
            or type(tile_positions) is not int or not 1 <= tile_positions <= 64
            or type(tile_outputs) is not int or not 1 <= tile_outputs <= 8
            or check is not None and not callable(check)):
        raise ValueError("Bounded INT8 NCHW block input and fixed-capacity tiles required")
    proof = accumulator_proof(codes.shape[1], input_zero_point, parameters.weight_zero_point)
    count, channels, height, width = codes.shape
    outputs = parameters.weights.shape[0]
    centered_weights = parameters.weights.astype(np.int64) - parameters.weight_zero_point.astype(np.int64)[:, None, None, None]
    weights = centered_weights.reshape(outputs, channels * 9)
    result = np.empty((count, outputs, height, width), dtype=np.int64)
    for sample in range(count):
        centered = codes[sample].astype(np.int64) - int(input_zero_point)
        padded = np.pad(centered, ((0, 0), (1, 1), (1, 1)), constant_values=0)
        for start in range(0, height * width, tile_positions):
            if check is not None:
                check()
            stop = min(start + tile_positions, height * width)
            patches = np.stack([padded[:, position // width:position // width + 3,
                position % width:position % width + 3].reshape(-1) for position in range(start, stop)])
            for out in range(0, outputs, tile_outputs):
                final = min(out + tile_outputs, outputs)
                result[sample, out:final].reshape(final - out, -1)[:, start:stop] = weights[out:final] @ patches.T
    if int(np.abs(result).max()) > proof["max_abs_bound"]:
        raise ArithmeticError("Computed accumulator exceeded the independent static bound")
    return result


def _bias_codes(parameters, input_q):
    product = np.multiply(np.float64(input_q.scale), parameters.weight_scale.astype(np.float64), dtype=np.float64)
    codes = np.rint(parameters.bias.astype(np.float64) / product)
    if (not np.isfinite(codes).all() or bool((codes < np.iinfo(np.int32).min).any())
            or bool((codes > np.iinfo(np.int32).max).any())):
        raise ValueError("Rounded bias does not fit the fixed INT32 counterfactual")
    return codes.astype(np.int64), product


def conv_epilogue(accumulator, parameters, input_q, *, rounded_bias=False):
    if not isinstance(accumulator, np.ndarray) or accumulator.dtype != np.int64 or accumulator.ndim != 4:
        raise ValueError("Exact int64 NCHW accumulator required")
    if accumulator.shape[1] != len(parameters.bias):
        raise ValueError("Accumulator output-channel shape differs")
    value = accumulator
    if rounded_bias:
        bias_codes, _ = _bias_codes(parameters, input_q)
        proof = accumulator_proof(parameters.weights.shape[1], input_q.zero_point, parameters.weight_zero_point)
        if max(proof["per_channel_abs_bound"]) + int(np.abs(bias_codes).max()) >= np.iinfo(np.int64).max:
            raise ValueError("Bias-added accumulator lacks an int64 overflow proof")
        value = np.add(value, bias_codes[None, :, None, None], dtype=np.int64)
    value = np.multiply(value.astype(np.float64), np.float64(input_q.scale), dtype=np.float64)
    value = np.multiply(value, parameters.weight_scale.astype(np.float64)[None, :, None, None], dtype=np.float64)
    if not rounded_bias:
        value = np.add(value, parameters.bias.astype(np.float64)[None, :, None, None], dtype=np.float64)
    return value


def bias_rounding_proof(parameters, input_q):
    codes, product = _bias_codes(parameters, input_q)
    reconstruction = codes.astype(np.float64) * product
    dot = accumulator_proof(parameters.weights.shape[1], input_q.zero_point, parameters.weight_zero_point)
    combined = np.asarray(dot["per_channel_abs_bound"], dtype=np.int64) + np.abs(codes)
    return dict(bias_codes=codes.tolist(), bias_code_dtype="int32_checked_stored_as_int64_for_addition",
        mathematical_abs_error_bound=(.5 * product).tolist(),
        computed_abs_reconstruction_error=np.abs(reconstruction - parameters.bias.astype(np.float64)).tolist(),
        bias_added_accumulator_abs_bound=combined.tolist(),
        bias_added_accumulator_fits_int32=bool((combined <= np.iinfo(np.int32).max).all()),
        bias_added_int64_to_float32_exact_by_bound=bool((combined <= 2**24).all()),
        rule="RNE(b64/(sx64*sw64)); add integer bias to A before sequential sx64 then sw64 scaling",
        limitation="half-step bound is real-arithmetic rounding bound; binary64 epilogue rounding is separate")


def evaluate_fixed_variants(stem_float32, parameters, *, check=None, on_convolution=None):
    """Exactly three predeclared variants, two integer convolutions each."""
    if on_convolution is not None and not callable(on_convolution):
        raise ValueError("Optional convolution accounting callback must be callable")
    if (not isinstance(stem_float32, np.ndarray) or stem_float32.dtype != np.float32
            or stem_float32.ndim != 4 or not 1 <= len(stem_float32) <= 8
            or stem_float32.shape[1] != parameters.conv1.weights.shape[1]
            or not np.isfinite(stem_float32).all()):
        raise ValueError("At most eight finite saved ORT stem activations required")
    result = {}
    for name in VARIANTS:
        stages = {}
        stages["stem_codes"] = quantize_fp32(stem_float32, parameters.stem_q)
        stages["stem_dq"] = dequantize_fp32(stages["stem_codes"], parameters.stem_q)
        if on_convolution is not None:
            on_convolution(name, 1, "started")
        accumulator1 = exact_int64_conv2d(stages["stem_codes"], parameters.conv1, parameters.stem_q.zero_point, check=check)
        if on_convolution is not None:
            on_convolution(name, 1, "completed")
        stages["conv1_accumulator"] = accumulator1
        stages["conv1_pre_q"] = conv_epilogue(accumulator1, parameters.conv1, parameters.stem_q,
            rounded_bias=name == "exact_acc_int32_rounded_bias")
        if name == "remove_only_conv1_output_qdq":
            relu_input = stages["conv1_pre_q"].astype(np.float32)
        else:
            stages["conv1_codes"] = quantize_conv_real(stages["conv1_pre_q"], parameters.conv1_q)
            stages["conv1_dq"] = dequantize_fp32(stages["conv1_codes"], parameters.conv1_q)
            relu_input = stages["conv1_dq"]
        stages["relu1"] = np.maximum(relu_input, np.float32(0))
        stages["conv2_input_codes"] = quantize_fp32(stages["relu1"], parameters.conv2_input_q)
        stages["conv2_input_dq"] = dequantize_fp32(stages["conv2_input_codes"], parameters.conv2_input_q)
        if on_convolution is not None:
            on_convolution(name, 2, "started")
        accumulator2 = exact_int64_conv2d(stages["conv2_input_codes"], parameters.conv2, parameters.conv2_input_q.zero_point, check=check)
        if on_convolution is not None:
            on_convolution(name, 2, "completed")
        stages["conv2_accumulator"] = accumulator2
        stages["conv2_pre_q"] = conv_epilogue(accumulator2, parameters.conv2, parameters.conv2_input_q,
            rounded_bias=name == "exact_acc_int32_rounded_bias")
        stages["conv2_codes"] = quantize_conv_real(stages["conv2_pre_q"], parameters.conv2_q)
        stages["conv2_dq"] = dequantize_fp32(stages["conv2_codes"], parameters.conv2_q)
        stages["add"] = np.add(stages["conv2_dq"], stages["stem_dq"], dtype=np.float32)
        stages["block_out"] = np.maximum(stages["add"], np.float32(0))
        proofs = {"conv1": accumulator_proof(parameters.conv1.weights.shape[1], parameters.stem_q.zero_point, parameters.conv1.weight_zero_point),
                  "conv2": accumulator_proof(parameters.conv2.weights.shape[1], parameters.conv2_input_q.zero_point, parameters.conv2.weight_zero_point)}
        if name == "exact_acc_int32_rounded_bias":
            proofs["conv1_bias"] = bias_rounding_proof(parameters.conv1, parameters.stem_q)
            proofs["conv2_bias"] = bias_rounding_proof(parameters.conv2, parameters.conv2_input_q)
        result[name] = VariantResult(name, stages, proofs)
    return result


def _source_nodes(model, expected_sha256):
    if hashlib.sha256(model.SerializeToString()).hexdigest() != expected_sha256:
        raise ValueError("Original frozen ONNX SHA-256 differs")
    nodes = {node.name: node for node in model.graph.node}
    if len(nodes) != len(model.graph.node):
        raise ValueError("Original graph contains ambiguous node names")
    for _, name in STAGE_NODES:
        if name not in nodes or len(nodes[name].output) != 1:
            raise ValueError("The fixed first-block stage is missing or ambiguous")
    return nodes


def parse_block_parameters(model, expected_sha256=SOURCE_ONNX_SHA256):
    """Read only existing source initializers and verify the exact shared-skip topology."""
    from onnx import numpy_helper
    nodes = _source_nodes(model, expected_sha256)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    stages = {key: nodes[name] for key, name in STAGE_NODES}
    tensors = {key: node.output[0] for key, node in stages.items()}
    def qpair(code_key, dq_key, source):
        q, dq = stages[code_key], stages[dq_key]
        if (q.op_type != "QuantizeLinear" or dq.op_type != "DequantizeLinear" or len(q.input) != 3
                or tuple(dq.input) != (tensors[code_key], *q.input[1:]) or q.input[0] != source):
            raise ValueError("Source Q/DQ ordering, scale identity or input changed")
        scale, zero = (initializers[name] for name in q.input[1:])
        if scale.shape != () or zero.shape != ():
            raise ValueError("First-block activation quantizers must be scalar")
        return Quantization(scale[()], zero[()])
    stem_q = qpair("stem_codes", "stem_dq", STEM)
    conv1_q = qpair("conv1_codes", "conv1_dq", tensors["conv1_pre_q"])
    conv2_input_q = qpair("conv2_input_codes", "conv2_input_dq", tensors["relu1"])
    conv2_q = qpair("conv2_codes", "conv2_dq", tensors["conv2_pre_q"])
    def conv(number, input_tensor):
        node = stages[f"conv{number}_pre_q"]
        weight_dq = nodes[f"layer1.0.conv{number}.weight_DequantizeLinear"]
        attrs = {a.name: list(a.ints) if a.ints else a.i for a in node.attribute}
        if (node.op_type != "Conv" or len(node.input) != 3 or node.input[0] != input_tensor
                or node.input[1] != weight_dq.output[0] or weight_dq.op_type != "DequantizeLinear"
                or tuple(a.i for a in weight_dq.attribute if a.name == "axis") != (0,)
                or attrs != dict(dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])):
            raise ValueError("First-block Conv must retain per-output weights, FP32 bias and fixed 3x3/pad1/stride1 geometry")
        return ConvParameters(*(initializers[name] for name in weight_dq.input), initializers[node.input[2]])
    if (stages["relu1"].op_type != "Relu" or tuple(stages["relu1"].input) != (tensors["conv1_dq"],)
            or stages["add"].op_type != "Add" or tuple(stages["add"].input) != (tensors["conv2_dq"], tensors["stem_dq"])
            or stages["block_out"].op_type != "Relu" or tuple(stages["block_out"].input) != (tensors["add"],)):
        raise ValueError("Residual Add must use shared stem DQ, with final Relu before downstream output Q")
    return BlockParameters(stem_q, conv1_q, conv2_input_q, conv2_q,
        conv(1, tensors["stem_dq"]), conv(2, tensors["conv2_input_dq"]), expected_sha256)


def extract_literal_block(model, parameters):
    """Independent in-memory literal graph with exactly thirteen stage outputs."""
    import onnx
    from onnx import helper
    nodes = _source_nodes(model, parameters.source_sha256)
    parse_block_parameters(model, parameters.source_sha256)
    producers = {value: node for node in model.graph.node for value in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    needed_nodes, needed_weights = set(), set()
    def visit(name):
        if name == STEM:
            return
        if name in initializers:
            needed_weights.add(name)
            return
        node = producers[name]
        if node.name not in needed_nodes:
            needed_nodes.add(node.name)
            for item in node.input:
                visit(item)
    visit(nodes[STAGE_NODES[-1][1]].output[0])
    channels = parameters.conv1.weights.shape[1]
    shape = ["batch", channels, "height", "width"]
    outputs = [helper.make_tensor_value_info(nodes[name].output[0],
        onnx.TensorProto.INT8 if key.endswith("codes") else onnx.TensorProto.FLOAT, shape) for key, name in STAGE_NODES]
    graph = helper.make_graph([deepcopy(node) for node in model.graph.node if node.name in needed_nodes],
        "literal_layer1_0_from_saved_stem", [helper.make_tensor_value_info(STEM, onnx.TensorProto.FLOAT, shape)], outputs,
        [deepcopy(item) for item in model.graph.initializer if item.name in needed_weights])
    copied = helper.make_model(graph, opset_imports=[deepcopy(item) for item in model.opset_import], ir_version=model.ir_version)
    onnx.checker.check_model(copied)
    return copied


def run_literal_control(block_model, stem_float32):
    """ONE actual ORT CPU call, not a hand-implemented floating Conv control."""
    import onnx
    import onnxruntime as ort
    if onnx.__version__ != "1.16.2" or ort.__version__ != "1.18.1":
        raise ValueError("Literal control requires frozen ONNX1.16.2 and ORT1.18.1")
    if (not isinstance(stem_float32, np.ndarray) or stem_float32.dtype != np.float32 or stem_float32.ndim != 4
            or not 1 <= stem_float32.shape[0] <= 8 or not 1 <= stem_float32.shape[1] <= 64
            or any(not 1 <= d <= 32 for d in stem_float32.shape[2:]) or not np.isfinite(stem_float32).all()
            or len(block_model.graph.output) != 13):
        raise ValueError("Bounded saved FP32 stem and thirteen-output literal graph required")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(block_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("Literal control must use only CPUExecutionProvider")
    values = session.run([value.name for value in block_model.graph.output], {STEM: np.ascontiguousarray(stem_float32)})
    return dict(zip((key for key, _ in STAGE_NODES), values))


def compare_with_saved(reference, actual):
    """No acceptance decision: record bitwise control and descriptive errors."""
    if (not isinstance(reference, np.ndarray) or not isinstance(actual, np.ndarray) or reference.shape != actual.shape
            or reference.size == 0 or reference.size > 8 * 64 * 32 * 32
            or reference.dtype.kind not in "fiu" or actual.dtype.kind not in "fiu"
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
        raise ValueError("Matching bounded finite saved/oracle arrays required")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    return dict(shape=list(reference.shape), reference_dtype=str(reference.dtype), actual_dtype=str(actual.dtype),
        bitwise_equal=reference.dtype == actual.dtype and reference.tobytes() == actual.tobytes(),
        equal_value_count=int(np.count_nonzero(reference == actual)), elements=int(reference.size),
        max_abs_error=float(absolute.max()), mean_abs_error=float(absolute.mean()),
        rms_error=float(np.sqrt(np.mean(difference * difference))),
        original_logit_tolerance_violations=int(np.count_nonzero(~np.isclose(actual, reference, rtol=.02, atol=.02))),
        tolerance_semantics="descriptive original logit tolerance only; no block-level acceptance or kernel equivalence claim")
