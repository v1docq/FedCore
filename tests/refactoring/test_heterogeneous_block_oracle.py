"""Small independent arithmetic/property tests; never replay the saved eight rows."""
import hashlib

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from fedcore.experiments import heterogeneous_block_oracle as oracle


def conv(weights, *, scale=1., bias=0., zero=0):
    outputs = weights.shape[0]
    return oracle.ConvParameters(weights.astype(np.int8), np.full(outputs, scale, np.float32),
        np.full(outputs, zero, np.int8), np.full(outputs, bias, np.float32))


def slow_conv(value, parameters, zero):
    n, channels, height, width = value.shape
    outputs = len(parameters.bias)
    result = np.zeros((n, outputs, height, width), np.int64)
    for sample in range(n):
        for out in range(outputs):
            for y in range(height):
                for x in range(width):
                    total = 0
                    for channel in range(channels):
                        for ky in range(3):
                            for kx in range(3):
                                iy, ix = y + ky - 1, x + kx - 1
                                if 0 <= iy < height and 0 <= ix < width:
                                    total += (int(value[sample, channel, iy, ix]) - int(zero)) * (
                                        int(parameters.weights[out, channel, ky, kx]) - int(parameters.weight_zero_point[out]))
                    result[sample, out, y, x] = total
    return result


@st.composite
def conv_cases(draw):
    n, channels, outputs = draw(st.integers(1, 2)), draw(st.integers(1, 3)), draw(st.integers(1, 3))
    height, width = draw(st.integers(1, 4)), draw(st.integers(1, 4))
    value = np.array(draw(st.lists(st.integers(-128, 127), min_size=n * channels * height * width,
        max_size=n * channels * height * width)), np.int8).reshape(n, channels, height, width)
    weights = np.array(draw(st.lists(st.integers(-128, 127), min_size=outputs * channels * 9,
        max_size=outputs * channels * 9)), np.int8).reshape(outputs, channels, 3, 3)
    zero = np.int8(draw(st.integers(-128, 127)))
    wz = np.array(draw(st.lists(st.integers(-128, 127), min_size=outputs, max_size=outputs)), np.int8)
    parameters = oracle.ConvParameters(weights, np.ones(outputs, np.float32), wz, np.zeros(outputs, np.float32))
    return value, parameters, zero


@given(conv_cases(), st.integers(1, 8), st.integers(1, 3))
@settings(max_examples=35, deadline=None)
def test_tiled_exact_dot_matches_independent_python_integer_loop_and_preserves_inputs(case, pixels, outputs):
    value, parameters, zero = case
    before = value.copy()
    actual = oracle.exact_int64_conv2d(value, parameters, zero, tile_positions=pixels, tile_outputs=outputs)
    np.testing.assert_array_equal(actual, slow_conv(value, parameters, zero))
    np.testing.assert_array_equal(value, before)
    assert actual.dtype == np.int64


def test_one_hot_asymmetric_channels_kernel_orientation_and_zero_point_padding():
    value = np.zeros((1, 2, 3, 4), np.int8)
    value[0, 1, 1, 2] = 7
    weights = np.zeros((3, 2, 3, 3), np.int8)
    weights[0, 1, 0, 2] = -3
    weights[1, 0, 2, 0] = 5  # another channel must remain zero
    weights[2, 1, 1, 1] = 2
    actual = oracle.exact_int64_conv2d(value, conv(weights), np.int8(0), tile_positions=1, tile_outputs=1)
    assert actual[0, 0, 2, 1] == -21 and actual[0, 2, 1, 2] == 14
    assert np.count_nonzero(actual[0, 1]) == 0
    np.testing.assert_array_equal(actual, slow_conv(value, conv(weights), np.int8(0)))
    zero_input = np.full((1, 2, 1, 1), 127, np.int8)
    assert not oracle.exact_int64_conv2d(zero_input, conv(weights), np.int8(127)).any()


def test_conservative_actual_geometry_bounds_and_adversarial_integer_extremes():
    proof = oracle.accumulator_proof(64, np.int8(0), np.zeros(64, np.int8))
    assert proof["terms"] == 576 and proof["max_abs_bound"] == 9437184
    assert proof["fits_int32"] and proof["int64_to_float32_exact_by_bound"]
    parameters = conv(np.full((1, 1, 3, 3), -128, np.int8))
    value = np.full((1, 1, 3, 3), -128, np.int8)
    assert oracle.exact_int64_conv2d(value, parameters, np.int8(0))[0, 0, 1, 1] == 9 * 128 * 128


def test_signed_halfway_rounding_clamping_and_safe_dequantization_subtraction():
    q = oracle.Quantization(np.float32(.5), np.int8(0))
    value = np.array([-100., -1.25, -.75, -.25, .25, .75, 1.25, 100.], np.float32)
    np.testing.assert_array_equal(oracle.quantize_fp32(value, q), np.array([-128, -2, -2, 0, 0, 2, 2, 127], np.int8))
    np.testing.assert_array_equal(oracle.dequantize_fp32(np.array([-128, 127], np.int8),
        oracle.Quantization(np.float32(.5), np.int8(127))), np.array([-127.5, 0], np.float32))


def test_fp64_conv_output_quantization_cannot_silently_use_literal_fp32_division():
    q = oracle.Quantization(np.float32(1.), np.int8(0))
    value = np.array([.5 + 2**-30], np.float64)
    assert oracle.quantize_conv_real(value, q).item() == 1
    assert oracle.quantize_fp32(value.astype(np.float32), q).item() == 0
    with pytest.raises(ValueError, match="FP32"):
        oracle.quantize_fp32(value, q)
    with pytest.raises(ValueError, match="FP64"):
        oracle.quantize_conv_real(value.astype(np.float32), q)


def test_epilogue_uses_output_channel_scale_and_declared_sequential_fp64_order():
    parameters = oracle.ConvParameters(np.zeros((3, 2, 3, 3), np.int8),
        np.array([.125, .25, .5], np.float32), np.zeros(3, np.int8), np.array([.125, -.25, .5], np.float32))
    q = oracle.Quantization(np.float32(.03125), np.int8(0))
    accumulator = np.array([2, -7, 11], np.int64).reshape(1, 3, 1, 1)
    actual = oracle.conv_epilogue(accumulator, parameters, q)
    expected = [((float(a) * float(q.scale)) * float(s)) + float(b)
                for a, s, b in zip(accumulator.reshape(-1), parameters.weight_scale, parameters.bias)]
    np.testing.assert_array_equal(actual.reshape(-1), expected)
    assert actual.dtype == np.float64


def test_integer_bias_rounding_ties_bound_and_integer_before_scale_addition():
    parameters = oracle.ConvParameters(np.zeros((2, 1, 3, 3), np.int8), np.ones(2, np.float32),
        np.zeros(2, np.int8), np.array([.5, 1.5], np.float32))
    q = oracle.Quantization(np.float32(1.), np.int8(0))
    proof = oracle.bias_rounding_proof(parameters, q)
    assert proof["bias_codes"] == [0, 2]
    assert proof["mathematical_abs_error_bound"] == [.5, .5]
    assert proof["computed_abs_reconstruction_error"] == [.5, .5]
    accumulator = np.array([3, -3], np.int64).reshape(1, 2, 1, 1)
    assert oracle.conv_epilogue(accumulator, parameters, q, rounded_bias=True).reshape(-1).tolist() == [3., -1.]
    with pytest.raises(ValueError, match="INT32"):
        oracle.bias_rounding_proof(conv(np.zeros((1, 1, 3, 3), np.int8), scale=1e-20, bias=1.), q)


def parameters():
    s1, s2 = np.float32(.0240773279), np.float32(.0168288182)
    first = conv(np.zeros((1, 1, 3, 3), np.int8), scale=.125, bias=np.float32((s1 + s2) / 4))
    weights = np.zeros((1, 1, 3, 3), np.int8)
    weights[0, 0, 1, 1] = 1
    return oracle.BlockParameters(oracle.Quantization(np.float32(.0213918332), np.int8(0)),
        oracle.Quantization(s1, np.int8(0)), oracle.Quantization(s2, np.int8(0)),
        oracle.Quantization(np.float32(.0307527091), np.int8(0)), first, conv(weights), "a" * 64)


def test_fixed_variants_remove_only_first_output_qdq_and_share_dequantized_residual(monkeypatch):
    params = parameters()
    stem = np.array([.011], np.float32).reshape(1, 1, 1, 1)
    calls = []
    original = oracle.exact_int64_conv2d
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(oracle, "exact_int64_conv2d", counted)
    result = oracle.evaluate_fixed_variants(stem, params)
    assert tuple(result) == oracle.VARIANTS and len(calls) == 6
    ordinary, removed = result[oracle.VARIANTS[0]], result[oracle.VARIANTS[2]]
    assert ordinary.stages["conv2_input_codes"].item() == 0
    assert removed.stages["conv2_input_codes"].item() == 1
    assert "conv1_codes" not in removed.stages and "conv1_dq" not in removed.stages
    for variant in result.values():
        assert variant.stages["stem_codes"].item() == 1
        assert variant.stages["stem_dq"].item() == params.stem_q.scale
        np.testing.assert_array_equal(variant.stages["add"], np.add(variant.stages["conv2_dq"],
            variant.stages["stem_dq"], dtype=np.float32))
        assert variant.stages["add"].dtype == variant.stages["block_out"].dtype == np.float32
        assert not variant.actual_tensorrt_graph


def test_checks_interrupt_tiled_work_without_retry():
    calls = []
    def stop():
        calls.append(True)
        raise TimeoutError("external original deadline")
    with pytest.raises(TimeoutError):
        oracle.exact_int64_conv2d(np.zeros((1, 1, 2, 2), np.int8), parameters().conv1, np.int8(0), check=stop)
    assert calls == [True]


def test_convolution_accounting_callback_records_exact_fixed_order():
    events = []
    oracle.evaluate_fixed_variants(np.zeros((1, 1, 1, 1), np.float32), parameters(),
        on_convolution=lambda *event: events.append(event))
    assert events == [(name, number, phase) for name in oracle.VARIANTS
                      for number in (1, 2) for phase in ("started", "completed")]


def test_first_accounting_callback_failure_prevents_native_convolution(monkeypatch):
    events = []
    def fail(*event):
        events.append(event)
        raise RuntimeError("durable accounting unavailable")
    monkeypatch.setattr(oracle, "exact_int64_conv2d", lambda *a, **kw: pytest.fail("Unaccounted convolution executed"))
    with pytest.raises(RuntimeError, match="accounting unavailable"):
        oracle.evaluate_fixed_variants(np.zeros((1, 1, 1, 1), np.float32), parameters(), on_convolution=fail)
    assert events == [(oracle.VARIANTS[0], 1, "started")]
    with pytest.raises(ValueError, match="callable"):
        oracle.evaluate_fixed_variants(np.zeros((1, 1, 1, 1), np.float32), parameters(), on_convolution=True)


def toy_model():
    onnx = pytest.importorskip("onnx")
    h, nh = onnx.helper, onnx.numpy_helper
    names = dict(oracle.STAGE_NODES)
    nodes, initializers = [], []
    def initializer(name, array):
        initializers.append(nh.from_array(array, name=name))
    for number in (1, 2):
        weights = np.zeros((1, 1, 3, 3), np.int8); weights[0, 0, 1, 1] = number
        for name, value in ((f"w{number}", weights), (f"ws{number}", np.array([.5], np.float32)),
                            (f"wz{number}", np.array([0], np.int8)), (f"b{number}", np.array([.125], np.float32))):
            initializer(name, value)
        nodes.append(h.make_node("DequantizeLinear", [f"w{number}", f"ws{number}", f"wz{number}"], [f"wdq{number}"],
            name=f"layer1.0.conv{number}.weight_DequantizeLinear", axis=0))
    previous = oracle.STEM
    for key, name in oracle.STAGE_NODES:
        if key.endswith("codes"):
            initializer(key + "s", np.array(.25, np.float32)); initializer(key + "z", np.array(0, np.int8))
            nodes.append(h.make_node("QuantizeLinear", [previous, key + "s", key + "z"], [key], name=name))
        elif key.endswith("dq"):
            code = previous
            nodes.append(h.make_node("DequantizeLinear", [code, code + "s", code + "z"], [key], name=name))
        elif key in ("conv1_pre_q", "conv2_pre_q"):
            number = 1 if key.startswith("conv1") else 2
            nodes.append(h.make_node("Conv", [previous, f"wdq{number}", f"b{number}"], [key], name=name,
                dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
        elif key == "add":
            nodes.append(h.make_node("Add", [previous, "stem_dq"], [key], name=name))
        else:
            nodes.append(h.make_node("Relu", [previous], [key], name=name))
        previous = key
    model = h.make_model(h.make_graph(nodes, "toy", [h.make_tensor_value_info(oracle.STEM, onnx.TensorProto.FLOAT, ["batch", 1, 3, 3])],
        [h.make_tensor_value_info("block_out", onnx.TensorProto.FLOAT, ["batch", 1, 3, 3])], initializers),
        opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    return model


def test_literal_copy_preserves_all_13_stages_shared_skip_and_source_bytes():
    model = toy_model(); before = model.SerializeToString()
    params = oracle.parse_block_parameters(model, hashlib.sha256(before).hexdigest())
    copied = oracle.extract_literal_block(model, params)
    assert model.SerializeToString() == before and len(copied.graph.output) == 13
    add = next(n for n in copied.graph.node if n.name == oracle.PREFIX + "Add")
    assert list(add.input) == ["conv2_dq", "stem_dq"]
    assert not params.conv1.weights.flags.writeable


def test_actual_ort_literal_control_one_small_synthetic_block_matches_explicit_fp32_topology(monkeypatch):
    onnx = pytest.importorskip("onnx"); ort = pytest.importorskip("onnxruntime")
    if onnx.__version__ != "1.16.2" or ort.__version__ != "1.18.1":
        pytest.skip("Frozen literal-control runtime versions unavailable")
    model = toy_model()
    params = oracle.parse_block_parameters(model, hashlib.sha256(model.SerializeToString()).hexdigest())
    copied = oracle.extract_literal_block(model, params)
    original_session = ort.InferenceSession
    calls = []
    def session(*args, **kwargs):
        actual = original_session(*args, **kwargs)
        def run(*run_args, **run_kwargs):
            calls.append(True)
            return actual.run(*run_args, **run_kwargs)
        return type("Session", (), {"get_providers": lambda self: actual.get_providers(), "run": lambda self, *a, **kw: run(*a, **kw)})()
    monkeypatch.setattr(ort, "InferenceSession", session)
    value = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 4
    result = oracle.run_literal_control(copied, value)
    assert tuple(result) == tuple(key for key, _ in oracle.STAGE_NODES) and calls == [True]
    np.testing.assert_array_equal(result["stem_dq"], value)
    np.testing.assert_array_equal(result["conv1_pre_q"], value * np.float32(.5) + np.float32(.125))
    assert result["conv1_codes"].dtype == np.int8 and result["block_out"].dtype == np.float32


def test_topology_validation_rejects_raw_stem_residual_and_changed_source_hash():
    model = toy_model()
    with pytest.raises(ValueError, match="SHA-256"):
        oracle.parse_block_parameters(model)
    add = next(node for node in model.graph.node if node.op_type == "Add")
    add.input[1] = oracle.STEM
    with pytest.raises(ValueError, match="shared stem DQ"):
        oracle.parse_block_parameters(model, hashlib.sha256(model.SerializeToString()).hexdigest())


def test_bitwise_control_distinguishes_equal_numeric_signed_zero():
    positive, negative = np.array([0.], np.float32), np.array([-0.], np.float32)
    result = oracle.compare_with_saved(positive, negative)
    assert result["equal_value_count"] == 1 and not result["bitwise_equal"]
    assert result["max_abs_error"] == 0
