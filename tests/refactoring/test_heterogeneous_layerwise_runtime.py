"""CPU fakes exercise the fixed replay and independent binding contracts."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fedcore.experiments import heterogeneous_layerwise as layerwise
from fedcore.experiments import heterogeneous_layerwise_runtime as runtime
from fedcore.experiments import heterogeneous_qonly_diagnostic as qonly
from fedcore.experiments.heterogeneous_pilot import PilotError


def rows():
    return tuple(layerwise.SelectedRow(i, i + 100, "original_first_two" if i < 2 else "largest_ORT_margin_disagreement", 2., 0, 1)
                 for i in range(8))


def inputs():
    value = np.zeros((8, 3, 32, 32), np.float32)
    value[:, 0, 0, 0] = np.arange(8)
    return value


def logits(value):
    return np.repeat(value[:, 0, 0, 0, None], 10, axis=1)


def test_selected_input_artifact_preserves_exact_bytes_ids_and_both_hashes(tmp_path):
    value = inputs()
    identity = runtime._save_selected_inputs(tmp_path, value, rows())
    assert np.array_equal(np.load(identity["path"], allow_pickle=False), value)
    assert identity["sha256"] == qonly._sha(identity["path"])
    assert identity["array_sha256"] == qonly._array_sha(value)
    assert identity["selected_rows"] == [asdict(row) for row in rows()]
    assert json.loads((tmp_path / "selected-inputs.json").read_text()) == identity
    with pytest.raises(FileExistsError):
        runtime._save_selected_inputs(tmp_path, value, rows())


def test_fixed_replay_and_independent_enriched_stages_have_exact_28calls_72visits(tmp_path):
    accounting = runtime.Accounting()
    recorder = runtime._Recorder(tmp_path, accounting, deadline=100., clock=lambda: 1.)
    selected, data = rows(), inputs()
    result = runtime.run_logit_replays(data, selected, ort_runner=logits, trt_runner=logits, record=recorder.call)
    assert accounting.calls_completed == 22 and accounting.row_visits_completed == 48
    for per_runtime in result.values():
        assert all(np.array_equal(output, logits(data)) for output in per_runtime.values())
    for start, stop in runtime.replay_spans("partitioned"):
        recorder.call("independent_binding_TensorRT", "partitioned", data[start:stop], selected[start:stop], logits)
    for name in ("instrumented_ORT", "instrumented_TensorRT"):
        for start, stop in runtime.replay_spans("partitioned"):
            recorder.call(name, "partitioned", data[start:stop], selected[start:stop], logits)
    assert accounting.calls_completed == accounting.calls_attempted == 28
    assert accounting.row_visits_completed == accounting.row_visits_attempted == 72
    lines = [json.loads(line) for line in (tmp_path / "layerwise-logits.jsonl").read_text().splitlines()]
    assert len(lines) == 28 and [row["call_id"] for row in lines] == list(range(28))
    assert lines[0]["input_shape"][0] == 2 and lines[1]["input_shape"][0] == 6
    assert lines[2]["input_shape"][0] == 1 and lines[10]["input_shape"][0] == 8
    with pytest.raises(PilotError, match="budget"):
        recorder.call("extra", "whole", data, selected, lambda x: pytest.fail("Budget overflow ran inference"))


def test_expired_deadline_never_invokes_runtime_or_counts_a_call(tmp_path):
    accounting = runtime.Accounting()
    recorder = runtime._Recorder(tmp_path, accounting, deadline=100., clock=lambda: 100.)
    with pytest.raises(TimeoutError):
        recorder.call("original_ORT", "whole", inputs(), rows(), lambda x: pytest.fail("Expired worker executed"))
    assert accounting.calls_attempted == 0 and not (tmp_path / "layerwise-logits.jsonl").exists()


def test_runtime_failure_preserves_attempted_cost_without_false_completion(tmp_path):
    accounting = runtime.Accounting()
    recorder = runtime._Recorder(tmp_path, accounting, deadline=100., clock=lambda: 1.)
    with pytest.raises(RuntimeError):
        recorder.call("original_ORT", "whole", inputs(), rows(), lambda x: (_ for _ in ()).throw(RuntimeError("failed")))
    assert accounting.calls_attempted == 1 and accounting.row_visits_attempted == 8
    assert accounting.calls_completed == accounting.row_visits_completed == 0
    progress = [json.loads(line) for line in (tmp_path / "layerwise-progress.jsonl").read_text().splitlines()]
    assert len(progress) == 1 and progress[0]["event"] == "inference_started"
    assert progress[0]["selected_rows"] == [asdict(row) for row in rows()]
    assert progress[0]["accounting"]["row_visits_attempted"] == 8


def test_prior_pair_loader_checks_full_order_ids_and_hash(tmp_path):
    path = tmp_path / "prior.jsonl"
    def publish(changed=False):
        with path.open("w") as stream:
            for number, (start, stop) in enumerate(qonly.batch_ranges(2000, 128)):
                indices = list(range(start, stop))
                if changed and number == 1:
                    indices[0] = 0
                row = dict(schema="heterogeneous-qonly-logit-pairs-v1", batch_number=number,
                    calibration_positions=list(range(start, stop)), train_indices=indices,
                    cpu_qdq_logits=np.zeros((stop - start, 10), np.float32).tolist(),
                    tensorrt_logits=np.ones((stop - start, 10), np.float32).tolist())
                stream.write(json.dumps(row) + "\n")
    publish()
    ref, actual = runtime.load_prior_pairs(path, tuple(range(2000)), expected_sha256=qonly._sha(path))
    assert ref.shape == (2000, 10) and np.all(actual == 1)
    with pytest.raises(PilotError, match="hash"):
        runtime.load_prior_pairs(path, tuple(range(2000)), expected_sha256="f" * 64)
    publish(changed=True)
    with pytest.raises(PilotError, match="positions or calibration IDs"):
        runtime.load_prior_pairs(path, tuple(range(2000)), expected_sha256=qonly._sha(path))


def fake_binding(monkeypatch, *, bad_address=False, bad_format=False, bad_shape=False, bad_location=False):
    from fedcore.experiments import heterogeneous_device
    events, registry = [], {}
    class Tensor:
        def __init__(self, array):
            self.array = array
            registry[id(self)] = self
        def to(self, device, non_blocking):
            events.append("copy")
            return self
        def data_ptr(self):
            return id(self)
        def cpu(self):
            events.append("cpu_copy")
            return self
        def numpy(self):
            return self.array
    stream = SimpleNamespace(cuda_stream=999, synchronize=lambda: events.append("synchronize"))
    @contextmanager
    def on_stream(value):
        assert value is stream
        events.append("enter_nondefault_stream")
        yield
    torch = SimpleNamespace(float32="float32", from_numpy=lambda x: Tensor(x),
        empty=lambda shape, **kw: Tensor(np.zeros(shape, np.float32)),
        cuda=SimpleNamespace(Stream=lambda **kw: stream, stream=on_stream))
    addresses = {}
    class Context:
        shape = None
        def set_input_shape(self, name, shape):
            self.shape = shape
            return True
        def get_tensor_shape(self, name):
            return (self.shape[0], 11 if bad_shape else 10)
        def set_tensor_address(self, name, address):
            addresses[name] = address
            return True
        def get_tensor_address(self, name):
            return addresses[name] + int(bad_address)
        def execute_async_v3(self, stream_handle):
            assert stream_handle == 999
            assert set(addresses) == {"images", "logits"}
            events.append("execute")
            registry[addresses["logits"]].array[:] = 7.
            return True
    engine = SimpleNamespace(create_execution_context=Context, num_io_tensors=2,
        get_tensor_name=lambda i: ("images", "logits")[i], get_tensor_mode=lambda name: "input" if name == "images" else "output",
        get_tensor_dtype=lambda name: "float32", get_tensor_format=lambda name: "blocked" if bad_format else "linear",
        get_tensor_location=lambda name: "host" if bad_location else "device")
    trt = SimpleNamespace(float32="float32", TensorIOMode=SimpleNamespace(INPUT="input", OUTPUT="output"),
        TensorFormat=SimpleNamespace(LINEAR="linear"), TensorLocation=SimpleNamespace(DEVICE="device"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(heterogeneous_device, "require_cuda_binding", lambda uuid: events.append("uuid_verified"))
    return engine, trt, events


def test_independent_binding_uses_fresh_context_nondefault_stream_addresses_and_sync(monkeypatch):
    engine, trt, events = fake_binding(monkeypatch)
    runner = runtime.NamedTensorRTRunner(engine, trt, qonly.GPU_UUID, {"logits": (10,)})
    result = runner(inputs()[:2])
    assert result["logits"].shape == (2, 10) and np.all(result["logits"] == 7)
    assert events[0] == "uuid_verified" and "enter_nondefault_stream" in events
    assert events.index("copy") < events.index("synchronize") < events.index("execute")
    assert events[-2:] == ["synchronize", "cpu_copy"]


@pytest.mark.parametrize("bad", ["bad_address", "bad_format", "bad_shape", "bad_location"])
def test_independent_binding_rejects_invalid_contract_before_execution(monkeypatch, bad):
    engine, trt, events = fake_binding(monkeypatch, **{bad: True})
    with pytest.raises(PilotError):
        runner = runtime.NamedTensorRTRunner(engine, trt, qonly.GPU_UUID, {"logits": (10,)})
        runner(inputs()[:2])
    assert "execute" not in events


def test_ort_optimized_output_uses_original_thread_and_provider_settings(tmp_path, monkeypatch):
    calls = []
    ort = SimpleNamespace(SessionOptions=SimpleNamespace,
        InferenceSession=lambda path, options, **kw: calls.append((path, options, kw)))
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    runtime._ort_session_with_optimized_graph("original.onnx", tmp_path / "fresh.onnx")
    path, options, kwargs = calls[0]
    assert path == "original.onnx" and options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert options.optimized_model_filepath == str(tmp_path / "fresh.onnx")
    assert kwargs == {"providers": ["CPUExecutionProvider"]}


def test_second_engine_build_is_rejected_before_build_or_budget_check(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="10.0.1"))
    accounting = runtime.Accounting(engine_builds_attempted=1)
    with pytest.raises(PilotError, match="one TensorRT"):
        runtime._build_instrumented_engine(tmp_path / "unused.onnx", tmp_path, accounting,
            lambda: pytest.fail("A repeated build advanced to execution"))


def test_native_builder_failure_leaves_durable_attempt_and_source_flags(tmp_path, monkeypatch):
    source = tmp_path / "instrumented.onnx"
    source.write_bytes(b"bounded diagnostic graph")
    def fail_builder(logger):
        events = [json.loads(line) for line in (tmp_path / "layerwise-progress.jsonl").read_text().splitlines()]
        assert len(events) == 1 and events[0]["event"] == "engine_build_started"
        assert events[0]["onnx_sha256"] == qonly._sha(source)
        assert events[0]["accounting"]["engine_builds_attempted"] == 1
        assert events[0]["tf32"] is False and events[0]["workspace_bytes"] == 1024**3
        raise RuntimeError("native build failed")
    class Logger:
        WARNING = 1
        def __init__(self, level):
            pass
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="10.0.1", Logger=Logger, Builder=fail_builder))
    accounting = runtime.Accounting()
    with pytest.raises(RuntimeError, match="native build"):
        runtime._build_instrumented_engine(source, tmp_path, accounting, lambda: None)
    assert accounting.engine_builds_attempted == 1 and accounting.engine_builds_completed == 0


@pytest.mark.parametrize("within", [True, False])
def test_original_controls_are_saved_before_independent_binding_gate(tmp_path, within):
    controls = dict(independent_binding_vs_existing_TensorRT=dict(within_original_tolerance=within,
        backend_rtol=.02, backend_atol=.02, max_abs_error=0. if within else 1.))
    if within:
        runtime._original_controls_gate(tmp_path, controls)
    else:
        with pytest.raises(PilotError, match="no diagnostic engine built"):
            runtime._original_controls_gate(tmp_path, controls)
    assert json.loads((tmp_path / "original-controls.json").read_text()) == controls
    assert not (tmp_path / "instrumented.engine").exists()


def test_tap_capture_persists_all_arrays_and_summary_without_original_graph_claim(tmp_path):
    taps = tuple(SimpleNamespace(tensor=f"tensor{i}", name=f"landmark{i}", shape=(8, 2)) for i in range(12))
    plan = SimpleNamespace(landmarks=taps)
    reference = {item.tensor: np.zeros((2, 2), np.float32) for item in taps}
    actual = {item.tensor: np.ones((2, 2), np.float32) for item in taps}
    accounting = runtime.Accounting()
    saved = runtime._save_taps(tmp_path, 0, rows()[:2], reference, actual, plan, accounting)
    with np.load(saved["path"]) as arrays:
        assert len(arrays.files) == 24 and np.all(arrays["trt__landmark0"] == 1)
    summaries = [json.loads(line) for line in (tmp_path / "tap-summaries.jsonl").read_text().splitlines()]
    assert len(summaries) == 12 and all(row["graph_provenance"] == "rebuilt_diagnostic_graph" for row in summaries)
    assert accounting.tap_payload_bytes == 12 * 2 * 2 * 2 * 4
    assert saved["sha256"] == qonly._sha(saved["path"])


def test_combined_tap_payload_cap_is_enforced_before_writing_arrays(tmp_path):
    taps = tuple(SimpleNamespace(tensor=f"tensor{i}", name=f"landmark{i}", shape=(8, 2)) for i in range(12))
    outputs = {item.tensor: np.zeros((2, 2), np.float32) for item in taps}
    accounting = runtime.Accounting(tap_payload_bytes=layerwise.MAX_TAP_BYTES)
    with pytest.raises(PilotError, match="64 MiB"):
        runtime._save_taps(tmp_path, 0, rows()[:2], outputs, outputs, SimpleNamespace(landmarks=taps), accounting)
    assert not (tmp_path / "tap-batch-0.npz").exists()
