"""Independent container/admission regressions; no CUDA or numerical workload."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from fedcore.experiments import heterogeneous_container as container
from fedcore.experiments.heterogeneous_pilot import PilotConfig, PilotError


GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def _boundary_fixture(tmp_path, monkeypatch):
    cfg = PilotConfig(GPU, "a" * 40)
    payload = json.loads(json.dumps(asdict(cfg)))
    plan = container.ContainerPlan(100., 101., 7000., "b" * 32,
        "/venv/bin/python", "/srv/fedcore/scripts/run_heterogeneous_container_pilot.py",
        "/runs/one/config.json", "/data/cifar", "/runs/one", payload)
    timeout_pid, supervisor_pid, guard_pid = 1700001, 1700002, 1700003
    worker_pid = container.os.getpid()
    timeout = container.LocalProcess(timeout_pid, 10000, 77, timeout_pid, timeout_pid,
        plan.timeout_argv, str(Path(container.TIMEOUT).resolve()))
    rows = {
        timeout_pid: timeout,
        supervisor_pid: container.LocalProcess(supervisor_pid, 10100, timeout_pid,
            timeout_pid, timeout_pid, plan.command("supervisor"), "/usr/bin/python3"),
        guard_pid: container.LocalProcess(guard_pid, 10200, supervisor_pid,
            timeout_pid, timeout_pid, plan.command("guard"), "/usr/bin/python3"),
        worker_pid: container.LocalProcess(worker_pid, 10300, guard_pid,
            timeout_pid, timeout_pid, plan.command("worker"), "/usr/bin/python3"),
    }
    receipt_path = tmp_path / "os-boundary.json"
    receipt_path.write_text('{"independent_fake_procfs":true}', encoding="utf-8")
    monkeypatch.setattr(container.sys, "platform", "linux")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(container, "_load_plan", lambda root: (plan, timeout))

    def verify():
        return container.verify_container_boundary(receipt_path, config=cfg,
            started_at=100., deadline=7240., process=rows.__getitem__,
            now=110., boot_now=110., hz=100, timer_present=lambda pid: True)

    return SimpleNamespace(plan=plan, timeout=timeout, rows=rows, verify=verify)


def test_detached_timeout_reparenting_preserves_its_verified_lifetime(tmp_path, monkeypatch):
    fixture = _boundary_fixture(tmp_path, monkeypatch)
    fixture.rows[fixture.timeout.pid] = replace(fixture.timeout, ppid=1)
    provenance = fixture.verify()
    assert provenance["kind"] == "live_gnu_timeout_container_boundary"
    assert provenance["host_pid_identity_proven"] is False
    assert provenance["original_started_at"] == 100.


@pytest.mark.parametrize("changed", [
    {"birth_ticks": 10001}, {"pgid": 1700999}, {"session": 1700999},
    {"argv": ("/usr/bin/timeout", "--foreground", "7000s", "unrelated")},
    {"executable": "/bin/an-unrelated-executable"},
])
def test_reparenting_does_not_hide_timeout_replacement_or_escape(tmp_path, monkeypatch, changed):
    fixture = _boundary_fixture(tmp_path, monkeypatch)
    fixture.rows[fixture.timeout.pid] = replace(fixture.timeout, ppid=1, **changed)
    with pytest.raises(PilotError):
        fixture.verify()


def test_worker_must_remain_in_the_direct_guard_supervisor_chain(tmp_path, monkeypatch):
    fixture = _boundary_fixture(tmp_path, monkeypatch)
    worker_pid = container.os.getpid()
    worker = fixture.rows[worker_pid]
    fixture.rows[worker_pid] = replace(worker, ppid=1700002)
    with pytest.raises(PilotError, match="ancestry"):
        fixture.verify()


def test_raw_gpu_pid_is_only_a_correlation_and_memory_may_change(monkeypatch):
    def forbidden(pid):
        pytest.fail("A raw NVML PID was resolved in local procfs")

    monkeypatch.setattr(container, "local_process", forbidden)
    initial = container.GpuCorrelation()
    assert container.observe_gpu((), initial, ready=True) == initial
    early = container.observe_gpu(("500001, 4",), initial, ready=False)
    assert early.host_pid == 500001 and not early.admitted
    admitted = container.observe_gpu(("500001, 4096",), early, ready=True)
    assert admitted.host_pid == 500001 and admitted.admitted
    assert "correlation only" in container.PROVENANCE


@pytest.mark.parametrize("rows", [
    (), ("500002, 4",), ("500001, 4", "500001, 4"),
    ("500001, 4", "500002, 4"), ("500001, N/A",),
    ("0, 4",), ("500001, -1",), ("",), ("500001, 16385",),
])
def test_every_missing_changed_extra_or_malformed_raw_row_stops(rows):
    with pytest.raises(PilotError):
        container.observe_gpu(rows, container.GpuCorrelation(500001, True), ready=True)


def test_capture_failure_cannot_prevent_killing_already_captured_owned_births():
    signalled = []

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return {101: 10., 102: 999.}[self.pid]

        def kill(self):
            signalled.append(self.pid)

    tree = container.OwnedDescendants.__new__(container.OwnedDescendants)
    tree.psutil = SimpleNamespace(Process=Process, NoSuchProcess=ProcessLookupError)
    tree.owned = {101: 10., 102: 20.}

    def escaped_after_capture():
        raise PilotError("A captured descendant escaped its process group")

    tree.capture = escaped_after_capture
    try:
        tree.kill()
    except PilotError:
        pass  # Reporting uncertain capture is allowed after owned signals.
    assert signalled == [101]  # Reused PID 102 remains untouched.


@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux procfs/GNU timeout CPU smoke")
def test_linux_cpu_timeout_covers_verified_roles_and_sleeping_grandchild(tmp_path):
    """Exercise native ancestry/timer checks and group KILL, without CUDA."""
    psutil = pytest.importorskip("psutil")
    version = subprocess.run((container.TIMEOUT, "--version"), capture_output=True,
        text=True, check=True, timeout=2).stdout
    assert version.startswith("timeout (GNU coreutils)")
    repository = Path(container.__file__).resolve().parents[2]
    script = tmp_path / "cpu_control_smoke.py"
    script.write_text("\n".join([
        "import argparse, json, os, subprocess, sys, time",
        "from dataclasses import asdict",
        "from pathlib import Path",
        "sys.path.insert(0, " + repr(str(repository)) + ")",
        "from fedcore.experiments import heterogeneous_container as c",
        "p = argparse.ArgumentParser()",
        "for name in ('mode', 'config', 'data-root', 'run-directory', 'started-at', 'token'):",
        "    p.add_argument('--' + name, required=True)",
        "a = p.parse_args()",
        "root = Path(a.run_directory)",
        "c._protect_parent(c.local_process(os.getppid()))",
        "c.wait_for_start_gate(root / 'os-boundary.json', timeout=10.)",
        "plan, _ = c._load_plan(root)",
        "cfg = c.pilot_config_from_payload(plan.config)",
        "proof = c.verify_container_boundary(root / 'os-boundary.json', config=cfg,",
        "    started_at=plan.started_at, deadline=plan.started_at + cfg.gpu_seconds - cfg.cleanup_seconds, role=a.mode)",
        "c._publish(root / (a.mode + '-verified.json'), proof)",
        "if a.mode in ('supervisor', 'guard'):",
        "    c._prctl(36, 1)",
        "    c._spawn(plan, 'guard' if a.mode == 'supervisor' else 'worker')",
        "else:",
        "    child = subprocess.Popen((sys.executable, '-c', 'import time; time.sleep(60)'), start_new_session=False)",
        "    local = c.local_process(child.pid)",
        "    c._publish(root / 'grandchild.json', dict(pid=child.pid, birth=__import__('psutil').Process(child.pid).create_time(), pgid=local.pgid))",
        "while True:",
        "    time.sleep(.02)",
        "",
    ]), encoding="utf-8")
    cfg = PilotConfig(GPU, "a" * 40, gpu_seconds=40., cleanup_seconds=12.)
    config_path = tmp_path / "config.json"
    container._publish(config_path, asdict(cfg))
    started_at = time.monotonic()
    plan = container.ContainerPlan(started_at, time.monotonic(), 3., "c" * 32,
        str(Path(sys.executable).absolute()), str(script), str(config_path), str(tmp_path),
        str(tmp_path), json.loads(json.dumps(asdict(cfg))))
    output_path = tmp_path / "cpu-control.log"
    captured = None
    with output_path.open("x", encoding="utf-8") as output:
        child = subprocess.Popen(plan.timeout_argv, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU))
    try:
        until = time.monotonic() + 1.
        while True:
            captured = container.local_process(child.pid)
            if captured.argv == plan.timeout_argv and container._timer_is_present(child.pid):
                break
            assert time.monotonic() < until, "Native watchdog did not arm"
            time.sleep(.01)
        container._publish(tmp_path / "os-boundary.json",
            dict(schema="fedcore-container-timeout-v1", plan=asdict(plan), timeout=asdict(captured)))
        assert child.wait(timeout=6) in (-9, 137)
        proofs = {}
        for role in ("supervisor", "guard", "worker"):
            path = tmp_path / (role + "-verified.json")
            assert path.is_file(), output_path.read_text(encoding="utf-8")
            proofs[role] = json.loads(path.read_text(encoding="utf-8"))
            assert proofs[role]["pgid"] == child.pid
            assert proofs[role]["live_posix_timer_observed"] is True
            assert proofs[role]["host_pid_identity_proven"] is False
        grandchild = json.loads((tmp_path / "grandchild.json").read_text(encoding="utf-8"))
        assert grandchild["pgid"] == child.pid
        until = time.monotonic() + 1.
        pids = [proof["worker_pid"] for proof in proofs.values()] + [grandchild["pid"]]
        while True:
            alive = []
            for pid in pids:
                try:
                    if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                break
            assert time.monotonic() < until, f"Owned CPU descendants survived timeout: {alive}"
            time.sleep(.01)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        # An assertion may fail after a descendant leaves the expected group.
        # Only the captured grandchild lifetime may receive fallback cleanup.
        path = tmp_path / "grandchild.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            try:
                process = psutil.Process(value["pid"])
                if process.create_time() == value["birth"]:
                    process.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize("accepted", [False, True])
def test_dynamic_trt_inspection_binds_context_and_preserves_rejected_diagnostics(
        tmp_path, monkeypatch, accepted):
    """All TensorRT objects are fakes; no builder, CUDA context or model runs."""
    from fedcore.algorithm.heterogeneous import backend

    events = []
    layer = {"Name": "Conv_10", "Metadata": "[ONNX Layer: Conv_10]",
        "LayerType": "CaskConvolution", "Inputs": [{"Format/Datatype": "Int8 row major"}],
        "Weights": {"Type": "Int8", "Count": 8}}
    if not accepted:
        layer["Inputs"] = [{"Format/Datatype": "Float row major"}]
    diagnostic = json.dumps({"Layers": [layer]}, indent=1) + "\n"

    class Context:
        shape = None

        def set_input_shape(self, name, shape):
            assert name == "images"
            self.shape = shape
            events.append("shape_bound")
            return True

    context = Context()

    class Inspector:
        execution_context = None

        def get_engine_information(self, output_format):
            assert self.execution_context is context
            assert context.shape == (4, 3, 32, 32)
            events.append("context_inspected")
            return diagnostic

    inspector = Inspector()
    engine = SimpleNamespace(create_execution_context=lambda: context,
        create_engine_inspector=lambda: inspector)

    class Logger:
        WARNING = 1

        def __init__(self, severity):
            assert severity == self.WARNING

    class Configuration:
        def set_memory_pool_limit(self, pool, count):
            pass

        def clear_flag(self, flag):
            pass

        def add_optimization_profile(self, profile):
            pass

    profile = SimpleNamespace(set_shape=lambda *args: True)
    builder = SimpleNamespace(create_network=lambda flags: object(),
        create_builder_config=Configuration, create_optimization_profile=lambda: profile,
        build_serialized_network=lambda network, config: b"fake-serialized-engine")
    fake_trt = SimpleNamespace(__version__="10.3.0", Logger=Logger,
        Builder=lambda logger: builder,
        OnnxParser=lambda network, logger: SimpleNamespace(parse=lambda value: True),
        Runtime=lambda logger: SimpleNamespace(deserialize_cuda_engine=lambda value: engine),
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        MemoryPoolType=SimpleNamespace(WORKSPACE=1),
        ProfilingVerbosity=SimpleNamespace(DETAILED=1), BuilderFlag=SimpleNamespace(TF32=1),
        LayerInformationFormat=SimpleNamespace(JSON=1))
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    source = tmp_path / "model.onnx"
    source.write_bytes(b"fake-onnx-not-parsed-by-any-runtime")
    output = tmp_path / "unpublished"
    output.mkdir()
    spec = backend.BackendSpec(runtime="tensorrt", device_uuid=GPU,
        max_batch_size=8, opt_batch_size=4)
    if accepted:
        path, evidence = backend._compile_tensorrt(source, output, spec, ("Conv_10",))
        assert path == output / "model.engine"
        assert evidence["tensorrt_int8_verified_nodes"] == ["Conv_10"]
        assert evidence["inspector_sha256"] == backend._sha(output / "engine-inspector.json")
    else:
        with pytest.raises(backend.BackendContractError) as failure:
            backend._compile_tensorrt(source, output, spec, ("Conv_10",))
        assert failure.value.code == "TensorRTFloatFallback"
    assert events == ["shape_bound", "context_inspected"]
    assert (output / "engine-inspector.json").read_text(encoding="utf-8") == diagnostic
    assert not (output / "manifest.json").exists()


def test_native_compact_qdq_export_preserves_nonzero_float_biases(tmp_path, monkeypatch):
    """Real CPU export: INT8 activation/weight QDQ and unchanged FP32 biases."""
    import numpy as np
    import torch

    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnxruntime import quantization
    from fedcore.algorithm.heterogeneous import backend
    from fedcore.algorithm.heterogeneous.cv import (
        BlockRecipe, HeterogeneousGene, HeterogeneousGenotype, compile_genotype,
        execute_float_workflow, prepare_space)
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = CifarResNet18().eval()
        calibration = torch.randn(2, 3, 32, 32)
    # Default BN produces all-zero folded biases, which cannot detect silent
    # bias rounding/removal. Exercise different nonzero per-channel values.
    with torch.no_grad():
        block = model.layer1[0]
        block.bn1.bias.copy_(torch.linspace(.125, .375, block.bn1.num_features))
        block.bn2.bias.copy_(torch.linspace(-.625, -.25, block.bn2.num_features))
    recipe = BlockRecipe(16, 8, True)
    space = prepare_space(model, calibration, alternatives={
        "layer1.0": (recipe,), "layer1.1": (BlockRecipe(),)})
    workflow = compile_genotype(HeterogeneousGenotype((
        HeterogeneousGene("layer1.0", recipe),
        HeterogeneousGene("layer1.1", BlockRecipe()))), space)
    built = execute_float_workflow(model, calibration, workflow)
    assert type(built) is CVWorkflowSuccess
    original_quantize = quantization.quantize_static
    float_biases, selected_names = {}, []

    def record_float_biases(model_input, model_output, reader, **kwargs):
        graph = onnx.load(model_input)
        initializers = {item.name: item for item in graph.graph.initializer}
        selected_names.extend(kwargs["nodes_to_quantize"])
        for node in graph.graph.node:
            if node.name in selected_names and len(node.input) == 3:
                assert node.op_type == "Conv"
                bias = initializers[node.input[2]]
                assert bias.data_type == onnx.TensorProto.FLOAT
                float_biases[node.name] = onnx.numpy_helper.to_array(bias).copy()
        return original_quantize(model_input, model_output, reader, **kwargs)

    monkeypatch.setattr(quantization, "quantize_static", record_float_biases)
    artifact = backend.export_onnx_qdq(built.model, calibration, workflow,
        tmp_path / "cpu-deployment", spec=backend.BackendSpec(max_batch_size=8, opt_batch_size=2))
    assert type(artifact) is backend.DeploymentArtifact, artifact
    graph = onnx.load(artifact.model_path)
    initializers = {item.name: item for item in graph.graph.initializer}
    nodes = {node.name: node for node in graph.graph.node}
    assert len(selected_names) == 3  # compact conv1 plus the two SVD factors
    assert float_biases and all(np.any(value != 0) for value in float_biases.values())
    for name, expected in float_biases.items():
        bias = initializers[nodes[name].input[2]]
        assert bias.data_type == onnx.TensorProto.FLOAT
        assert np.array_equal(onnx.numpy_helper.to_array(bias), expected)
    assert set(backend.inspect_qdq(graph, selected_names)) == set(selected_names)
    for node in graph.graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0] in initializers:
            assert initializers[node.input[0]].data_type != onnx.TensorProto.INT32
    reloaded = artifact.load()(calibration.numpy())
    assert reloaded.shape == (2, 10) and np.isfinite(reloaded).all()
    assert artifact.evidence["hardware_validation"] == "CPU_ONNX_Runtime_only"
