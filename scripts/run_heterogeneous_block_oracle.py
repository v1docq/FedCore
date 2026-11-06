"""One supervised local CPU replay of the saved layer1.0 arithmetic maps."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SECONDS = 180.
WORK_SECONDS = 176.
RSS_LIMIT = 1024**3
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
THREAD_ENV["CUDA_VISIBLE_DEVICES"] = ""
POLL_SECONDS = .05


class OwnedCpuTree:
    """Sample and stop only captured child lifetimes, including a venv redirector."""
    def __init__(self, child, psutil_module):
        self.child, self.psutil = child, psutil_module
        process = self.psutil.Process(child.pid)
        self.birth = process.create_time()
        self.owned = {child.pid: (self.birth, 0)}
        self.reused = set()

    def _live(self, pid, birth):
        try:
            process = self.psutil.Process(pid)
            if process.create_time() != birth:
                self.reused.add(pid)
                raise RuntimeError("Captured CPU process birth changed: " + str(pid))
            return process
        except self.psutil.NoSuchProcess:
            return None

    def capture(self):
        rss, live, seen = 0, [], set()
        pending = list(self.owned)
        while pending:
            pid = pending.pop(0)
            if pid in seen:
                continue
            seen.add(pid)
            birth, depth = self.owned[pid]
            process = self._live(pid, birth)
            if process is None:
                continue
            try:
                children = process.children(recursive=False)
                candidates = []
                for child in children:
                    try:
                        child_birth = child.create_time()
                        observed = self._live(child.pid, child_birth)
                        if observed is not None and observed.ppid() == pid:
                            candidates.append((child.pid, child_birth))
                    except self.psutil.NoSuchProcess:
                        continue
                # Do not acquire descendants from a reused parent PID.
                parent = self._live(pid, birth)
                if parent is None:
                    if candidates:
                        raise RuntimeError("Captured parent exited during descendant acquisition")
                    continue
                for child_pid, child_birth in candidates:
                    if child_pid in self.owned and self.owned[child_pid][0] != child_birth:
                        self.reused.add(child_pid)
                        raise RuntimeError("Captured CPU descendant birth changed: " + str(child_pid))
                    self.owned.setdefault(child_pid, (child_birth, depth + 1))
                    pending.append(child_pid)
                memory = parent.memory_info().rss
                if self._live(pid, birth) is not None:
                    rss += memory
                    live.append(pid)
            except self.psutil.NoSuchProcess:
                continue
        return rss, tuple(live)

    def kill(self):
        records = []
        try:
            self.capture()
        except Exception as error:
            records.append(dict(result="capture_failed", error=repr(error)))
        for pid, (birth, depth) in sorted(self.owned.items(), key=lambda item: item[1][1], reverse=True):
            try:
                process = self._live(pid, birth)
                if process is not None:
                    process.kill()  # psutil checks reuse again before signalling.
                    records.append(dict(pid=pid, birth=birth, depth=depth, result="owned_kill_sent"))
            except self.psutil.NoSuchProcess:
                continue
            except Exception as error:
                records.append(dict(pid=pid, birth=birth, result="not_signalled_or_failed", error=repr(error)))
        return records

    def known_gone(self):
        remaining = []
        for pid, (birth, _) in self.owned.items():
            try:
                if self._live(pid, birth) is not None:
                    remaining.append(pid)
            except RuntimeError:
                # The captured lifetime is gone; its replacement is unrelated.
                continue
        return not remaining, tuple(remaining)


def remaining_budget(previous_seconds):
    if (type(previous_seconds) not in (int, float) or not math.isfinite(previous_seconds)
            or not 0 <= previous_seconds < 175):
        raise ValueError("Previous CPU cost must be finite and in [0,175) seconds")
    return SECONDS - previous_seconds, WORK_SECONDS - previous_seconds


def monitor_owned_child(child, tree, *, work_deadline, hard_deadline, rss_limit=RSS_LIMIT,
                        clock=time.monotonic, sleep=time.sleep):
    """One original deadline; cleanup uses only its pre-reserved final four seconds."""
    peak, result, reason, failure, signals = 0, None, "worker_exited", None, []
    try:
        while True:
            rss, live = tree.capture()
            peak = max(peak, rss)
            if rss > rss_limit:
                reason = "RSS_limit"; break
            if clock() >= work_deadline:
                reason = "deadline"; break
            result = child.poll()
            if result is not None:
                child.wait(timeout=0)
                # A descendant may exit between the RSS snapshot and poll.
                # Only a fresh lifetime check can classify an orphan here.
                _, remaining = tree.known_gone()
                if any(pid != child.pid for pid in remaining):
                    reason = "launcher_exited_with_live_descendants"
                break
            sleep(min(POLL_SECONDS, max(0., work_deadline - clock())))
    except Exception as error:
        reason, failure = "supervision_failed", repr(error)
    finally:
        # Reap the direct Popen lifetime before the final procfs/handle check.
        while True:
            if child.poll() is not None:
                child.wait(timeout=0)
            try:
                gone, remaining = tree.known_gone()
            except Exception as error:
                gone, remaining = False, tuple(tree.owned)
                failure = failure or repr(error)
            if gone:
                break
            signals.extend(tree.kill())
            left = hard_deadline - clock()
            if left <= 0:
                break
            try:
                child.wait(timeout=min(POLL_SECONDS, left))
            except subprocess.TimeoutExpired:
                pass
            sleep(min(POLL_SECONDS, max(0., hard_deadline - clock())))
        try:
            gone, remaining = tree.known_gone()
        except Exception as error:
            gone, remaining = False, tuple(tree.owned)
            failure = failure or repr(error)
        result = child.poll()
    if tree.reused:
        reason = "supervision_failed"
        failure = failure or "Captured PID birth changed; replacements were not signalled"
    return dict(returncode=result, reason=reason, failure=failure, peak_RSS_bytes=peak,
        captured_pid=child.pid, captured_birth=tree.birth,
        owned_processes=[dict(pid=pid, birth=birth, depth=depth) for pid, (birth, depth) in tree.owned.items()],
        reused_pids_not_signalled=sorted(tree.reused), cleanup_signals=signals,
        known_owned_tree_gone=gone, remaining_owned_pids=list(remaining),
        RSS_semantics="sampled sum of stable owned launcher and recursively captured descendants; no OS memory quota")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def event(root, name, **value):
    with (root / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event=name, at=time.monotonic(), **value), allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def supervise(args):
    started = time.monotonic()
    previous_seconds = getattr(args, "previous_seconds", 0.)
    remaining_seconds, work_seconds = remaining_budget(previous_seconds)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=3).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True, timeout=3).strip()
    if head != args.source_revision or dirty:
        raise ValueError("Clean frozen source revision required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    module = REPO / "fedcore/experiments/heterogeneous_block_oracle.py"
    publish(root / "plan.json", dict(source_revision=head, started_at=started, deadline=started+work_seconds,
        max_seconds=SECONDS, max_RSS_bytes=RSS_LIMIT, script_sha256=sha(__file__), module_sha256=sha(module),
        model=str(Path(args.model).resolve(strict=True)), evidence=str(Path(args.evidence).resolve(strict=True)),
        unique_saved_rows=8, literal_ORT_calls=1, mathematical_variants=3, integer_convolutions=6,
        thread_environment=THREAD_ENV, GPU_requested=False, training=False, search=False,
        cleanup_reserve_seconds=SECONDS-WORK_SECONDS, monitoring_poll_seconds=POLL_SECONDS,
        previous_seconds=previous_seconds, remaining_seconds=remaining_seconds,
        RSS_semantics="sampled sum of stable owned launcher and recursively captured descendants; no OS memory quota"))
    import psutil
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "worker", "--model", args.model,
               "--evidence", args.evidence, "--output", str(root), "--source-revision", head]
    environment = dict(os.environ, **THREAD_ENV, PYTHONDONTWRITEBYTECODE="1")
    with (root / "console.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            cwd=REPO, env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        tree = OwnedCpuTree(child, psutil)
    except Exception:
        if child.poll() is None:
            child.kill(); child.wait(timeout=3)
        raise
    receipt = monitor_owned_child(child, tree, work_deadline=started + work_seconds,
                                  hard_deadline=started + remaining_seconds)
    ended = time.monotonic()
    cumulative = previous_seconds + ended - started
    receipt.update(started_at=started, ended_at=ended, previous_seconds=previous_seconds,
        elapsed_seconds=ended-started, cumulative_seconds=cumulative, remaining_seconds=remaining_seconds,
        within_original_total_budget=cumulative <= SECONDS, max_seconds=SECONDS,
        max_RSS_bytes=RSS_LIMIT, GPU_requested=False)
    publish(root / "supervisor.json", receipt)
    print(json.dumps(receipt))
    return 0 if (receipt["returncode"] == 0 and receipt["reason"] == "worker_exited"
                 and receipt["known_owned_tree_gone"] and receipt["within_original_total_budget"]) else 1


def worker(args):
    os.environ.update(THREAD_ENV)  # before importing a numerical package
    import numpy as np
    import onnx
    import onnxruntime as ort
    from fedcore.experiments import heterogeneous_block_oracle as oracle
    root, evidence = Path(args.output), Path(args.evidence)
    plan = json.loads((root / "plan.json").read_text())
    if (plan["source_revision"] != args.source_revision or plan["script_sha256"] != sha(__file__)
            or plan["module_sha256"] != sha(oracle.__file__)
            or plan["model"] != str(Path(args.model).resolve()) or plan["evidence"] != str(evidence.resolve())):
        raise ValueError("CPU worker arguments/source differ from frozen plan")
    deadline = plan["deadline"]
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Original CPU oracle deadline reached")
    literal_calls, conv_started, conv_completed = 0, 0, 0
    def on_conv(variant, number, state):
        nonlocal conv_started, conv_completed
        check()
        if state == "started":
            if conv_started >= 6: raise ValueError("Fixed six-convolution budget exceeded")
            conv_started += 1
        elif state == "completed": conv_completed += 1
        else: raise ValueError("Unknown convolution event")
        event(root, "integer_conv_"+state, variant=variant, convolution=number,
              attempted=conv_started, completed=conv_completed)
    before, after, results, literal, baseline = {}, {}, {}, {}, {}
    failure, status = None, "failed"
    try:
        check()
        if onnx.__version__ != "1.16.2" or ort.__version__ != "1.18.1":
            raise ValueError("Frozen ONNX1.16.2/ORT1.18.1 required")
        manifest = json.loads((REPO / "docs/heterogeneous_compression/validation/layerwise-run-2026-11-05/ARTIFACTS.json").read_text())
        paths = {name: evidence/name for name in ["tap-batch-0.npz", "tap-batch-1.npz", "diagnostic-receipt.json"]}
        for name,path in paths.items():
            if sha(path) != manifest[name]["sha256"]: raise ValueError("Saved evidence hash changed: "+name)
        before = {name: sha(path) for name,path in paths.items()}
        before["source_model"] = sha(args.model)
        if before["source_model"] != oracle.SOURCE_ONNX_SHA256: raise ValueError("Frozen source ONNX differs")
        source_receipt = json.loads(paths["diagnostic-receipt.json"].read_text())
        ids = [row["train_index"] for row in source_receipt["selected_rows"]]
        if ids != [11,20,33848,3998,11666,3192,31270,6088]: raise ValueError("Frozen eight-row order differs")
        groups = []
        for number in [0,1]:
            with np.load(paths[f"tap-batch-{number}.npz"], allow_pickle=False) as archive:
                groups.append({name:archive[name] for name in ["ort__stem","ort__layer1.0","trt__layer1.0"]})
        stem = np.concatenate([g["ort__stem"] for g in groups])
        saved_ort = np.concatenate([g["ort__layer1.0"] for g in groups])
        saved_trt = np.concatenate([g["trt__layer1.0"] for g in groups])
        if stem.shape != (8,64,32,32) or stem.dtype != np.float32: raise ValueError("Fixed saved stem required")
        model = onnx.load(args.model, load_external_data=False)
        parameters = oracle.parse_block_parameters(model)
        block = oracle.extract_literal_block(model, parameters)
        with (root / "literal-block.onnx").open("xb") as stream: stream.write(block.SerializeToString())
        publish(root / "contract.json", dict(precision=oracle.PRECISION_CONTRACT, variants=list(oracle.VARIANTS),
            selected_train_ids=ids, source_hashes=before, literal_output_count=len(block.graph.output),
            literal_node_count=len(block.graph.node), array_hash_rule="SHA256(shape,dtype prefix + contiguous bytes)",
            parameters={"stem_scale":float(parameters.stem_q.scale),"conv1_scale":float(parameters.conv1_q.scale),
                        "conv2_input_scale":float(parameters.conv2_input_q.scale),"conv2_scale":float(parameters.conv2_q.scale)},
            CPU_versions=dict(numpy=np.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__)))
        check(); literal_calls += 1
        event(root, "literal_ORT_started", attempted=literal_calls, output_count=13)
        literal = oracle.run_literal_control(block, stem)
        event(root, "literal_ORT_completed", completed=literal_calls)
        baseline = oracle.compare_with_saved(saved_ort, literal["block_out"])
        publish(root / "baseline-control.json", baseline)
        check()
        results = oracle.evaluate_fixed_variants(stem, parameters, check=check, on_convolution=on_conv)
        check()
        def compare(ref,actual):
            value = oracle.compare_with_saved(ref,actual)
            if ref.dtype.kind in "iu" and actual.dtype.kind in "iu":
                value.pop("original_logit_tolerance_violations",None)
                value["exact_code_mismatches"] = value["elements"]-value["equal_value_count"]
                value["comparison_rule"] = "exact integer-code equality"
            return value
        summaries = {}
        for name,result in results.items():
            summaries[name] = dict(precision=result.precision_contract,proofs=result.proofs,
                vs_saved_ORT=compare(saved_ort,result.stages["block_out"]),
                vs_saved_rebuilt_TensorRT=compare(saved_trt,result.stages["block_out"]),
                vs_literal_stages={key:compare(literal[key],array) for key,array in result.stages.items() if key in literal})
        base = results[oracle.VARIANTS[0]]
        pairs = {name:{key:compare(base.stages[key],array) for key,array in result.stages.items() if key in base.stages}
                 for name,result in results.items() if name != base.name}
        publish(root / "comparisons.json", dict(variants=summaries,paired_variant_effects=pairs))
        arrays = {"saved_ORT_stem":stem,"saved_ORT_block":saved_ort,"saved_rebuilt_TRT_block":saved_trt}
        arrays.update({"literal__"+key:array for key,array in literal.items()})
        for name,result in results.items(): arrays.update({name+"__"+key:array for key,array in result.stages.items()})
        with (root / "arrays.npz").open("xb") as stream:
            np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        after = {name:sha(path) for name,path in paths.items()}
        after["source_model"] = sha(args.model)
        if before != after: raise ValueError("Read-only inputs changed")
        if literal_calls != 1 or conv_started != 6 or conv_completed != 6 or tuple(results) != oracle.VARIANTS:
            raise ValueError("Fixed CPU arithmetic accounting differs")
        check(); status = "complete"
    except Exception as error:
        failure = dict(type=type(error).__name__,message=str(error))
    receipt = dict(schema="heterogeneous-block-oracle-v1",execution_status=status,failure=failure,
        worker_pid=os.getpid(), parent_pid=os.getppid(),
        baseline_status="bitwise_reproduced" if baseline.get("bitwise_equal") else "baseline_reproduction_failed",
        baseline_gate_passed=bool(status=="complete" and baseline.get("bitwise_equal")),
        causal_attribution_to_TensorRT_supported=False,
        source_revision=args.source_revision, started_at=plan["started_at"], ended_at=time.monotonic(),
        deadline=deadline, literal_ORT_calls=literal_calls, integer_convs_started=conv_started,
        integer_convs_completed=conv_completed, completed_variants=list(results), before=before, after=after,
        CPU_versions=dict(numpy=np.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__),
        GPU_requested=False, SGD=0, training=False, search=False, official_test=False,
        deployment_accepted=False, original_engine_first_divergence_located=False,
        source_module_sha256=sha(oracle.__file__), script_sha256=sha(__file__),
        interpretation="Frozen mathematical maps only; similarity does not identify actual TensorRT kernel or original engine")
    publish(root / "receipt.json", receipt)
    return 0 if status == "complete" else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["launch","worker"],default="launch")
    parser.add_argument("--previous-seconds", type=float, default=0.)
    for field in ["model","evidence","output","source-revision"]: parser.add_argument("--"+field,required=True)
    args=parser.parse_args()
    if args.mode == "worker" and args.previous_seconds != 0:
        parser.error("--previous-seconds belongs only to the supervisor; worker uses the frozen original plan")
    return supervise(args) if args.mode=="launch" else worker(args)


if __name__ == "__main__": raise SystemExit(main())
