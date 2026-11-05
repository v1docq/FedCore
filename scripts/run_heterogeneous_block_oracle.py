"""One supervised local CPU replay of the saved layer1.0 arithmetic maps."""
import argparse
import hashlib
import json
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
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=3).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True, timeout=3).strip()
    if head != args.source_revision or dirty:
        raise ValueError("Clean frozen source revision required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    module = REPO / "fedcore/experiments/heterogeneous_block_oracle.py"
    publish(root / "plan.json", dict(source_revision=head, started_at=started, deadline=started+WORK_SECONDS,
        max_seconds=SECONDS, max_RSS_bytes=RSS_LIMIT, script_sha256=sha(__file__), module_sha256=sha(module),
        model=str(Path(args.model).resolve(strict=True)), evidence=str(Path(args.evidence).resolve(strict=True)),
        unique_saved_rows=8, literal_ORT_calls=1, mathematical_variants=3, integer_convolutions=6,
        thread_environment=THREAD_ENV, GPU_requested=False, training=False, search=False,
        cleanup_reserve_seconds=SECONDS-WORK_SECONDS, monitoring_poll_seconds=.05,
        RSS_semantics="sampled worker RSS; no OS memory quota"))
    import psutil
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "worker", "--model", args.model,
               "--evidence", args.evidence, "--output", str(root), "--source-revision", head]
    environment = dict(os.environ, **THREAD_ENV, PYTHONDONTWRITEBYTECODE="1")
    with (root / "console.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            cwd=REPO, env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    identity = psutil.Process(child.pid)
    birth = identity.create_time()
    peak, reason = 0, "worker_exited"
    try:
        while child.poll() is None:
            try:
                if identity.create_time() != birth:
                    raise RuntimeError("Owned CPU worker lifetime changed")
                rss = identity.memory_info().rss
                peak = max(peak, rss)
                if rss > RSS_LIMIT:
                    reason = "RSS_limit"; child.kill(); break
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() >= started + WORK_SECONDS:
                reason = "deadline"; child.kill(); break
            time.sleep(.05)
        result = child.wait(timeout=3)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=3)
    receipt = dict(returncode=result, reason=reason, started_at=started, ended_at=time.monotonic(),
                   peak_RSS_bytes=peak, captured_pid=child.pid, captured_birth=birth,
                   max_seconds=SECONDS, max_RSS_bytes=RSS_LIMIT, GPU_requested=False)
    publish(root / "supervisor.json", receipt)
    print(json.dumps(receipt))
    return 0 if result == 0 and reason == "worker_exited" else 1


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
    for field in ["model","evidence","output","source-revision"]: parser.add_argument("--"+field,required=True)
    args=parser.parse_args()
    return supervise(args) if args.mode=="launch" else worker(args)


if __name__ == "__main__": raise SystemExit(main())
