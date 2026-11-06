"""Prepare a CPU-only plan from the frozen rejected Q-only files and prior logits."""
from dataclasses import asdict
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-directory', required=True)
    parser.add_argument('--prior-pairs', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from fedcore.experiments import heterogeneous_layerwise as layerwise
    from fedcore.experiments import heterogeneous_qonly_diagnostic as diagnostic

    config = diagnostic.DiagnosticConfig(args.source_revision, str(Path(args.artifact_directory).resolve()),
        rows=8, batch_size=8, profile='layerwise_v1')
    before = diagnostic.artifact_snapshot(config)
    pairs_raw = Path(args.prior_pairs).read_bytes()
    if len(pairs_raw) > 8 * 1024**2 or hashlib.sha256(pairs_raw).hexdigest() != diagnostic.PRIOR_PAIRS_SHA256:
        raise ValueError('Exact bounded prior diagnostic logit file required')
    pairs = [json.loads(line) for line in pairs_raw.splitlines()]
    if ([i for batch in pairs for i in batch['calibration_positions']] != list(range(2000))
            or any(batch['backend_rtol'] != .02 or batch['backend_atol'] != .02 for batch in pairs)):
        raise ValueError('Prior logit file does not cover the exact original calibration order')
    indices = tuple(i for batch in pairs for i in batch['train_indices'])
    ort = np.concatenate([np.asarray(batch['cpu_qdq_logits'], dtype=np.float32) for batch in pairs])
    trt = np.concatenate([np.asarray(batch['tensorrt_logits'], dtype=np.float32) for batch in pairs])
    selected = layerwise.select_calibration_rows(indices, ort, trt)
    if len(selected) != 8:
        raise ValueError('Frozen layerwise profile requires eight selected images')
    artifact = Path(config.artifact_directory)
    model = onnx.load(str(artifact/'model.onnx'), load_external_data=False)
    original = model.SerializeToString()
    plan = layerwise.build_layerwise_plan(model, (artifact/'engine-inspector.json').read_bytes(), selected,
        source_onnx_sha256=diagnostic.SAVED_SHA256['model.onnx'],
        source_engine_sha256=diagnostic.SAVED_SHA256['model.engine'],
        source_inspector_sha256=diagnostic.SAVED_SHA256['engine-inspector.json'])
    instrumented = layerwise.instrument_onnx_copy(model, plan)
    if model.SerializeToString() != original or diagnostic.artifact_snapshot(config) != before:
        raise ValueError('Read-only source changed during planning')
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for tensor in node.input:
            consumers[tensor].append(node)
    def single(tensor, op):
        candidates = [node for node in consumers[tensor] if node.op_type == op]
        return candidates[0] if len(candidates) == 1 else None
    boundaries = []
    for conv in model.graph.node:
        if conv.op_type != 'Conv' or not conv.name.endswith('/conv1/Conv'):
            continue
        q1 = single(conv.output[0], 'QuantizeLinear')
        dq1 = single(q1.output[0], 'DequantizeLinear') if q1 else None
        relu = single(dq1.output[0], 'Relu') if dq1 else None
        q2 = single(relu.output[0], 'QuantizeLinear') if relu else None
        if q1 and q2:
            scale1, scale2 = float(values[q1.input[1]]), float(values[q2.input[1]])
            boundaries.append(dict(conv=conv.name, relu=relu.name, pre_relu_scale=scale1,
                post_relu_scale=scale2, scales_differ=scale1 != scale2))
    report = dict(schema='heterogeneous-layerwise-preparation-v1', requested_source_revision=args.source_revision,
        GPU_started=False, source_artifacts=before, prior_pairs_sha256=diagnostic.PRIOR_PAIRS_SHA256,
        requested_config=asdict(config), requested_budget=config.actual_budget(), plan=asdict(plan),
        original_node_counts=dict(Counter(node.op_type for node in model.graph.node)),
        conv1_relu_quantization_boundaries=boundaries,
        instrumented_onnx_sha256=hashlib.sha256(instrumented.SerializeToString()).hexdigest(),
        instrumented_output_names=[value.name for value in instrumented.graph.output], source_unchanged=True,
        interpretation='Static preparation only; no runtime or hardware diagnosis has been performed.')
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=args.output, selected_train_ids=[row.train_index for row in selected],
        landmarks=len(plan.landmarks), tap_bytes=plan.total_tap_bytes, GPU_started=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
