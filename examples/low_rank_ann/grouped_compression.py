"""CPU integration of shared-basis groups and actual current-student schedules.

Random untrained models demonstrate contracts, not model quality or speedup.
Separate calibration and held-out inputs use explicit independent seeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.artifacts import StatisticsStore
from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.algorithm.low_rank.compression_graph import SourceMode
from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention
from fedcore.algorithm.low_rank.coupled_specs import FlatQueryKey
from fedcore.algorithm.low_rank.grouped_execution import execute_grouped_workflow, prepare_grouped_workflow
from fedcore.algorithm.low_rank.grouped_workflow import (
    CompressionRegion, ConsumerCoordinates, CoordinateAxis, GroupRequest,
)
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import BasisSharing, DRONE, SVDLLMV1
from fedcore.algorithm.low_rank.topology import module_paths, replace_modules_atomically
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


def _bytes(model):
    return inventory_cost(capture_inventory(model), "bytes")


def _request(model, key, paths, spec, rank, *, shared_inputs=False):
    modules = module_paths(model)
    bindings = tuple(ConsumerCoordinates(path,
        CoordinateAxis("declared_shared_input" if shared_inputs else key + ":input",
                       tuple(range(modules[path].in_features))),
        CoordinateAxis(key + ":" + path + ":output", tuple(range(modules[path].out_features))))
        for path in paths)
    return GroupRequest(key, CompressionRegion(key + ":region", paths, bindings), paths, spec, rank)


def _execute(model, calibration, requests, **options):
    workflow = prepare_grouped_workflow(model, calibration, requests, **options)
    with_store = StatisticsStore(max_bytes=4 * 1024**2)
    try:
        return execute_grouped_workflow(model, calibration, workflow, store=with_store)
    finally:
        with_store.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    random.seed(163)
    np.random.seed(163)
    torch.manual_seed(163)
    calibration = torch.randn(48, 6, generator=torch.Generator().manual_seed(164), dtype=torch.float64) + .5
    heldout = torch.randn(64, 6, generator=torch.Generator().manual_seed(165), dtype=torch.float64) + .5
    source = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 6)).double().eval()
    basis_request = _request(source, "shared", ("0", "2"), BasisSharing(), 2, shared_inputs=True)
    shared = _execute(source, calibration, (basis_request,), source_mode=SourceMode.FROZEN_SNAPSHOT,
        graph_version="g4-functional-basis", tensor_byte_budget=384)
    assert shared.model[0].right is shared.model[2].right
    manager = CheckpointManager(str(args.output_dir), auto_cleanup=False)
    payload = manager.serialize_to_bytes(shared.model)
    reloaded = manager.deserialize_from_bytes(payload)
    assert reloaded[0].right is reloaded[2].right
    torch.testing.assert_close(reloaded(heldout), shared.model(heldout), rtol=1e-12, atol=1e-12)
    assert _bytes(reloaded) == _bytes(shared.model) <= 384
    legacy_shared = transform_method(source, calibration, BasisSharing(), rank=2, target_paths=("0", "2"))
    torch.testing.assert_close(shared.model(heldout), legacy_shared.model(heldout), rtol=1e-9, atol=1e-10)

    sequence_source = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 4)).double().eval()
    sequential = []
    for name, spec in (("drone", DRONE()), ("svdllm_v1", SVDLLMV1())):
        requests = tuple(_request(sequence_source, name + ":" + path, (path,), spec, 2) for path in ("0", "2"))
        result = _execute(sequence_source, calibration, requests, source_mode=SourceMode.CURRENT_STUDENT,
            graph_version="g4-functional-" + name, tensor_byte_budget=392)
        legacy = transform_method(sequence_source, calibration, spec, rank=2, target_paths=("0", "2"))
        torch.testing.assert_close(result.model(heldout), legacy.model(heldout), rtol=1e-8, atol=1e-9)
        replacements = {}
        for path in ("0", "2"):
            individual = transform_method(sequence_source, calibration, spec, rank=2, target_paths=(path,))
            replacements[path] = module_paths(individual.model)[path]
        frozen = replace_modules_atomically(deepcopy(sequence_source), replacements)
        difference = float((result.model(heldout) - frozen(heldout)).square().mean())
        assert difference > 1e-12, "The fixture must distinguish current-student and frozen-M0 assembly"
        advances = [event for event in result.evidence["trace"] if event["stage"] == "advance"]
        assert len(advances) == 2 and result.evidence["accepted_versions"] == 2
        sequential.append({"method": name, "source_bytes": _bytes(sequence_source),
            "result_bytes": _bytes(result.model), "heldout_mse_against_source": float(
                (result.model(heldout) - sequence_source(heldout)).square().mean()),
            "heldout_mse_against_frozen_assembly": difference, "legacy_parity": True,
            "execution": result.evidence})

    attention = ReferenceAttention(8, 2, 1, 4, dtype=torch.float64).eval()
    tokens = torch.randn(4, 5, 8, generator=torch.Generator().manual_seed(166), dtype=torch.float64)
    axis = CoordinateAxis("residual", tuple(range(8)))
    qk_spec = FlatQueryKey(coordinates=(0, 1))
    qk_request = GroupRequest("qk", CompressionRegion("qk:region", ("",),
        (ConsumerCoordinates("", axis, axis),)), ("",), qk_spec, 2)
    qk = _execute(attention, tokens, (qk_request,), source_mode=SourceMode.FROZEN_SNAPSHOT,
        graph_version="g4-functional-reference-qk")
    qk_legacy = transform_method(attention, tokens, qk_spec, rank=2, target_paths=("",))
    torch.testing.assert_close(qk.model(tokens), qk_legacy.model(tokens), rtol=1e-10, atol=1e-10)
    assert qk.model.group_coordinates.tolist() == [[0, 1]]
    assert torch.equal(qk.model.head_map, attention.head_map)
    assert qk.model.d_model == attention.d_model and qk.model.logits_scale == attention.logits_scale
    report = {"status": "succeeded", "scope": "untrained random CPU fixtures; no utility or speedup claim",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "seeds": {"models": 163, "calibration": 164, "heldout": 165, "attention_calibration": 166},
        "calibration_shape": list(calibration.shape), "heldout_shape": list(heldout.shape),
        "heldout_role": "functional comparison only; no model selection or training",
        "basis": {"source_bytes": _bytes(source), "result_bytes": _bytes(shared.model),
            "reloaded_bytes": _bytes(reloaded), "serialized_bytes": len(payload),
            "same_parameter_after_apply": True, "same_parameter_after_reload": True,
            "coordinate_alignment": "caller_declared_assumption", "legacy_parity": True,
            "execution": shared.evidence},
        "sequential": sequential,
        "reference_qk": {"source_bytes": _bytes(attention), "result_bytes": _bytes(qk.model),
            "coordinates": qk.model.group_coordinates.tolist(), "head_map": qk.model.head_map.tolist(),
            "external_residual_width": qk.model.d_model, "legacy_parity": True, "execution": qk.evidence}}
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "succeeded", "result": str(path), "shared_bytes_after_reload": _bytes(reloaded),
        "sequence_contrasts": [item["heldout_mse_against_frozen_assembly"] for item in sequential]}))


if __name__ == "__main__":
    main()
