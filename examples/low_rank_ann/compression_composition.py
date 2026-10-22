"""CPU witnesses for representation order, packed residuals and exact storage.

Untrained matrices and independent test inputs demonstrate contracts, not
accuracy on a task or a speedup. The example does not calibrate or train.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.algorithm.low_rank.compression_composition import (
    AddResidual, CheckedComposition, CompositionRecipe, CompositionRequest,
    CompositionSuccess, LowRank, MaterializeDense, PruneMask, QuantizeRemapped,
    execute_composition, prepare_composition,
)
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


def _bytes(model):
    return inventory_cost(capture_inventory(model), "bytes")


def _request(steps, **options):
    return CompositionRequest((CompositionRecipe("0", tuple(steps)),), **options)


def _run(source, steps):
    checked = prepare_composition(source, _request(steps))
    assert type(checked) is CheckedComposition, checked
    result = execute_composition(source, checked)
    assert type(result) is CompositionSuccess, result
    assert result.storage_bytes == checked.expected_storage_bytes == _bytes(result.model)
    return result


def _record(result, inputs):
    manager = CheckpointManager("unused-composition-checkpoint", auto_cleanup=False)
    loaded = manager.deserialize_from_bytes(result.checkpoint_bytes)
    torch.testing.assert_close(loaded(inputs), result.model(inputs), rtol=1e-13, atol=1e-13)
    assert _bytes(loaded) == result.storage_bytes
    return {"fingerprint": result.fingerprint, "tensor_bytes": result.storage_bytes,
        "reloaded_tensor_bytes": _bytes(loaded), "checkpoint_bytes": len(result.checkpoint_bytes),
        "checkpoint_sha256": hashlib.sha256(result.checkpoint_bytes).hexdigest(),
        "history": [asdict(event) for event in result.history],
        "checked_plan": asdict(result.workflow), "reload_execution_verified": True}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(164)
        source = nn.Sequential(nn.Linear(8, 8, dtype=torch.float64)).eval()
        diagonal = nn.Sequential(nn.Linear(2, 2, bias=False, dtype=torch.float64)).eval()
    diagonal[0].weight.copy_(torch.diag(torch.tensor([2., 1.], dtype=torch.float64)))
    inputs = torch.randn(64, 8, generator=torch.Generator().manual_seed(165), dtype=torch.float64)
    diagonal_inputs = torch.eye(2, dtype=torch.float64)
    before = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    prune_then_svd = _run(diagonal, (PruneMask((0,)), LowRank(1)))
    svd_then_prune = _run(diagonal, (LowRank(1), MaterializeDense(), PruneMask((0,))))
    torch.testing.assert_close(prune_then_svd.model(diagonal_inputs), torch.diag(
        torch.tensor([0., 1.], dtype=torch.float64)), rtol=0, atol=0)
    torch.testing.assert_close(svd_then_prune.model(diagonal_inputs), torch.zeros(2, 2,
        dtype=torch.float64), rtol=0, atol=0)
    assert prune_then_svd.fingerprint != svd_then_prune.fingerprint
    pack = QuantizeRemapped(tail_dtype="float64", scale_dtype="float64", compute_dtype="float64")
    svd_then_pack = _run(source, (LowRank(2), pack))
    pack_then_svd = _run(source, (pack, MaterializeDense(), LowRank(2)))
    quantization_order_mse = float((svd_then_pack.model(inputs) - pack_then_svd.model(inputs)).square().mean())
    assert quantization_order_mse > 1e-14
    residual = _run(source, (LowRank(2), pack, AddResidual(1)))
    merged = _run(source, (LowRank(2), pack, AddResidual(1), MaterializeDense()))
    torch.testing.assert_close(residual.model(inputs), merged.model(inputs), rtol=1e-12, atol=1e-12)
    assert svd_then_pack.storage_bytes < residual.storage_bytes < merged.storage_bytes == _bytes(source)
    failure = prepare_composition(source, _request((LowRank(2), pack, AddResidual(1), MaterializeDense()),
        max_storage_bytes=residual.storage_bytes))
    assert type(failure) is ValidationFailure, failure
    assert any(item.code == "BudgetExceeded" for item in failure.violations), failure
    unsupported = prepare_composition(source, _request((LowRank(2),), export_format="onnx"))
    assert type(unsupported) is ValidationFailure, unsupported
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, before[name])
    report = {"status": "succeeded", "scope": "untrained CPU functional witnesses; no training, calibration, utility or speedup claim",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "seeds": {"models": 164, "test_inputs": 165}, "test_input_shape": list(inputs.shape),
        "source_tensor_bytes": _bytes(source),
        "prune_svd_order": {"prune_then_svd": _record(prune_then_svd, diagonal_inputs),
            "svd_then_prune": _record(svd_then_prune, diagonal_inputs),
            "expected_effective_weights": [[[0., 0.], [0., 1.]], [[0., 0.], [0., 0.]]]},
        "quantization_svd_order": {"quantization": "Dobi portable SVD-factor remapping, not dense-weight TorchAO",
            "svd_then_pack": _record(svd_then_pack, inputs), "pack_then_dense_then_svd": _record(pack_then_svd, inputs),
            "test_output_difference_mse": quantization_order_mse},
        "packed_residual": _record(residual, inputs), "dense_merge": _record(merged, inputs),
        "dense_merge_budget_refusal": [asdict(item) for item in failure.violations],
        "unsupported_onnx_refusal": [asdict(item) for item in unsupported.violations], "source_preserved": True}
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "succeeded", "result": str(path), "source_bytes": _bytes(source),
        "packed_bytes": svd_then_pack.storage_bytes, "residual_bytes": residual.storage_bytes,
        "merged_bytes": merged.storage_bytes, "quantization_order_mse": quantization_order_mse}))


if __name__ == "__main__":
    main()
