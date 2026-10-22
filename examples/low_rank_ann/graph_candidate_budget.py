"""Finite graph candidates under a whole-model budget: a CPU integration example.

The deterministic random MLP is a functional fixture, not a trained benchmark.
Local errors guide allocation; separate held-out inputs measure the whole model.
No training, dataset download or GPU execution is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.allocation import BudgetInfeasible
from fedcore.algorithm.low_rank.artifacts import StatisticsStore
from fedcore.algorithm.low_rank.candidate_graph import (
    CandidateRecipe, allocate_candidate_catalog, apply_candidate_allocation,
    prepare_candidate_catalog, verify_candidate_export,
)
from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.experiments.measurement import measure_artifact
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import quality_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    random.seed(161)
    np.random.seed(161)
    torch.manual_seed(161)
    source = nn.Sequential(nn.Linear(12, 10), nn.Tanh(), nn.Linear(10, 6)).double().eval()
    calibration = torch.randn(64, 12, generator=torch.Generator().manual_seed(162), dtype=torch.float64)
    validation = torch.randn(128, 12, generator=torch.Generator().manual_seed(163), dtype=torch.float64)
    with torch.inference_mode():
        labels = source(validation).clone()
    split = TensorSplit(validation, labels, tuple(f"validation:{i}" for i in range(len(validation))))
    recipes = tuple(CandidateRecipe(method, rank=rank)
                    for method in ("weighted_svd", "asvd") for rank in (1, 2, 4))
    store = StatisticsStore(max_bytes=4 * 1024**2)
    try:
        catalog = prepare_candidate_catalog(source, calibration,
            alternatives={"0": recipes, "2": recipes}, store=store,
            graph_version="g2b-functional-example-v1")
        plan = allocate_candidate_catalog(catalog, args.budget_bytes, unit="bytes")
        if isinstance(plan, BudgetInfeasible):
            report = {"status": "infeasible", "allocation": asdict(plan)}
        else:
            result = apply_candidate_allocation(source, calibration, plan, store=store)
            exported = verify_candidate_export(result, format="torch_save_module")
            measurement = measure_artifact(exported.model, validation[:16],
                args.output_dir / "selected-torchscript.pt", format="torchscript", device="cpu",
                warmup=5, repeats=30, threads=2)
            if measurement["status"] != "succeeded":
                raise RuntimeError(f"Whole-model artifact measurement failed: {measurement}")
            with (args.output_dir / "selected-torchscript.pt").open("rb") as stream:
                loaded = torch.jit.load(stream).eval()
            runtime_bytes = inventory_cost(capture_inventory(loaded), "bytes")
            if runtime_bytes > args.budget_bytes:
                raise RuntimeError("Reloaded TorchScript tensor storage exceeds the declared budget")
            report = {
                "status": "succeeded",
                "scope": "functional random-MLP fixture; no trained-model or practical-speedup claim",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "device": "cpu"},
                "seed": 161, "calibration_seed": 162, "validation_seed": 163,
                "calibration_shape": list(calibration.shape), "validation_shape": list(validation.shape),
                "calibration_role": "candidate_construction", "validation_role": "whole_model_evaluation_only",
                "budget_bytes": args.budget_bytes,
                "source_tensor_bytes": inventory_cost(capture_inventory(source), "bytes"),
                "selected_tensor_bytes": inventory_cost(capture_inventory(result.model), "bytes"),
                "torchscript_tensor_bytes_after_reload": runtime_bytes,
                "allocation": {
                    "fingerprint": plan.fingerprint, "catalog_fingerprint": catalog.fingerprint,
                    "context": asdict(catalog.context), "selection": asdict(plan.allocation),
                    "recipes": [
                        {"table": table.table_id, "recipe": asdict(candidate.recipe)}
                        for table in catalog.tables for candidate in table.candidates
                        if any(choice.table_id == table.table_id and choice.option.option_id == candidate.option_id
                               for choice in plan.allocation.selected)
                    ],
                },
                "execution": result.evidence,
                "whole_model_evaluation": {"status": "completed", "model": "reloaded TorchScript",
                                           "quality_role": "validation", "latency_batch_size": 16},
                "module_export": exported.evidence,
                "quality": quality_metrics(loaded, split, "regression", batch_size=16, device="cpu"),
                "measurement": measurement,
            }
    finally:
        store.close()
    destination = args.output_dir / "result.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    main()
