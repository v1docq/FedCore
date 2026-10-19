"""Closed compression grammar and actual GOLEM search on a tiny CPU fixture.

This untrained random MLP checks integration, not compression quality or speed.
Calibration and held-out inputs use independent seeds. No data are downloaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.algorithm.low_rank.compression_graph import RuntimeOptions
from fedcore.algorithm.low_rank.compression_grammar import (
    ASVDChoice, AffinePCAChoice, CompressionGenotype, MutationKind, ScopeGene,
    WeightedChoice, compile_genotype, crossover_genotypes, genotype_payload,
    mutate_genotype, normalize_genotype,
)
from fedcore.algorithm.low_rank.grammar_execution import (
    ResourceFailure, WorkflowSuccess, execute_workflow, prepare_grammar_space,
)
from fedcore.algorithm.low_rank.grammar_search import (
    GrammarFitness, GrammarSearchConfig, run_grammar_search,
)
from fedcore.algorithm.low_rank.plans import ValidationFailure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    random.seed(162)
    np.random.seed(162)
    torch.manual_seed(162)
    source = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 2)).double().eval()
    calibration = torch.randn(16, 4, generator=torch.Generator().manual_seed(163), dtype=torch.float64) + 2
    heldout = torch.randn(64, 4, generator=torch.Generator().manual_seed(164), dtype=torch.float64) + 2
    with torch.no_grad():
        labels = source(heldout).clone()
    choices = (WeightedChoice(1), WeightedChoice(2), ASVDChoice(1), AffinePCAChoice(1), AffinePCAChoice(2))
    space = prepare_grammar_space(source, calibration, alternatives={"0": choices, "2": choices},
        graph_version="g3-functional-v1", options=RuntimeOptions(tensor_byte_budget=192))
    if isinstance(space, ValidationFailure):
        raise RuntimeError(space)
    weighted = CompressionGenotype((ScopeGene(("0",), WeightedChoice(1)), ScopeGene(("2",), WeightedChoice(1))))
    centered = mutate_genotype(weighted, space, kind=MutationKind.CENTERING, seed=0)
    children = crossover_genotypes(weighted, centered, space, seed=0)
    if isinstance(centered, ValidationFailure) or isinstance(children, ValidationFailure):
        raise RuntimeError("The declared closed operators failed")
    assert normalize_genotype(normalize_genotype(centered, space), space) == centered
    executions = []

    def evaluate(workflow):
        outcome = execute_workflow(source, calibration, workflow)
        if type(outcome) is not WorkflowSuccess:
            return outcome
        with torch.no_grad():
            mse = float(torch.mean((outcome.model(heldout) - labels) ** 2))
        cost = outcome.evidence["actual_backing_storage_bytes"]
        executions.append({"workflow_fingerprint": workflow.fingerprint,
            "genotype": genotype_payload(workflow.genotype), "heldout_mse": mse,
            "actual_backing_storage_bytes": cost})
        return GrammarFitness(mse, cost, quality_accepted=mse <= .05)

    # A single centered branch checks the otherwise easy-to-miss new bias.
    afm = CompressionGenotype((ScopeGene(("0",), AffinePCAChoice(1)),))
    afm_result = execute_workflow(source, calibration, compile_genotype(afm, space))
    assert type(afm_result) is WorkflowSuccess, afm_result
    with torch.no_grad():
        mean_error = float((afm_result.model[0](calibration).mean(0)
                            - source[0](calibration).mean(0)).abs().max())
    assert afm_result.model[0].bias is not None and mean_error < 2e-5
    # A structurally valid workflow can still exceed the physical storage cap.
    over_budget = CompressionGenotype((ScopeGene(("0",), AffinePCAChoice(2)),
                                      ScopeGene(("2",), WeightedChoice(1))))
    refusal = execute_workflow(source, calibration, compile_genotype(over_budget, space))
    assert type(refusal) is ResourceFailure and refusal.code == "BudgetInfeasible", refusal
    initial = tuple(CompressionGenotype((ScopeGene(("0",), choice),
                                         ScopeGene(("2",), WeightedChoice(1))))
                    for choice in choices[:4])
    search = run_grammar_search(space, GrammarSearchConfig(seed=13, population_size=4,
        generations=2, max_evaluations=12, max_seconds=60), evaluate,
        fitness_context="independent heldout:64:seed164; teacher MLP MSE; minimize MSE and actual bytes",
        initial_genotypes=initial)
    if search["status"] != "succeeded":
        raise RuntimeError(search)
    search["best_genotypes"] = [genotype_payload(item) for item in search["best_genotypes"]]
    report = {
        "status": "succeeded", "scope": "untrained random-MLP functional fixture; no utility or speedup claim",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "seeds": {"source": 162, "calibration": 163, "heldout": 164, "search": 13},
        "calibration_shape": list(calibration.shape), "heldout_shape": list(heldout.shape),
        "calibration_role": "construct recipes", "heldout_role": "full-model search fitness; not final test data",
        "source_bytes": inventory_cost(capture_inventory(source), "bytes"), "budget_bytes": 192,
        "centering_mutation": genotype_payload(centered),
        "crossover_children": [genotype_payload(item) for item in children],
        "afm_mean_max_abs_error": mean_error,
        "afm_bytes_with_added_bias": afm_result.evidence["actual_backing_storage_bytes"],
        "over_budget": {"outcome": type(refusal).__name__, "code": refusal.code},
        "successful_evaluations": executions, "search": search,
    }
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result": str(target),
        "charged_evaluations": search["charged_evaluations"], "afm_mean_max_abs_error": mean_error}))


if __name__ == "__main__":
    main()
