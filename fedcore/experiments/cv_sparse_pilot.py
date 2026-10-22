"""Named, bounded ResNet18/CIFAR10 physical-channel/SVD A100 pilot.

Use ``python -m fedcore.experiments.cv_sparse_pilot ...``. Without --execute the
entry point verifies prepared files and provenance without importing ML/CUDA.
No auto-retry, test-set tuning, arbitrary graph training, or speed claim occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from .cv_sparse_protocol import PilotProtocol, digest, eligible_finalists
from .cv_sparse_runtime import (GPUData, Output, PilotFatal, PilotStop, Runtime, admit_free_gpu,
    calibrate_bn, checkpoint, evaluate, paired_latency, preparation, recover, recovery_tape,
    sha256_file, torchscript_artifact, independent_torchscript, train_baseline, verify_loaded_sources)
from .cv_sparse_runtime import validate_selected_artifacts, classify_cv_candidate_failure


def cv_menu():
    from fedcore.algorithm.low_rank.cv_grammar import DenseCV, ChannelPrune, ConvSVD, ChannelPruneSVD
    from fedcore.algorithm.pruning.sparse.cv_channels import BLOCK_PATHS, BLOCK_OUTPUTS
    return {path: (DenseCV(), ChannelPrune(width * 3 // 4), ChannelPrune(width // 2),
        ConvSVD(width // 2), ConvSVD(width // 4), ChannelPruneSVD(width * 3 // 4, width // 2),
        ChannelPruneSVD(width * 3 // 4, width // 4)) for path, width in zip(BLOCK_PATHS, BLOCK_OUTPUTS)}


def fixed_prefix(menu):
    from fedcore.algorithm.low_rank.cv_grammar import CVGenotype, CVScopeGene
    return tuple(CVGenotype(tuple(CVScopeGene(path, choices[index]) for path, choices in menu.items()))
                 for index in (0, 1, 3, 5))


def solution_id(genotype):
    from fedcore.algorithm.low_rank.cv_grammar import cv_genotype_payload
    return digest(cv_genotype_payload(genotype))


def _cv_search_report(result):
    """Project typed search winners at the reporting boundary, without mutation."""
    from fedcore.algorithm.low_rank.cv_grammar import cv_genotype_payload
    winners = []
    for genotype in result["best_genotypes"]:
        payload = cv_genotype_payload(genotype)
        if type(payload) is not dict:
            raise ValueError("A search report requires valid metadata-only CV genotypes")
        winners.append(payload)
    return {**result, "best_genotypes": winners}


class Pilot:
    def __init__(self, runtime, data):
        os.environ["FEDCORE_MODEL_REGISTRY_PATH"] = str(runtime.output.root / "registry")
        from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
        from fedcore.tools.registry.model_registry import ModelRegistry
        self.runtime, self.protocol, self.output, self.data = runtime, runtime.protocol, runtime.output, data
        ModelRegistry(auto_cleanup=False)
        self.facade = LowRankModel({"device": "cpu", "fedcore_id": "bounded-sparse-cv-pilot"})
        self.gpu = GPUData(data, runtime)
        self.calibration = data.rows(data.roles.calibration)
        self.menu, self.prefix = cv_menu(), fixed_prefix(cv_menu())
        self.teacher, self.space, self.teacher_fingerprint = None, None, None
        self.records, self.genotypes, self.artifacts = {}, {}, {}

    def bind_teacher(self, model):
        from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
        from fedcore.algorithm.low_rank.plans import ValidationFailure
        self.teacher = model.cpu().eval().requires_grad_(False)
        self.teacher_fingerprint = inventory_fingerprint(capture_inventory(self.teacher))
        self.space = self.facade.prepare_cv_grammar(self.teacher, self.calibration, alternatives=self.menu)
        if isinstance(self.space, ValidationFailure):
            raise ValueError(f"Dense source failed public CV admission: {self.space}")
        self.byte_ceiling = math.floor(self.protocol.storage_fraction * self.space.source_storage_bytes)

    def build(self, genotype):
        """Every candidate crosses the actual public LowRankModel boundary."""
        from fedcore.algorithm.low_rank.cv_grammar import compile_cv_genotype
        from fedcore.algorithm.low_rank.plans import ValidationFailure
        space = self.facade.prepare_cv_grammar(self.teacher, self.calibration, alternatives=self.menu)
        if isinstance(space, ValidationFailure):
            return space
        if space != self.space:
            raise ValueError("The immutable dense teacher or calibration changed")
        workflow = compile_cv_genotype(genotype, space)
        if isinstance(workflow, ValidationFailure):
            return workflow
        return self.facade.compress_cv_workflow(self.teacher, self.calibration, workflow)

    def short(self, genotype, run, deadline):
        from fedcore.algorithm.low_rank.compression_graph import _canonical
        from fedcore.algorithm.low_rank.cv_grammar import cv_genotype_payload
        from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
        from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure, StructuralInvalid, NumericalFailure
        from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
        from fedcore.algorithm.low_rank.plans import ValidationFailure
        from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
        rows = self.records.setdefault(run, [])
        key, started = solution_id(genotype), time.perf_counter()
        self.genotypes[key] = genotype
        record = {"run": run, "attempt_id": len(rows) + 1, "solution_id": key, "genotype": cv_genotype_payload(genotype),
                  "started_at": started, "status": "started", "quality_accepted": False, "deadline": deadline,
                  "charged_attempt": True}
        self.output.append({"event": "short_started", **record})
        outcome = None
        try:
            self.runtime.check(deadline, run=run, attempt=record["attempt_id"])
            result = self.build(genotype)
            if type(result) is not CVWorkflowSuccess:
                known = type(result) in (ValidationFailure, StructuralInvalid, NumericalFailure, ResourceFailure)
                original = asdict(result) if known else {"unknown_type": type(result).__name__}
                disposition = classify_cv_candidate_failure(result)
                record.update(status="fatal" if disposition == "fatal" else "failed",
                    failure_type=type(result).__name__, failure=original, failure_disposition=disposition,
                    build_outcome={"type": type(result).__name__, "metadata": original})
                if disposition == "fatal":
                    raise PilotFatal("Fatal CV build outcome: " + json.dumps(record["build_outcome"], sort_keys=True, allow_nan=False))
                outcome = result
            else:
                record.update(workflow_fingerprint=result.workflow.fingerprint,
                    tensor_bytes=result.evidence["storage_bytes"], source_tensor_bytes=self.space.source_storage_bytes,
                    workflow=self.output.json(f"workflows/{run}/{key}.json", _canonical(result.workflow)))
                self.runtime.check(deadline, run=run, attempt=record["attempt_id"], phase="after_cpu_build")
                model = result.model.to("cuda")
                calibrate_bn(model, self.gpu, deadline)
                metrics = evaluate(model, self.gpu, self.data.roles.search_validation, deadline=deadline, label=f"short_{run}_{key}")
                record.update(metrics)
                artifact, _restored = checkpoint(model, self.gpu, label=f"short_{run}_{key}", solution_id=key, phase="short", deadline=deadline)
                record.update(checkpoint=artifact, checkpoint_file_bytes=artifact["file_bytes"],
                    cost_ratio=record["tensor_bytes"] / self.space.source_storage_bytes,
                    storage_accepted=record["tensor_bytes"] <= self.byte_ceiling, status="succeeded", quality_accepted=True)
                if artifact["tensor_bytes"] != record["tensor_bytes"]:
                    raise ValueError("BN calibration or checkpoint changed whole-model physical bytes")
                outcome = GrammarFitness(metrics["cross_entropy"], record["cost_ratio"], quality_accepted=True)
            if inventory_fingerprint(capture_inventory(self.teacher)) != self.teacher_fingerprint:
                raise ValueError("Short evaluation mutated the dense teacher")
        except PilotStop as error:
            record.update(status="incomplete", failure_type="ResourceFailure", failure=str(error))
            if error.reason != "absolute_deadline":
                raise
            outcome = ResourceFailure("EvaluationDeadline", str(error))
        except Exception as error:
            record.update(status="fatal", failure_type=type(error).__name__, failure=str(error))
            raise PilotFatal(f"Unknown short-evaluation failure: {type(error).__name__}: {error}") from error
        finally:
            ended = time.perf_counter()
            record.update(ended_at=ended, wall_seconds=ended - started,
                completed_within_deadline=ended <= min(deadline, self.runtime.deadline, self.runtime.stage_limit))
            if not record["completed_within_deadline"] and record["status"] == "succeeded":
                record["status"] = "late"
            rows.append(record)
            self.output.append({"event": "short_ended", **record})
            self.output.json(f"search/{run}_attempts.json", rows)
        if time.perf_counter() > min(deadline, self.runtime.deadline, self.runtime.stage_limit):
            record["completed_within_deadline"] = False
            if record["status"] == "succeeded":
                record["status"] = "late"
            self.output.append({"event": "short_deadline_excluded_after_journal", "run": run, "solution_id": key,
                "attempt_id": record["attempt_id"], "timestamp": time.perf_counter()})
            self.output.json(f"search/{run}_attempts.json", rows)
        if not record["completed_within_deadline"]:
            return ResourceFailure("EvaluationDeadline", "All expenses retained; late observation is excluded")
        return outcome

    def random_search(self, deadline):
        from fedcore.algorithm.low_rank.cv_grammar import generate_cv_genotype, cv_genotype_payload
        rng, cache, proposals = random.Random(self.protocol.search_seed), set(), 0
        run = "random"
        self.records[run] = []
        self.output.append({"event": "search_started", "run": run, "cache_entries": 0, "deadline": deadline})
        while len(self.records[run]) < self.protocol.search_evaluations and proposals < self.protocol.search_proposals:
            if time.perf_counter() >= deadline:
                break
            genotype = self.prefix[proposals] if proposals < len(self.prefix) else generate_cv_genotype(self.space, seed=rng.randrange(2**32))
            proposals += 1
            key = solution_id(genotype)
            self.output.append({"event": "proposal", "run": run, "proposal_id": proposals, "solution_id": key,
                "genotype": cv_genotype_payload(genotype), "operator": "shared_prefix" if proposals <= 4 else "uniform_independent_sampling",
                "cache_hit": key in cache})
            if key in cache:
                continue
            cache.add(key)
            self.short(genotype, run, deadline)
        result = {"engine": "uniform_random_over_declared_finite_CV_grammar", "run": run,
            "charged_evaluations": len(self.records[run]), "created_proposals": proposals, "cache_entries": len(cache),
            "deadline": deadline, "ended_at": time.perf_counter(), "attempts": self.records[run]}
        self.output.json("search/random.json", result)
        return result

    def evolution(self, deadline):
        from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig, run_cv_grammar_search
        run = "evolution"
        self.records[run] = []
        result = run_cv_grammar_search(self.space,
            GrammarSearchConfig(seed=self.protocol.search_seed, population_size=self.protocol.search_population,
                generations=self.protocol.search_generations, max_evaluations=self.protocol.search_evaluations,
                max_proposals=self.protocol.search_proposals, max_seconds=self.protocol.search_seconds),
            lambda workflow: self.short(workflow.genotype, run, deadline),
            fitness_context="fixed image-only BN calibration; 5000 heldout CE and physical bytes/M0; 0 SGD updates",
            initial_genotypes=self.prefix, absolute_deadline=deadline,
            trace_callback=lambda event: self.output.append({"event": "evolution_trace", "trace": event}))
        # Common-search completion includes the callback journal overhead. It
        # is the authoritative eligibility bound for the evolution archive.
        by_solution = {record["solution_id"]: record for record in self.records[run]}
        for event in result["trace"]:
            if event["event"] == "evaluated" and event["canonical_solution_id"] in by_solution:
                record = by_solution[event["canonical_solution_id"]]
                record["common_search_completed_at"] = event["ended_at"]
                record["completed_within_deadline"] = record["completed_within_deadline"] and event["completed_within_deadline"]
                if not record["completed_within_deadline"] and record["status"] == "succeeded":
                    record["status"] = "late"
        self.output.json("search/evolution_attempts.json", self.records[run])
        self.output.json("search/evolution.json", _cv_search_report(result))
        if result["status"] == "failed":
            raise PilotFatal(f"Evolution stopped on an unknown optimizer/callback failure: {result.get('reason')}")
        return result


def run_pilot(data_root, output_root, source_manifest, *, started=None, protocol=PilotProtocol()):
    started = time.perf_counter() if started is None else started
    prepared = preparation(data_root, output_root, source_manifest, protocol)
    if prepared["status"] != "prepared":
        raise ValueError("CIFAR files must be prepared before an execution slot")
    output = Output(output_root)
    report = {"schema": protocol.schema, "status": "started", "started_at": started, "protocol_sha256": protocol.sha256,
        "model_seed": protocol.model_seed, "search_seed": protocol.search_seed,
        "scope": "one model seed and one search seed; no statistical significance or general search-utility claim",
        "gpu_sparse_kernel_claim": False, "latency_objective": False}
    output.json("protocol.json", protocol.payload())
    output.json("source_manifest.json", prepared["source"])
    runtime, pilot, baseline_artifact = None, None, None
    try:
        if time.perf_counter() >= started + min(15 * 60, protocol.max_slot_seconds):
            raise PilotStop("manifest", "deadline_before_gpu_admission")
        report["gpu_admission"] = admit_free_gpu()
        runtime = Runtime(protocol, output, started)
        report["absolute_deadline"] = runtime.deadline
        with runtime.phase("manifest"):
            from .cv_sparse_data import CifarData
            data = CifarData(data_root, protocol, download=False)
            pilot = Pilot(runtime, data)
            report["environment"] = runtime.environment
            report["data"] = {"roles_sha256": data.roles.sha256, "source_files": data.source_hashes}
            output.json("data_roles.json", data.roles.payload())
            output.json("environment.json", runtime.environment)
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True, timeout=30).stdout
            output.bytes("environment_freeze.txt", freeze.encode())
            output.json("imported_sources.json", verify_loaded_sources(prepared["source"]))
        with runtime.phase("baseline") as deadline:
            model, validation, training = train_baseline(pilot.gpu, deadline)
            baseline_key = solution_id(pilot.prefix[0])
            baseline_artifact, teacher = checkpoint(model, pilot.gpu, label="dense_original", solution_id=baseline_key,
                phase="dense_original", deadline=deadline)
            baseline_artifact.update(validation=validation, training=training)
            pilot.bind_teacher(teacher)
            pilot.genotypes[baseline_key] = pilot.prefix[0]
            report.update(baseline=baseline_artifact, byte_ceiling=pilot.byte_ceiling, source_tensor_bytes=pilot.space.source_storage_bytes)
            output.json("baseline.json", baseline_artifact)
        initial_aa = {"accepted": False, "reason": "not_completed"}
        with runtime.phase("admission") as deadline:
            try:
                baseline_ts, baseline_ts_record = torchscript_artifact(baseline_artifact, pilot.gpu, deadline)
                baseline_ts_copy = independent_torchscript(baseline_ts_record, pilot.gpu, deadline)
                initial_aa = paired_latency(baseline_ts, baseline_ts_copy, pilot.gpu, deadline=deadline, label="initial_dense_AA", aa=True)
                del baseline_ts, baseline_ts_copy
                report["baseline_torchscript"] = baseline_ts_record
            except PilotStop as error:
                if error.reason != "absolute_deadline" or time.perf_counter() >= runtime.deadline:
                    raise
                initial_aa = {"accepted": False, "reason": "admission_stage_deadline", "details": str(error)}
                output.json("latency/initial_dense_AA_incomplete.json", initial_aa)
        report["initial_aa"] = initial_aa
        with runtime.phase("controls") as deadline:
            for label, genotype in zip(("fixed_prune", "fixed_svd", "fixed_hybrid"), pilot.prefix[1:]):
                runtime.check(deadline)
                pilot.short(genotype, label, deadline)
        with runtime.phase("search") as deadline:
            random_deadline = min(deadline, time.perf_counter() + protocol.search_seconds)
            report["random"] = pilot.random_search(random_deadline)
            runtime.check(deadline)
            evolution_deadline = min(deadline, time.perf_counter() + protocol.search_seconds)
            report["evolution"] = _cv_search_report(pilot.evolution(evolution_deadline))
            output.json("imported_sources.json", verify_loaded_sources(prepared["source"]))
        finalists = {}
        for run in ("random", "evolution"):
            eligible = eligible_finalists(pilot.records[run], byte_ceiling=pilot.byte_ceiling)
            finalists[run] = eligible[0]["solution_id"] if eligible else None
        report["finalist_solution_ids"] = finalists
        selected = {}
        for role, genotype in zip(("dense_recovery", "fixed_prune", "fixed_svd", "fixed_hybrid"), pilot.prefix):
            key = solution_id(genotype)
            pilot.genotypes[key] = genotype
            selected.setdefault(key, []).append(role)
        for role, key in finalists.items():
            if key is not None:
                selected.setdefault(key, []).append(role + "_finalist")
        if len(selected) > 6:
            raise ValueError("Frozen recovery cap exceeded")
        tape = recovery_tape(data.roles.train, protocol)
        output.json("recovery_tape.json", {"sha256": digest(tape), "updates": [{"row_ids": list(ids), "augmentation_seed": seed} for ids, seed in tape]})
        recovered = {}
        report["recovered"] = recovered
        with runtime.phase("recovery") as deadline:
            from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
            for key, roles in selected.items():
                runtime.check(deadline, recovery_solution=key)
                result = pilot.build(pilot.genotypes[key])
                if type(result) is not CVWorkflowSuccess:
                    raise PilotFatal(f"A selected recovery graph failed compilation: {result}")
                metrics = recover(result.model, pilot.gpu, tape, deadline=deadline, role=key)
                artifact, _restored = checkpoint(result.model, pilot.gpu, label=f"recovered_{key}", solution_id=key,
                    phase="recovered_200_updates", deadline=deadline)
                artifact.update(roles=roles, validation=metrics, recovery_updates=protocol.recovery_updates,
                    recovery_tape_sha256=digest(tape))
                recovered[key] = artifact
                output.json(f"recovery/{key}.json", artifact)
        report["recovered"] = recovered
        with runtime.phase("final") as deadline:
            from .cv_sparse_data import FrozenSelection
            all_artifacts = {baseline_artifact["artifact_id"]: baseline_artifact,
                             **{artifact["artifact_id"]: artifact for artifact in recovered.values()}}
            runtime.check(deadline)
            artifact_admission = validate_selected_artifacts(output, all_artifacts, protocol)
            output.json("selected_artifacts_admission.json", artifact_admission)
            runtime.check(deadline)
            roles = [("dense_original", baseline_artifact["artifact_id"])]
            roles.extend((role, artifact["artifact_id"]) for artifact in recovered.values() for role in artifact["roles"])
            selection = FrozenSelection(protocol.sha256, tuple(sorted(roles)),
                tuple(sorted((key, artifact["sha256"]) for key, artifact in all_artifacts.items())))
            output.json("frozen_selection.json", {**asdict(selection), "solution_ids": {key: artifact["solution_id"] for key, artifact in all_artifacts.items()},
                "frozen_at": time.perf_counter(), "all_artifacts": all_artifacts})
            # No official test labels or tensors are opened before this durable seal.
            runtime.check(deadline)
            test_data = data.open_test(selection)
            report["test_opened_after_frozen_selection"] = True
            from fedcore.tools.registry.checkpoint_manager import CheckpointManager
            test_results = {}
            for key, artifact in all_artifacts.items():
                runtime.check(deadline, test_artifact=key)
                path = output.root / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError("A frozen checkpoint changed before official test")
                model = CheckpointManager(".", auto_cleanup=False).deserialize_from_bytes(path.read_bytes(), device="cuda").eval()
                test_results[key] = evaluate(model, pilot.gpu, tuple(range(10000)), deadline=deadline,
                    label=f"official_test_{key}", test_data=test_data)
                output.json(f"official_test/{key}.json", test_results[key])
                model.cpu()
            report["official_test"] = test_results
            dense_recovery = next(artifact for artifact in recovered.values() if "dense_recovery" in artifact["roles"])
            threshold = test_results[dense_recovery["artifact_id"]]["top1"] - protocol.final_quality_drop
            report["final_quality_threshold"] = threshold
            report["final_acceptance"] = {key: {"storage_accepted": artifact["tensor_bytes"] <= pilot.byte_ceiling,
                "quality_accepted": test_results[key]["top1"] >= threshold,
                "accepted": artifact["tensor_bytes"] <= pilot.byte_ceiling and test_results[key]["top1"] >= threshold}
                for key, artifact in all_artifacts.items()}
            # At most four distinct report-only latency artifacts, each admitted
            # against itself before any comparison against the dense baseline.
            timing_artifacts = [baseline_artifact]
            for role in ("fixed_hybrid", "random_finalist", "evolution_finalist"):
                candidate = next((artifact for artifact in recovered.values() if role in artifact["roles"]), None)
                if candidate is not None and candidate["artifact_id"] not in {item["artifact_id"] for item in timing_artifacts}:
                    timing_artifacts.append(candidate)
            latency = {}
            try:
                baseline_ts, baseline_record = torchscript_artifact(baseline_artifact, pilot.gpu, deadline)
                reference_aa_passed = initial_aa["accepted"]
                for artifact in timing_artifacts:
                    runtime.check(deadline)
                    candidate_ts, ts_record = torchscript_artifact(artifact, pilot.gpu, deadline)
                    key = artifact["artifact_id"]
                    candidate_ts_copy = independent_torchscript(ts_record, pilot.gpu, deadline)
                    aa = paired_latency(candidate_ts, candidate_ts_copy, pilot.gpu, deadline=deadline, label=f"final_{key}_AA", aa=True)
                    del candidate_ts_copy
                    latency[key] = {"torchscript": ts_record, "AA": aa}
                    if key == baseline_artifact["artifact_id"]:
                        reference_aa_passed = reference_aa_passed and aa["accepted"]
                    if aa["accepted"] and reference_aa_passed:
                        latency[key]["AB"] = paired_latency(baseline_ts, candidate_ts, pilot.gpu, deadline=deadline, label=f"final_{key}_AB")
                    latency[key]["comparison_admitted"] = aa["accepted"] and reference_aa_passed
                    del candidate_ts
                del baseline_ts
            except PilotStop as error:
                if error.reason != "absolute_deadline" or time.perf_counter() >= runtime.deadline:
                    raise
                latency["incomplete"] = {"reason": "final_stage_deadline", "details": str(error)}
            report["latency"] = latency
            report["latency_claim"] = "paired raw measurements only; failed/incomplete AA prohibits speed claims"
        with runtime.phase("close"):
            from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
            if inventory_fingerprint(capture_inventory(pilot.teacher)) != pilot.teacher_fingerprint:
                raise ValueError("Final teacher storage identity differs from frozen M0")
            report["status"] = "completed"
            report["search_claim"] = "functional comparison at one search seed; no claim that evolution outperforms uniform random"
    except PilotStop as error:
        report.update(status="pilot_incomplete_" + error.stage, stop_reason=error.reason)
    except (Exception, PilotFatal) as error:
        report.update(status="failed", failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
    finally:
        report["ended_at"] = time.perf_counter()
        report["wall_seconds"] = report["ended_at"] - started
        report["charged_slot_seconds"] = report["wall_seconds"]
        if runtime is not None:
            report["peak_allocated_bytes"] = runtime.torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = runtime.torch.cuda.max_memory_reserved()
            report["within_slot_deadline"] = report["ended_at"] <= runtime.deadline
        if pilot is not None:
            report["all_short_attempts"] = pilot.records
        output.json("report.json", report)
        output.append({"event": "pilot_stopped", "status": report["status"], "ended_at": report["ended_at"], "wall_seconds": report["wall_seconds"]})
    return report


def main(argv=None, *, execute_started=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol = PilotProtocol()
    if not args.execute:
        result = preparation(args.data_root, args.output_root, args.source_manifest, protocol)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["status"] == "prepared" else 2
    result = run_pilot(args.data_root, args.output_root, args.source_manifest,
        started=execute_started if execute_started is not None else time.perf_counter(), protocol=protocol)
    print(json.dumps({"status": result["status"], "report": str(Path(args.output_root).resolve() / "report.json")}, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    # The execution clock begins before any ML imports; the external supervisor's
    # earlier absolute deadline narrows it further when supplied.
    raise SystemExit(main(execute_started=time.perf_counter()))
