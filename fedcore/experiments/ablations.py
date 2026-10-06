"""Paired, real-operator ablations; resource and quality hypotheses stay separate."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import threading
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .math_checks import approximate_layer, factor_diagnostics, orthogonal_penalty, matched_rank_policy_audit
from .streaming import capture_moments, layer_error, model_error


def _hash_model(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _replace(model, path, layer):
    if not path:
        return layer
    parent, _, leaf = path.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, layer)
    return model


def _loader(split, batch_size):
    return DataLoader(TensorDataset(split.x, split.y), batch_size=batch_size, shuffle=False)


def _loss(task):
    if task == "classification":
        return nn.CrossEntropyLoss()
    if task in ("regression", "forecasting"):
        return nn.MSELoss()
    raise ValueError("These training ablations support tensor classification/regression/forecasting")


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _quality(model, split, task, batch_size=16):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Quality batch size must be a positive integer")
    if task not in ("classification", "regression", "forecasting"):
        raise ValueError("Ablation quality supports classification/regression/forecasting")
    total, count = 0., 0
    device = next(model.parameters(), split.x).device
    with torch.inference_mode():
        for x, target in _loader(split, batch_size):
            output = model(x.to(device)).detach().cpu()
            if not torch.isfinite(output).all():
                raise ValueError("Quality predictions must be finite")
            if task == "classification":
                if output.ndim != 2 or target.ndim != 1 or len(output) != len(target):
                    raise ValueError("Classification requires [N,C] scores and [N] labels")
                total += int((output.argmax(-1) == target).sum())
                count += len(target)
            else:
                if output.shape != target.shape:
                    raise ValueError("Regression predictions and targets must have identical shape")
                total += float((output.double() - target).square().sum())
                count += target.numel()
    return {"name": "accuracy" if task == "classification" else "mean_squared_error",
            "value": total / count, "role": "validation", "samples": len(split.x)}


def _train(model, split, protocol, *, regularizer=None, criterion=None):
    model.train()
    device = torch.device(protocol.device)
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=protocol.learning_rate)
    criterion = criterion or _loss("classification" if split.y.dtype == torch.int64 else "regression")
    history = []
    for _ in range(protocol.finetune_epochs):
        for x, y in _loader(split, protocol.batch_size):
            optimizer.zero_grad()
            prediction = model(x.to(device))
            supervised = criterion(prediction, y.to(device))
            penalty = regularizer(model) if regularizer else supervised * 0
            loss = supervised + penalty
            if loss.ndim or not torch.isfinite(loss):
                raise ValueError("Training must produce a finite scalar loss")
            loss.backward()
            optimizer.step()
            history.append({"supervised": float(supervised.detach()), "penalty": float(penalty.detach())})
    return model.eval(), history


def _save_report(root, name, report):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _prepare_baseline(bundle, protocol, output_dir, checkpoint=None):
    from .baseline import prepare_baseline
    return prepare_baseline(bundle, protocol, output_dir, checkpoint=checkpoint)


class _TrainingRSS:
    """Sample process RSS, including tensors/optimizer; not an allocator peak."""
    def __init__(self, interval=.002):
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None
        self.samples = []
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None

    def __enter__(self):
        if self.process is not None:
            self.samples.append(self.process.memory_info().rss)
            def sample():
                while not self.stop.wait(self.interval):
                    self.samples.append(self.process.memory_info().rss)
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            self.samples.append(self.process.memory_info().rss)

    def report(self):
        if not self.samples:
            return {"status": "unsupported", "reason": "psutil process RSS is unavailable"}
        return {"status": "sampled", "peak_bytes": max(self.samples), "before_bytes": self.samples[0],
                "after_bytes": self.samples[-1], "sampling_interval_seconds": self.interval,
                "method": "process RSS; includes baseline, optimizer and library buffers; short peaks may be missed"}


def run_rank_ablation(bundle, protocol, output_dir, *, layer_path="", ranks=(1,), ridge=0.0,
                      moment_max_bytes=256 * 1024 * 1024, baseline_checkpoint=None):
    """Calibration fits M; validation evaluates layer and network before/after FT.

    No access to bundle.test is made. The caller freezes a selected configuration
    before using its independent test split. Existing rank policies are audited
    on the real operator; matched-rank comparisons hold factor storage fixed.
    """
    from .measurement import measure_artifact
    experiment_start = time.perf_counter()
    original, baseline_training = _prepare_baseline(bundle, protocol, output_dir, baseline_checkpoint)
    layer = original.get_submodule(layer_path) if layer_path else original
    moment_start = time.perf_counter()
    moments, moment_memory = capture_moments(original, layer, bundle.calibration, protocol.batch_size,
                                             max_bytes=moment_max_bytes)
    moment_seconds = time.perf_counter() - moment_start
    records = []
    policy_audits = {str(rank): matched_rank_policy_audit(layer, None, rank) for rank in ranks}
    for rank in ranks:
        for weighted in (False, True):
            start = time.perf_counter()
            replacement, diagnostics = approximate_layer(layer, None, rank, weighted=weighted, ridge=ridge,
                                                          calibration_moments=moments if weighted else None)
            approximation_seconds = time.perf_counter() - start
            candidate = _replace(deepcopy(original), layer_path, replacement).eval()
            error_before = layer_error(original, layer, replacement, bundle.validation, protocol.batch_size)
            quality_before = _quality(candidate, bundle.validation, bundle.task, protocol.batch_size)
            factors_before = factor_diagnostics(candidate)
            with torch.random.fork_rng():
                torch.manual_seed(protocol.seed)
                _synchronize(protocol.device)
                train_start = time.perf_counter()
                candidate, history = _train(candidate, bundle.train, protocol, criterion=_loss(bundle.task))
                _synchronize(protocol.device)
                train_seconds = time.perf_counter() - train_start
            final_layer = candidate.get_submodule(layer_path) if layer_path else candidate
            error_after = layer_error(original, layer, final_layer, bundle.validation, protocol.batch_size)
            label = f"{'weighted' if weighted else 'ordinary'}-r{rank}"
            measurement = measure_artifact(candidate, bundle.validation.x[:protocol.batch_size],
                                           Path(output_dir) / (label + '.pt'), format=protocol.artifact_format,
                                           device=protocol.device, repeats=protocol.measurement_repeats, warmup=protocol.warmup)
            records.append({"variant": label, "requested_rank": rank, "groups": diagnostics,
                            "calibration_source": "calibration",
                            "calibration_seconds": approximation_seconds + (moment_seconds if weighted else 0.),
                            "operator_approximation_seconds": approximation_seconds,
                            "second_moment_seconds": moment_seconds if weighted else 0.,
                            "calibration_cost_accounting": "weighted variant charged full shared moment capture; ordinary SVD requires no capture",
                            "layer_error_before": error_before, "layer_error_after": error_after,
                            "quality_before": quality_before, "quality_after": _quality(candidate, bundle.validation, bundle.task, protocol.batch_size),
                            "factors_before": factors_before, "factors_after": factor_diagnostics(candidate),
                            "training_seconds": train_seconds, "training_steps": len(history), "measurement": measurement})
    report = {"experiment": "rank_second_moment", "layer_path": layer_path, "source_sha256": _hash_model(original), "rank_policy_audits": policy_audits,
              "protocol": asdict(protocol), "baseline_training": baseline_training, "ridge": ridge, "records": records,
              "activation_memory": moment_memory,
              "shared_second_moment_preparation_seconds": moment_seconds,
              "actual_wall_seconds": time.perf_counter() - experiment_start,
              "claim_status": {"layer_error": "measured_on_validation", "network_quality_advantage": "not_established"}}
    return _save_report(output_dir, "rank_ablation.json", report)


@dataclass(frozen=True)
class RegularizationVariant:
    name: str
    coefficient: float
    normalization: str | None = None

    def __post_init__(self):
        if self.name not in ("none", "hoyer", "orthogonal", "norm", "lai_mse", "lai_mae"):
            raise ValueError("Unsupported regularization variant; experimental manifold losses are excluded")
        if isinstance(self.coefficient, bool) or not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be explicit, finite and nonnegative")
        if self.name == "orthogonal" and self.normalization not in ("rank", "rank_squared"):
            raise ValueError("Orthogonality normalization must be explicit")
        if self.name != "orthogonal" and self.normalization is not None:
            raise ValueError("Only orthogonality has rank normalization")
        if self.name.startswith("lai") and self.coefficient <= 0:
            raise ValueError("Lai residual-loss factor must be positive")


def _regularizer(variant):
    from fedcore.losses.low_rank_loss import HoyerLoss
    from fedcore.losses.regularization_losses import NormLoss
    if variant.name == "hoyer":
        return HoyerLoss(variant.coefficient)
    if variant.name == "orthogonal":
        return lambda model: orthogonal_penalty(model, normalization=variant.normalization, coefficient=variant.coefficient)
    if variant.name == "norm":
        # NormLoss normally sees dense weight attributes; expose actual composed
        # factors as differentiable tensors, avoiding detached reassembly.
        def norm(model):
            from fedcore.models.network_impl.decomposed_layers import IDecomposed
            view = nn.Module()
            for index, layer in enumerate(model.modules()):
                weight = layer._get_composed_weight() if isinstance(layer, IDecomposed) else getattr(layer, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
                    leaf = nn.Module()
                    leaf.weight = weight
                    view.add_module(str(index), leaf)
            return NormLoss(variant.coefficient)(view)
        return norm
    return None


def run_regularization_ablation(bundle, protocol, output_dir, *, variants, layer_path="", rank=1, baseline_checkpoint=None):
    from fedcore.losses.regularization_losses import LaiMSE, LaiMAE
    from .measurement import measure_artifact
    records = []
    original, baseline_training = _prepare_baseline(bundle, protocol, output_dir, baseline_checkpoint)
    layer = original.get_submodule(layer_path) if layer_path else original
    replacement, _ = approximate_layer(layer, None, rank)
    common = _replace(deepcopy(original), layer_path, replacement)
    common_hash = _hash_model(common)
    for index, variant in enumerate(variants):
        candidate = deepcopy(common)
        criterion = _loss(bundle.task)
        if variant.name.startswith("lai"):
            if bundle.task == "classification":
                raise ValueError("Lai is a residual loss; it is supported only for regression/forecasting")
            criterion = (LaiMSE if variant.name == "lai_mse" else LaiMAE)(variant.coefficient)
        before = {"quality": _quality(candidate.eval(), bundle.validation, bundle.task, protocol.batch_size), "factors": factor_diagnostics(candidate)}
        with torch.random.fork_rng():
            torch.manual_seed(protocol.seed)
            _synchronize(protocol.device)
            start = time.perf_counter()
            candidate, history = _train(candidate, bundle.train, protocol, regularizer=_regularizer(variant), criterion=criterion)
            _synchronize(protocol.device)
            elapsed = time.perf_counter() - start
        records.append({"variant": asdict(variant), "starting_sha256": common_hash,
                        "supervised_loss": variant.name if variant.name.startswith("lai") else type(criterion).__name__,
                        "coefficient_role": "residual_weight_factor" if variant.name.startswith("lai") else "additive_penalty",
                        "before": before, "after": {"quality": _quality(candidate, bundle.validation, bundle.task, protocol.batch_size), "factors": factor_diagnostics(candidate)},
                        "training_seconds": elapsed, "steps": len(history), "history": history,
                        "measurement": measure_artifact(candidate, bundle.validation.x[:protocol.batch_size],
                             Path(output_dir) / f"regularization-{index}.pt", format=protocol.artifact_format, device=protocol.device,
                             repeats=protocol.measurement_repeats, warmup=protocol.warmup)})
    return _save_report(output_dir, "regularization_ablation.json", {"experiment": "regularization", "protocol": asdict(protocol),
                        "baseline_training": baseline_training, "coefficient_selection_role": "validation_only", "rank": rank, "records": records, "claim_status": "not_established"})


def compare_validation_timing(proposals, validate, execute, *, summarize=None):
    """Replay the identical finite proposal stream; use the existing checker.

    validate(proposal)->bool and execute(proposal)->dict are provided by the
    supported runtime. Late validation really executes invalid proposals. A
    prediction/execution mismatch stops the study instead of reporting speedup.
    """
    proposals = tuple(proposals)
    runs = {}
    for mode in ("early", "late"):
        records = []
        start = time.perf_counter()
        for index, proposal in enumerate(proposals):
            check_start = time.perf_counter()
            predicted = bool(validate(proposal))
            check_seconds = time.perf_counter() - check_start
            result, error, execution_seconds = None, None, 0.
            if mode == "late" or predicted:
                execute_start = time.perf_counter()
                try:
                    result = execute(proposal)
                except (ValueError, RuntimeError, TypeError) as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                execution_seconds = time.perf_counter() - execute_start
            actual_success = result is not None and result.get("status", "succeeded") in ("succeeded", "completed")
            records.append({"proposal_index": index, "predicted_valid": predicted,
                            "executed": mode == "late" or predicted, "success": actual_success,
                            "check_seconds": check_seconds, "execution_seconds": execution_seconds,
                            "result": result, "error": error})
        runs[mode] = {"records": records, "wall_seconds": time.perf_counter() - start,
                      "successful_fraction": sum(r["success"] for r in records) / len(records) if records else None,
                      "failed_execution_seconds": sum(r["execution_seconds"] for r in records if r["executed"] and not r["success"])}
        if summarize is not None:
            runs[mode]["archive"] = summarize([r["result"] for r in records if r["success"]])
    mismatches = [r for r in runs["late"]["records"] if r["predicted_valid"] != r["success"]]
    return {"experiment": "early_late_validation", "proposal_count": len(proposals), "distribution": "identical_ordered_stream",
            "runs": runs, "mismatches": mismatches, "status": "stop_validity_mismatch" if mismatches else "completed",
            "claim_status": "not_established", "budget_mode": "equal_proposal_count; wall time measured separately"}


def run_order_ablation(bundle, protocol, output_dir, *, pruning_amount=.25, rank=1, baseline_checkpoint=None):
    """Pr→LR versus LR→Pr through the common real candidate interpreter."""
    from .runner import apply_candidate
    from .protocol import CandidateSpec
    from .measurement import measure_artifact
    pruning = CandidateSpec("pruning", {"amount": pruning_amount, "finetune_epochs": 0})
    low_rank = CandidateSpec("svd", {"rank": rank, "finetune_epochs": 0})
    candidates = [CandidateSpec("chain", chain=chain) for chain in ((pruning, low_rank), (low_rank, pruning))]
    original, baseline_training = _prepare_baseline(bundle, protocol, output_dir, baseline_checkpoint)
    records = []
    for candidate in candidates:
        with torch.random.fork_rng():
            torch.manual_seed(protocol.seed)
            _synchronize(protocol.device)
            start = time.perf_counter()
            model, evidence = apply_candidate(original, bundle, protocol, candidate)
            model, history = _train(model, bundle.train, protocol, criterion=_loss(bundle.task))
            _synchronize(protocol.device)
        records.append({"chain": [step.to_dict() for step in candidate.chain], "parameters": dict(candidate.parameters), "operations": evidence,
                        "wall_seconds": time.perf_counter() - start, "steps": len(history), "quality": _quality(model, bundle.validation, bundle.task, protocol.batch_size),
                        "factors": factor_diagnostics(model),
                        "measurement": measure_artifact(model, bundle.validation.x[:protocol.batch_size], Path(output_dir) / f"order-{len(records)}.pt",
                              format=protocol.artifact_format, device=protocol.device, repeats=protocol.measurement_repeats, warmup=protocol.warmup)})
    return _save_report(output_dir, "ordering_ablation.json", {"experiment": "operator_order", "records": records,
             "protocol": asdict(protocol), "baseline_training": baseline_training, "claim_status": "not_established"})


def run_validity_ablation(bundle, protocol, output_dir, *, proposals, validator, baseline_checkpoint=None):
    """Real fixed-stream replay with an explicitly supplied existing checker.

    This adapter does not define a second applicability checker. It evaluates
    checker predictions against actual apply_candidate execution, artifact
    loading, validation quality and the same archive summary as the runner.
    """
    from .runner import apply_candidate, quality_metrics
    from .measurement import measure_artifact
    from .search import archive_summary
    original, baseline_training = _prepare_baseline(bundle, protocol, output_dir, baseline_checkpoint)
    root = Path(output_dir)
    baseline = {"validation": quality_metrics(original, bundle.validation, bundle.task, protocol.batch_size, protocol.device)}
    attempt = 0
    def execute(candidate):
        nonlocal attempt
        attempt += 1
        with torch.random.fork_rng():
            torch.manual_seed(protocol.seed)
            model, evidence = apply_candidate(original, bundle, protocol, candidate)
        measurement = measure_artifact(model, bundle.validation.x[:protocol.batch_size], root / f"attempt-{attempt}.pt",
            format=protocol.artifact_format, device=protocol.device, repeats=protocol.measurement_repeats, warmup=protocol.warmup)
        return {"candidate_id": candidate.candidate_id, "status": measurement["status"], "operations": evidence,
                "validation": quality_metrics(model, bundle.validation, bundle.task, protocol.batch_size, protocol.device), "measurement": measurement}
    report = compare_validation_timing(proposals, validator, execute,
                                       summarize=lambda records: archive_summary(records, baseline, protocol))
    report.update({"protocol": asdict(protocol), "proposals": [candidate.to_dict() for candidate in proposals],
                   "baseline_training": baseline_training, "checker": "explicit caller-provided existing checker; predictions independently executed in late mode", "test_used": False})
    return _save_report(output_dir, "validity_ablation.json", report)


def _merged_lora(model):
    from fedcore.models.network_modules.layers.lora import LoRALayer
    result = deepcopy(model).eval()
    for name, layer in list(result.named_modules()):
        if isinstance(layer, LoRALayer):
            layer.merge(safe_merge=True)
            result = _replace(result, name, deepcopy(layer.base_layer))
    return result


def run_training_cost_controls(bundle, protocol, output_dir, *, lora_rank=1, backend="fbgemm", baseline_checkpoint=None):
    """Equal Adam/lr/epochs/data: full FT, LoRA, student-only, KD, QAT/float.

    Teacher forward time is part of the KD total; adapter storage is separate
    from the dense merged model. CPU memory is sampled process RSS, explicitly
    distinguished from an exact tensor allocator peak.
    """
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    from fedcore.algorithm.distillation.distilator import BaseDistilator
    from fedcore.algorithm.quantization.quantizers import BaseQuantizer
    from fedcore.models.network_modules.layers.lora import LoRALayer
    from .measurement import measure_artifact
    class MeasuredQAT(BaseQuantizer):
        def _train_qat(self, input_data):
            self.updated_parameter_count = sum(p.numel() for p in self.quant_model.parameters() if p.requires_grad)
            return super()._train_qat(input_data)
    if bundle.task != "classification":
        raise ValueError("Native BaseLoRA and soft-logit KD require tensor classification in this suite")
    if protocol.finetune_epochs < 1:
        raise ValueError("Training-cost controls require positive finetune_epochs")
    if str(protocol.device) != "cpu":
        raise ValueError("This paired suite uses the checked CPU QAT profile")
    original, baseline_training = _prepare_baseline(bundle, protocol, output_dir, baseline_checkpoint)
    source_hash = _hash_model(original)
    records = []
    for name in ("full_finetune", "lora", "student_only", "distillation", "qat_float_control", "qat"):
        source = deepcopy(original)
        data = SimpleNamespace(model=source, train_dataloader=_loader(bundle.train, protocol.batch_size),
                               calibration_dataloader=_loader(bundle.calibration, protocol.batch_size))
        teacher_seconds, teacher_forwards = 0., 0
        handles = []
        memory = _TrainingRSS()
        with torch.random.fork_rng(), memory:
            torch.manual_seed(protocol.seed)
            start = time.perf_counter()
            if name == "lora":
                operation = BaseLoRA({"epochs": protocol.finetune_epochs, "lr": protocol.learning_rate, "device": "cpu",
                                      "lora_r": lora_rank, "lora_alpha": lora_rank, "lora_dropout": 0.})
                model = operation.fit(data)
                steps = len(operation.history)
            elif name == "distillation":
                # Same student architecture and initial weights as student_only.
                # The total includes the real frozen teacher forward each step.
                operation = BaseDistilator({"epochs": protocol.finetune_epochs, "lr": protocol.learning_rate, "device": "cpu",
                                            "student_model": deepcopy(source), "optimizer": "adam",
                                            "loss_weight": .5, "last_layer_loss_weight": .5})
                forward_started = []
                def before_teacher(_module, _inputs):
                    forward_started.append(time.perf_counter())
                def after_teacher(_module, _inputs, _outputs):
                    nonlocal teacher_seconds, teacher_forwards
                    teacher_seconds += time.perf_counter() - forward_started.pop()
                    teacher_forwards += 1
                    if any(p.requires_grad for p in _module.parameters()):
                        raise RuntimeError("Teacher was not frozen during execution")
                # Existing fit builds a deepcopy preserving hooks.
                handles = [source.register_forward_pre_hook(before_teacher), source.register_forward_hook(after_teacher)]
                # Student must have no teacher hooks.
                model = operation.fit(data)
                steps = len(operation.history)
                for handle in handles:
                    handle.remove()
                if _hash_model(operation.base_model) != source_hash:
                    raise RuntimeError("Distillation changed teacher weights")
            elif name == "qat":
                operation = MeasuredQAT({"quant_type": "qat", "backend": backend,
                                          "qat_params": {"epochs": protocol.finetune_epochs, "lr": protocol.learning_rate, "optimizer": "adam"}})
                model = operation.fit(data)
                steps = operation.quantization_result.training_steps
            else:
                model, history = _train(source, bundle.train, protocol, criterion=_loss(bundle.task))
                steps = len(history)
            training_seconds = time.perf_counter() - start
        if _hash_model(original) != source_hash:
            raise RuntimeError("A training control mutated the shared baseline")
        unmerged_quality = _quality(model, bundle.validation, bundle.task, protocol.batch_size)
        adapter = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                   if any(part in key.split('.') for part in LoRALayer.adapter_layer_names)} if name == "lora" else {}
        adapter_file = None
        if adapter:
            from fedcore.external_runtime.security import safe_save
            adapter_file = Path(output_dir) / "adapter.fcb"
            adapter_file.parent.mkdir(parents=True, exist_ok=True)
            safe_save(adapter, adapter_file)
        updated = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inference_model = _merged_lora(model) if name == "lora" else model
        if name == "lora":
            merge_error = model_error(model, inference_model, bundle.validation, protocol.batch_size)
        else:
            merge_error = None
        records.append({"variant": name, "starting_sha256": source_hash, "optimizer": "Adam", "learning_rate": protocol.learning_rate,
                        "epochs": protocol.finetune_epochs, "steps": steps, "training_seconds_including_setup": training_seconds,
                        "teacher_forward_seconds": teacher_seconds, "teacher_forwards": teacher_forwards,
                        "updated_parameters": updated if name != "qat" else operation.updated_parameter_count,
                        "updated_parameters_note": "QAT updates float prepared weights before conversion" if name == "qat" else "requires_grad parameters",
                        "dense_baseline_parameters": sum(p.numel() for p in original.parameters()),
                        "adapter_tensor_bytes": sum(v.numel() * v.element_size() for v in adapter.values()),
                        "adapter_file_bytes": adapter_file.stat().st_size if adapter_file else None,
                        "unmerged_quality": unmerged_quality, "merged_quality": _quality(inference_model, bundle.validation, bundle.task, protocol.batch_size),
                        "merge_error": merge_error,
                        "training_memory": memory.report(),
                        "measurement": measure_artifact(inference_model, bundle.validation.x[:protocol.batch_size], Path(output_dir) / f"{name}.pt",
                                     format=protocol.artifact_format, device="cpu", repeats=protocol.measurement_repeats, warmup=protocol.warmup)})
    return _save_report(output_dir, "training_cost.json", {"experiment": "training_cost_controls", "protocol": asdict(protocol),
                        "source_sha256": source_hash, "baseline_training": baseline_training, "records": records, "claim_status": "not_established", "test_used": False})
