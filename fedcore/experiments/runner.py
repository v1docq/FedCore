"""Effect shell for training, compression, provenance and the frozen test gate."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fedcore.tools.atomic_json import write_atomic_json

from .measurement import file_hash, load_artifact, _call_loaded, measure_artifact, tensor_state_bytes
from .protocol import (ROLES, CandidateSpec, ExperimentBundle, ExperimentProtocol,
                       ProtocolError, TensorSplit, canonical_json, json_value,
                       role_content_identity, stable_hash, tensor_hash, validate_roles)


class UnsupportedOperation(RuntimeError):
    pass


def _write_json(path, value):
    write_atomic_json(path, value)


def _redact(value):
    if isinstance(value, dict):
        return {key: "[redacted]" if any(part in key.lower() for part in
                    ("secret", "password", "credential", "api_key")) or key.lower() in
                    ("token", "access_token", "refresh_token", "auth_token", "authorization") or
                    key.lower().endswith('token') else _redact(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def environment_manifest():
    """Allowlisted versions, never environment variables or credentials."""
    versions = {}
    for name in ("torch", "numpy", "scikit-learn", "fedot", "thegolem", "tdecomp", "onnx", "onnxruntime", "optuna"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    # Include uncommitted source edits without exposing their contents.
    source = {str(path.relative_to(root)): file_hash(path)
              for path in sorted((root / "fedcore").rglob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "torch_cuda_build": torch.version.cuda,
            "cpu_processor": platform.processor(), "machine": platform.machine(),
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "git_revision": revision, "source_sha256": stable_hash(source),
            "source_files": source}


def model_state_hash(model):
    def describe(value):
        if isinstance(value, torch.Tensor):
            if value.is_quantized:
                return {"tensor": tensor_hash(value.int_repr()), "scale": value.q_scale(),
                        "zero_point": value.q_zero_point()} if value.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric) else {
                            "tensor": tensor_hash(value.int_repr()),
                            "scales": value.q_per_channel_scales().tolist(),
                            "zero_points": value.q_per_channel_zero_points().tolist()}
            return {"tensor": tensor_hash(value)}
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if isinstance(value, torch.dtype):
            return str(value)
        return json_value(value)
    return stable_hash({"architecture": str(model), "state": describe(model.state_dict())})


@contextmanager
def _seeded(seed, device):
    state = random.getstate()
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(state)


def task_loss(output, target, task):
    if task == "classification":
        if output.ndim != 2 or target.ndim != 1 or len(output) != len(target):
            raise ProtocolError("Classification requires [N,C] scores and [N] labels")
        return nn.functional.cross_entropy(output, target.long())
    if task == "language_model":
        if output.ndim != 3 or target.shape != output.shape[:2] or output.shape[1] < 2:
            raise ProtocolError("Causal LM requires [B,T,V] logits and [B,T] labels")
        labels = target[:, 1:].reshape(-1).long()
        if not (labels != -100).any():
            raise ProtocolError("No nonpadding causal tokens")
        return nn.functional.cross_entropy(output[:, :-1].reshape(-1, output.shape[-1]), labels, ignore_index=-100)
    if output.shape != target.shape:
        raise ProtocolError(f"Regression output {tuple(output.shape)} != target {tuple(target.shape)}")
    return nn.functional.mse_loss(output, target.to(output.dtype))


def _quality(output, split, task):
    if not torch.isfinite(output).all():
        raise ProtocolError("Nonfinite predictions cannot enter an archive")
    loss = task_loss(output, split.y, task)
    if not torch.isfinite(loss):
        raise ProtocolError("Nonfinite quality metric")
    if task == "classification":
        value = float((output.argmax(-1) == split.y).double().mean())
        predictions = output.argmax(-1)
        f1 = []
        for label in range(output.shape[-1]):
            tp = int(((predictions == label) & (split.y == label)).sum())
            fp = int(((predictions == label) & (split.y != label)).sum())
            fn = int(((predictions != label) & (split.y == label)).sum())
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return {"metric": "accuracy", "value": value, "loss": 1 - value,
                "macro_f1": sum(f1) / len(f1), "cross_entropy": float(loss),
                "samples": len(split.ids), "direction": "maximize"}
    if task == "language_model":
        tokens = int((split.y[:, 1:] != -100).sum())
        nll = float(loss)
        return {"metric": "causal_token_nll", "value": nll, "loss": nll,
                "perplexity": math.exp(nll) if nll < 700 else None,
                "token_count": tokens, "samples": len(split.ids), "direction": "minimize"}
    return {"metric": "mse", "value": float(loss), "loss": float(loss),
            "samples": len(split.ids), "direction": "minimize"}


def _predictions(model, split, batch_size, device):
    model = model.to(device).eval()
    with torch.inference_mode():
        outputs = [model(split.x[start:start + batch_size].to(device)).detach().cpu()
                   for start in range(0, len(split.x), batch_size)]
    return torch.cat(outputs)


def quality_metrics(model, split, task, batch_size=16, device="cpu"):
    split.verify_integrity()
    return _quality(_predictions(model, split, batch_size, device), split, task)


def train_model(model, split, task, *, epochs, batch_size, learning_rate, device="cpu", seed=0,
                optimizer_validator=None, teacher=None, feature_path=None):
    """Fixed-budget Adam training; validation and test cannot stop it."""
    started = time.perf_counter()
    if type(epochs) is not int or epochs < 0:
        raise ProtocolError("Training epochs must be nonnegative")
    model.to(device)
    loader = DataLoader(TensorDataset(split.x, split.y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if epochs and not trainable:
        raise ProtocolError('Training requires at least one live trainable Parameter')
    optimizer = torch.optim.Adam(trainable, lr=learning_rate) if epochs else None
    if optimizer is not None and optimizer_validator is not None:
        optimizer_validator(optimizer)
    if (teacher is None) != (feature_path is None):
        raise ProtocolError('GFM requires both a fixed teacher and an explicit feature path')
    if teacher is not None:
        if teacher is model or not isinstance(feature_path, str):
            raise ProtocolError('GFM teacher must be independent with an explicit late feature point')
        teacher = copy.deepcopy(teacher).to(device).eval().requires_grad_(False)
    history = []
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            if teacher is None:
                loss = task_loss(model(x.to(device)), y.to(device), task)
            else:
                student_features, teacher_features = [], []
                def student_hook(_layer, _args, output):
                    student_features.append(output)
                def teacher_hook(_layer, _args, output):
                    teacher_features.append(output.detach())
                handles = [model.get_submodule(feature_path).register_forward_hook(student_hook),
                           teacher.get_submodule(feature_path).register_forward_hook(teacher_hook)]
                try:
                    model(x.to(device))
                    with torch.no_grad():
                        teacher(x.to(device))
                    if (len(student_features)!=1 or len(teacher_features)!=1
                            or not isinstance(student_features[0],torch.Tensor)
                            or student_features[0].shape!=teacher_features[0].shape):
                        raise ProtocolError('GFM requires one compatible tensor feature per model')
                    loss = nn.functional.mse_loss(student_features[0],teacher_features[0])
                finally:
                    for handle in handles:
                        handle.remove()
            if not torch.isfinite(loss):
                raise ProtocolError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
    model.eval()
    evidence = {"training_steps": len(history), "epochs": epochs, "train_loss": history,
            "optimizer": "Adam", "learning_rate": learning_rate, "training_role": "train",
            "training_seconds": time.perf_counter()-started,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "model_tensor_bytes": tensor_state_bytes(model)}
    if teacher is not None:
        evidence.update(objective='GFM_late_feature_mse',teacher_checkpoint=model_state_hash(teacher),
                        teacher_feature_map={'teacher':feature_path,'student':feature_path})
    return evidence


def _compression_input(model, bundle, protocol):
    from fedcore.data.data import CompressionInputData
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    def loader(split, shuffle=False):
        return DataLoader(TensorDataset(split.x, split.y), batch_size=protocol.batch_size,
                          shuffle=shuffle, generator=torch.Generator().manual_seed(protocol.seed))
    problem = "classification" if bundle.task == "classification" else "regression"
    data = CompressionInputData(model=model, task=Task(getattr(TaskTypesEnum, problem)),
                                train_dataloader=loader(bundle.train, True),
                                val_dataloader=loader(bundle.validation),
                                num_classes=int(bundle.train.y.max()) + 1 if problem == "classification" else None,
                                input_dim=bundle.train.x.shape[-1])
    data.calibration_dataloader = loader(bundle.calibration)
    return data


def _fedcore_operation(model, bundle, protocol, parameters):
    if bundle.task == "language_model":
        raise UnsupportedOperation("The current FedCore generic facade profile does not accept causal LM tensors")
    from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, ComputeConfigTemplate,
                                        DeviceConfigTemplate, DistributedConfigTemplate, FedotConfigTemplate,
                                        LearningConfigTemplate, LowRankTemplate, PruningTemplate,
                                        QuantizationTemplate, TrainingTemplate)
    from fedcore.api.config_factory import ConfigFactory
    from fedcore.api.main import FedCore
    from distributed import Client, LocalCluster
    templates = {"training": TrainingTemplate, "low_rank": LowRankTemplate,
                 "pruning": PruningTemplate, "quantization": QuantizationTemplate}
    name = parameters.get("operation", "training")
    if name not in templates or set(parameters) - {"operation", "operation_parameters"}:
        raise UnsupportedOperation("Unknown FedCore operation/schema")
    operation = templates[name](**json_value(parameters.get("operation_parameters", {"epochs": protocol.finetune_epochs})))
    problem = "classification" if bundle.task == "classification" else "regression"
    template = APIConfigTemplate(
        device_config=DeviceConfigTemplate(device="cuda" if protocol.device.startswith("cuda") else "cpu"),
        automl_config=AutoMLConfigTemplate(fedot_config=FedotConfigTemplate(problem=problem, initial_assumption=model,
            metric=[f"MulticlassAccuracy__{int(bundle.train.y.max()) + 1}" if problem == "classification" else "MeanSquaredError"], n_jobs=1)),
        learning_config=LearningConfigTemplate(learning_strategy="checkpoint", criterion="cross_entropy" if problem == "classification" else "mse", peft_strategy_params=operation),
        compute_config=ComputeConfigTemplate(distributed=DistributedConfigTemplate(threads_per_worker=1)))
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                           protocol="inproc", dashboard_address=None, memory_limit=0)
    client = None
    try:
        client = Client(cluster)
        api = FedCore(ConfigFactory.from_template(template)(), dask_client=client, dask_cluster=cluster)
        api.fit_no_evo(_compression_input(model, bundle, protocol))
        if api.compressed_model is None:
            raise RuntimeError("FedCore produced no compressed model")
        return copy.deepcopy(api.compressed_model), {"implementation": "ConfigFactory + FedCore.fit_no_evo",
                                                    "operation": name, "parameters": json_value(parameters)}
    finally:
        if client is not None:
            client.close()
        cluster.close()


def _materialize_decomposed(model):
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
    aliases = {}
    def visit(layer):
        if id(layer) in aliases:
            return aliases[id(layer)]
        if isinstance(layer, IDecomposed):
            aliases[id(layer)] = to_standard_module(layer)
            return aliases[id(layer)]
        aliases[id(layer)] = layer
        for name, child in tuple(layer._modules.items()):
            if child is not None:
                layer._modules[name] = visit(child)
        return layer
    return visit(model)


def apply_candidate(model, bundle, protocol, candidate):
    """Execute a declared operation on an independent copy, never a no-op stub."""
    from fedcore.algorithm.low_rank.topology import inspect_topology
    if inspect_topology(model).storage_aliases:
        raise UnsupportedOperation("Candidate copying does not support shared storage views")
    model = copy.deepcopy(model)
    method, parameters = candidate.method, json_value(candidate.parameters)
    if method == "chain":
        steps = []
        for step in candidate.chain:
            model, evidence = apply_candidate(model, bundle, protocol, step)
            steps.append(evidence)
        return model, {"implementation": "ordered compression chain", "steps": steps}
    if method == "baseline":
        if parameters:
            raise ProtocolError("Baseline takes no compression parameters")
        return model, {"implementation": "independent trained baseline clone", "training_steps": 0}
    if method == "train":
        if set(parameters) - {"epochs"}:
            raise ProtocolError("Unknown training parameters")
        evidence = train_model(model, bundle.train, bundle.task, epochs=parameters.get("epochs", protocol.finetune_epochs),
                               batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
                               device=protocol.device, seed=protocol.seed)
        return model, {"implementation": "fixed-budget torch Adam", **evidence}
    if method == "pruning":
        if set(parameters) - {"amount", "finetune_epochs"}:
            raise ProtocolError("Unknown magnitude pruning parameters")
        amount = parameters.get("amount", 0.3)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not 0 < amount < 1:
            raise ProtocolError("Pruning amount must be a fraction in (0,1)")
        from torch.nn.utils import prune
        model = _materialize_decomposed(model)
        layers = [(layer, "weight") for layer in model.modules() if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d))]
        if not layers:
            raise UnsupportedOperation("No supported float weights for magnitude pruning")
        prune.global_unstructured(layers, pruning_method=prune.L1Unstructured, amount=amount)
        zero_count = sum(int((getattr(layer, "weight") == 0).sum()) for layer, _ in layers)
        evidence = train_model(model, bundle.train, bundle.task,
                               epochs=parameters.get("finetune_epochs", protocol.finetune_epochs),
                               batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
                               device=protocol.device, seed=protocol.seed)
        for layer, name in layers:
            prune.remove(layer, name)
        return model, {"implementation": "torch global L1 unstructured pruning", "amount": amount,
                       "pruned_weight_count": zero_count, "representation": "dense masked weights; no claimed structural size saving", **evidence}
    if method == "structural_pruning":
        if set(parameters) - {"pruning_ratio", "finetune_epochs", "importance"}:
            raise ProtocolError("Unknown structural pruning parameters")
        ratio = parameters.get("pruning_ratio", 0.3)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
            raise ProtocolError("pruning_ratio must lie in (0,1)")
        if bundle.task == "language_model":
            raise UnsupportedOperation("Structural pruning of causal recurrent LM is not validated by this profile")
        if parameters.get("importance", "magnitude") not in ("magnitude", "lamp", "random"):
            raise UnsupportedOperation("This protocol supports only zero-shot importance without validation gradient reuse")
        from fedcore.algorithm.pruning.pruners import BasePruner
        model = _materialize_decomposed(model).to(protocol.device).eval()
        before = sum(parameter.numel() for parameter in model.parameters())
        data = _compression_input(model, bundle, protocol)
        with torch.no_grad():
            output_shape = model(bundle.train.x[:1].to(protocol.device)).shape
        data.num_classes = output_shape[-1]
        operation = BasePruner({"pruning_ratio": ratio, "importance": parameters.get("importance", "magnitude"),
                                "device": protocol.device, "pruning_iterations": 1})
        operation._init_model_before_model_after(data)
        pruner = operation._init_pruner_with_model_after(data)
        if pruner is None:
            raise UnsupportedOperation("No structural pruning agent was created")
        pruner.step()
        model = operation.model_after
        after = sum(parameter.numel() for parameter in model.parameters())
        if after >= before:
            raise UnsupportedOperation("No channels were structurally removed under the declared constraints")
        evidence = train_model(model, bundle.train, bundle.task,
                               epochs=parameters.get("finetune_epochs", protocol.finetune_epochs),
                               batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
                               device=protocol.device, seed=protocol.seed)
        return model, {"implementation": "FedCore BasePruner dependency graph + Torch-Pruning step", "pruning_ratio": ratio,
                       "parameters_before": before, "parameters_after": after,
                       "representation": "structurally removed channels; decomposed inputs materialized first", **evidence}
    p2_methods = ('asvd','fwsvd','afm','bolaco','flar_svd','drone','svdllm_v1',
                  'svdllm_v2','svdllm_v5','mixed_rank','basis_sharing','groupreduce','eora')
    if method in p2_methods:
        from fedcore.algorithm.low_rank.method_specs import parse_method, SVDLLMV5, Bolaco, FWSVD
        from fedcore.algorithm.low_rank.method_execution import transform_method
        allowed = {'method_options','rank','rank_ratio','parameter_fraction','ranks',
            'target_paths','parameter_budget','tensor_byte_budget','max_workspace_bytes','max_peak_bytes',
            'finetune_epochs','gfm_feature_path','group_ids','base_candidate'}
        if set(parameters)-allowed:
            raise ProtocolError('Unknown P2 method parameters')
        if protocol.device!='cpu':
            raise UnsupportedOperation('P2 method profiles require CPU')
        spec = parse_method(method, parameters.get('method_options'))
        options = {key:value for key,value in parameters.items() if key not in
                   {'method_options','finetune_epochs','gfm_feature_path','group_ids','base_candidate'}}
        base_evidence = None
        if method=='eora':
            base_payload=parameters.get('base_candidate')
            if not isinstance(base_payload,dict) or base_payload.get('method')!='pruning':
                raise ProtocolError('First EoRA PETRA profile requires an explicit pruning base_candidate')
            base_candidate=CandidateSpec('pruning',base_payload.get('parameters',{}))
            options['base_model'],base_evidence=apply_candidate(model,bundle,protocol,base_candidate)
        elif 'base_candidate' in parameters:
            raise ProtocolError('base_candidate is only valid for EoRA')
        if type(spec) is FWSVD:
            options['labels']=bundle.calibration.y
        if type(spec) is Bolaco:
            # Explicit group IDs, never silently inferred from task labels.
            groups = parameters.get('group_ids')
            if groups is None:
                raise ProtocolError('Bolaco requires explicit calibration group_ids')
            options['group_labels']=torch.tensor(groups,dtype=torch.long)
        if type(spec) is SVDLLMV5:
            if parameters.get('finetune_epochs',0)!=0 or 'gfm_feature_path' in parameters:
                raise ProtocolError('v5 stage budgets are distinct from ordinary fine-tuning/GFM')
            def recover(current, side, epochs, stage):
                return train_model(current,bundle.train,bundle.task,epochs=epochs,
                    batch_size=protocol.batch_size,learning_rate=protocol.learning_rate,
                    device=protocol.device,seed=protocol.seed,
                    optimizer_validator=stage.validate_optimizer)
            options['recovery_executor']=recover
        transformed = transform_method(model,bundle.calibration.x,spec,
                                       batch_size=protocol.batch_size,**options)
        result=transformed.model
        training={'training_steps':0,'training_role':'train'}
        if type(spec) is not SVDLLMV5:
            training=train_model(result,bundle.train,bundle.task,
                epochs=parameters.get('finetune_epochs',protocol.finetune_epochs),
                batch_size=protocol.batch_size,learning_rate=protocol.learning_rate,
                device=protocol.device,seed=protocol.seed,
                teacher=model if 'gfm_feature_path' in parameters else None,
                feature_path=parameters.get('gfm_feature_path'))
        return result,{'implementation':'FedCore shared named-method interpreter',
                       'method_transform':transformed.evidence,'base_candidate':base_evidence,**training}
    if method == "weighted_svd":
        allowed = {"rank", "rank_ratio", "parameter_fraction", "target_paths", "ridge", "rcond",
                   "nullspace_policy", "max_workspace_bytes", "max_peak_bytes", "finetune_epochs"}
        if set(parameters) - allowed:
            raise ProtocolError("Unknown weighted SVD parameters")
        if protocol.device != "cpu":
            raise UnsupportedOperation("The first weighted SVD profile requires CPU")
        from fedcore.algorithm.low_rank.execution import transform_weighted
        from fedcore.algorithm.low_rank.plans import MetricPolicy
        options = {key: value for key, value in parameters.items()
                   if key not in {"ridge", "rcond", "nullspace_policy", "finetune_epochs"}}
        result = transform_weighted(model, bundle.calibration.x,
            policy=MetricPolicy(parameters.get("ridge", 0.0), parameters.get("rcond"),
                                parameters.get("nullspace_policy", "support_only")),
            batch_size=protocol.batch_size, **options)
        model = result.model
        training = train_model(model, bundle.train, bundle.task,
            epochs=parameters.get("finetune_epochs", protocol.finetune_epochs),
            batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
            device=protocol.device, seed=protocol.seed)
        return model, {"implementation": "FedCore shared weighted transform interpreter",
                       "weighted_transform": result.evidence, **training}
    if method == "svd":
        if set(parameters) - {"threshold", "strategy", "decomposer", "finetune_epochs", "rank", "rank_ratio"}:
            raise ProtocolError("Unknown SVD parameters")
        if sum(key in parameters for key in ("rank", "rank_ratio", "threshold")) > 1:
            raise ProtocolError("Choose exactly one explicit rank, rank_ratio or threshold policy")
        if "rank" in parameters and (type(parameters["rank"]) is not int or parameters["rank"] < 1):
            raise ProtocolError("Explicit SVD rank must be positive integer")
        if "rank_ratio" in parameters and (isinstance(parameters["rank_ratio"], bool) or not isinstance(parameters["rank_ratio"], (int, float)) or not 0 < parameters["rank_ratio"] <= 1):
            raise ProtocolError("SVD rank_ratio must lie in (0,1]")
        from fedcore.algorithm.low_rank.svd_tools import decompose_module
        from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning_in_place
        from fedcore.models.network_impl.decomposed_layers import IDecomposed
        model = decompose_module(model, decomposer=parameters.get("decomposer", "svd"), compose_mode="two_layers")
        layers = [layer for layer in model.modules() if isinstance(layer, IDecomposed)]
        if not layers:
            raise UnsupportedOperation("No supported layer was decomposed")
        ranks = []
        for layer in layers:
            if "rank" in parameters or "rank_ratio" in parameters:
                u, s, vh = layer.canonicalize()
                rank = parameters.get("rank", max(1, int(s.shape[-1] * parameters.get("rank_ratio", 1))))
                if rank > s.shape[-1]:
                    raise ProtocolError("Explicit rank exceeds a layer's actual matrix rank bound")
                layer.set_U_S_Vh(u[..., :rank], s[..., :rank], vh[..., :rank, :])
                layer.rank_pruning_info = {"method": "operator_svd", "strategy": "explicit_rank" if "rank" in parameters else "rank_ratio",
                                           "rank": rank, "rank_bound": s.shape[-1]}
            else:
                rank_threshold_pruning_in_place(layer, threshold=parameters.get("threshold", 0.8),
                                                 strategy=parameters.get("strategy", "explained_variance"), round_to_times=1)
            ranks.append(dict(layer.rank_pruning_info))
        evidence = train_model(model, bundle.train, bundle.task,
                               epochs=parameters.get("finetune_epochs", protocol.finetune_epochs),
                               batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
                               device=protocol.device, seed=protocol.seed)
        for layer in layers:
            layer.compose_weight_for_inference()
        return model, {"implementation": "FedCore decompose_module + rank_threshold_pruning_in_place", "ranks": ranks, **evidence}
    if method in ("ptq", "qat"):
        if protocol.device != "cpu":
            raise UnsupportedOperation("Quantization runtime is explicitly CPU; use a separate CPU protocol")
        if set(parameters) - {"mode", "backend", "epochs"}:
            raise ProtocolError("Unknown quantization parameters")
        from fedcore.algorithm.quantization.quantizers import BaseQuantizer, QuantizationError
        mode = "qat" if method == "qat" else parameters.get("mode", "static")
        if method == "ptq" and mode not in ("static", "dynamic"):
            raise ProtocolError("PTQ mode must be static or dynamic")
        epochs = parameters.get("epochs", protocol.finetune_epochs)
        criterion = lambda scores, targets: task_loss(scores, targets, bundle.task)
        operation = BaseQuantizer({"quant_type": mode, "backend": parameters.get("backend", "fbgemm"),
                                   "qat_params": {"epochs": epochs, "optimizer": "adam", "lr": protocol.learning_rate,
                                                  "criterion": criterion}})
        try:
            operation.fit(_compression_input(model, bundle, protocol))
        except QuantizationError as error:
            if error.result.status == "not_applicable":
                raise UnsupportedOperation(str(error)) from error
            raise
        result = operation.quantization_result
        return operation.model_after, {"implementation": "FedCore BaseQuantizer", "mode": mode,
                                        "backend": operation.backend, "training_steps": result.training_steps,
                                        "calibration_role": "calibration", "training_role": "train"}
    if method == "fedcore":
        return _fedcore_operation(model, bundle, protocol, parameters)
    raise UnsupportedOperation(f"No implementation for operation {method!r}")


class ExperimentRunner:
    """Each instance owns one run; final test access starts after selection freezes."""
    def __init__(self, bundle, protocol, output_dir, *, cache_dir=None, use_cache=False, external_preparation_seconds=0.0,
                 baseline_checkpoint=None):
        preparation_start = time.perf_counter()
        if isinstance(external_preparation_seconds, bool) or not isinstance(external_preparation_seconds, (int, float)) or not math.isfinite(external_preparation_seconds) or external_preparation_seconds < 0:
            raise ProtocolError("External data preparation cost must be finite nonnegative seconds")
        if not isinstance(bundle, ExperimentBundle) or not isinstance(protocol, ExperimentProtocol):
            raise ProtocolError("Use ExperimentBundle and ExperimentProtocol")
        validate_roles(bundle)
        if protocol.device.startswith("cuda") and not torch.cuda.is_available():
            raise ProtocolError("Requested CUDA device is unavailable")
        self.bundle, self.protocol = bundle, protocol
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "manifest.json").exists():
            raise ProtocolError("A run directory already contains a manifest; choose a new directory")
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else self.output_dir / "cache"
        self.use_cache = use_cache
        self.baseline_checkpoint = baseline_checkpoint
        from fedcore.algorithm.low_rank.topology import inspect_topology
        if inspect_topology(bundle.original_model).storage_aliases:
            raise UnsupportedOperation("Experiment copying does not support shared storage views")
        self.initial_model = copy.deepcopy(bundle.original_model).cpu()
        self.environment = environment_manifest()
        self.manifest = {"version": 1, "status": "created", "protocol": protocol.to_dict(),
                         "data": _redact(bundle.manifest()), "environment": self.environment,
                         "initial_model_sha256": model_state_hash(self.initial_model),
                         "candidates": [], "test_gate": {"status": "closed"},
                         "external_preparation_seconds": external_preparation_seconds,
                         "external_preparation_status": "charged" if external_preparation_seconds else "not measured; dataset loading and preprocessing outside the runner are excluded",
                         "empirical_claim": "pilot only; superiority is not established"}
        self.started = None
        self.baseline_model = None
        self.baseline_record = None
        self.preparation_seconds = time.perf_counter() - preparation_start + external_preparation_seconds

    def _save(self):
        self.manifest["wall_seconds"] = time.perf_counter() - self.started if self.started is not None else 0
        _write_json(self.output_dir / "manifest.json", self.manifest)

    def _event(self, event):
        event = {"elapsed_seconds": time.perf_counter() - self.started, **event}
        with (self.output_dir / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(_redact(event)) + "\n")

    def _prepare(self):
        if self.started is not None:
            raise ProtocolError("An ExperimentRunner cannot be reused for another run")
        self.started = time.perf_counter() - self.preparation_seconds
        self.manifest["preparation_seconds"] = self.preparation_seconds
        self.manifest["status"] = "running"
        self._save()
        tensors, backing_files = {}, {}
        for role in ROLES:
            split = getattr(self.bundle, role)
            if split._backing:
                files = {}
                for axis in ("x", "y"):
                    name = f"{role}-{axis}.npy"
                    target = self.output_dir / name
                    shutil.copyfile(split._backing[axis]["path"], target)
                    files[axis] = {"path": str(target.resolve()), "sha256": file_hash(target)}
                    backing_files[name] = files[axis]
                tensors[role] = {"files": files}
            else:
                tensors[role] = {"x": split.x, "y": split.y}
        torch.save(tensors, self.output_dir / "data.pt")
        torch.save(self.initial_model.state_dict(), self.output_dir / "initial_state.pt")
        self.manifest["replay_files"] = {name: {"path": str((self.output_dir / name).resolve()),
                                                "sha256": file_hash(self.output_dir / name)}
                                         for name in ("data.pt", "initial_state.pt")}
        self.manifest["replay_files"].update(backing_files)
        start = time.perf_counter()
        self.manifest["baseline_training"] = {"status": "running"}
        try:
            from .baseline import prepare_baseline
            self.baseline_model, training = prepare_baseline(self.bundle, self.protocol, self.output_dir,
                                                           checkpoint=self.baseline_checkpoint, environment=self.environment)
        except BaseException as error:
            self.manifest["baseline_training"] = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                   "wall_seconds": time.perf_counter() - start,
                                                   "reason": type(error).__name__}
            self._save()
            raise
        self.manifest["baseline_training"] = {**training, "training_status": training["status"], "status": "succeeded"}
        if training["reused"]:
            # Charge the same common training work to each paired method; retain
            # actual verification time separately instead of implying retraining.
            self.started -= training["wall_seconds"]
        self.manifest["baseline_cost_accounting"] = {"mode": "common_baseline_preparation_charged_per_run",
                    "charged_common_baseline_seconds": training["wall_seconds"],
                    "charged_training_seconds": training["training_seconds"],
                    "actual_training_or_verification_seconds": training["actual_work_seconds"]}
        self.manifest["replay_files"].update({name: {"path": str((self.output_dir / name).resolve()),
                                                   "sha256": file_hash(self.output_dir / name)}
                                             for name in ("baseline_state.pt", "baseline.json")})
        self.baseline_record = self._evaluate(CandidateSpec("baseline"))
        if self.baseline_record["status"] != "succeeded":
            raise RuntimeError("Baseline export/validation failed; no valid paired experiment")
        threshold = self.protocol.minimum_baseline_quality
        quality = self.baseline_record["validation"]
        if threshold is not None and ((quality["direction"] == "maximize" and quality["value"] < threshold) or
                (quality["direction"] == "minimize" and quality["value"] > threshold)):
            raise ProtocolError("Baseline fails the prospectively frozen quality threshold")

    def _cache_key(self, candidate):
        data = role_content_identity({role: getattr(self.bundle, role).manifest() for role in ROLES if role != "test"})
        return stable_hash({"candidate": candidate.to_dict(), "baseline": model_state_hash(self.baseline_model),
                            "data": data, "protocol": self.protocol.to_dict(), "environment": self.environment})

    def _evaluate(self, candidate):
        if self.manifest["test_gate"]["status"] != "closed":
            raise ProtocolError("Cannot evaluate a new candidate after the test gate opens")
        validate_roles(self.bundle)
        start = time.perf_counter()
        record = {"candidate_id": candidate.candidate_id, "configuration": _redact(candidate.to_dict()),
                  "status": "running", "cache_hit": False,
                  "source_model_sha256": model_state_hash(self.baseline_model),
                  "started_seconds": start - self.started, "stages_seconds": {}}
        self.manifest["candidates"].append(record)
        self._event({"event": "candidate_started", "candidate_id": candidate.candidate_id})
        self._save()
        path = self.output_dir / "candidates" / f"{len(self.manifest['candidates']):03d}-{candidate.candidate_id}"
        path.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(candidate)
        record["cache_key"] = key
        try:
            cache_path = self.cache_dir / f"{key}.json"
            if self.use_cache and cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") != key or cached.get("status") != "succeeded":
                    raise ProtocolError("Invalid cache entry")
                load_artifact(cached["measurement"]["artifact"])
                record.update({name: cached[name] for name in ("measurement", "validation", "operation", "result_model_sha256", "validation_predictions")})
                if file_hash(record["validation_predictions"]["path"]) != record["validation_predictions"]["sha256"]:
                    raise ProtocolError("Cached prediction artifact hash mismatch")
                record.update(status="succeeded", cache_hit=True, cached_work_seconds=cached["wall_seconds"])
            else:
                stage = time.perf_counter()
                with _seeded(self.protocol.seed, self.protocol.device):
                    model, evidence = apply_candidate(self.baseline_model, self.bundle, self.protocol, candidate)
                record["stages_seconds"]["operation"] = time.perf_counter() - stage
                record["operation"] = evidence
                record["result_model_sha256"] = model_state_hash(model)
                stage = time.perf_counter()
                measurement = measure_artifact(model, self.bundle.validation.x[:self.protocol.batch_size], path / "model",
                                               format=self.protocol.artifact_format, device=self.protocol.device,
                                               repeats=self.protocol.measurement_repeats, warmup=self.protocol.warmup,
                                               threads=self.protocol.threads)
                record["stages_seconds"]["export_reload_measure"] = time.perf_counter() - stage
                record["measurement"] = measurement
                if measurement["status"] != "succeeded":
                    record["status"] = measurement["status"]
                    record["reason"] = measurement.get("reason", "Artifact measurement failed")
                else:
                    stage = time.perf_counter()
                    quality, prediction_path = self._artifact_quality(measurement["artifact"], self.bundle.validation,
                                                                      path / "validation_predictions.pt")
                    record["validation"] = quality
                    record["validation_predictions"] = prediction_path
                    record["stages_seconds"]["validation"] = time.perf_counter() - stage
                    record["status"] = "succeeded"
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["reason"] = "Interrupted during candidate execution"
            raise
        except (ImportError, UnsupportedOperation) as error:
            record.update(status="unsupported", reason=str(error), error_type=type(error).__name__)
        except Exception as error:
            record.update(status="failed", reason=str(error), error_type=type(error).__name__)
        finally:
            record["wall_seconds"] = time.perf_counter() - start
            record["completed_seconds"] = time.perf_counter() - self.started
            if record["source_model_sha256"] != model_state_hash(self.baseline_model):
                record.update(status="failed", reason="Candidate mutated shared baseline")
            if record["status"] == "succeeded" and not record["cache_hit"]:
                _write_json(self.cache_dir / f"{key}.json", record)
            self._event({"event": "candidate_finished", "candidate_id": candidate.candidate_id,
                         "status": record["status"], "wall_seconds": record["wall_seconds"],
                         "reason": record.get("reason")})
            self._save()
        return record

    def _artifact_quality(self, artifact, split, output_path):
        loaded = load_artifact(artifact)
        with torch.inference_mode():
            values = [_call_loaded(loaded, artifact, split.x[start:start + self.protocol.batch_size].to(artifact["device"])).detach().cpu()
                      for start in range(0, len(split.x), self.protocol.batch_size)]
        predictions = torch.cat(values)
        quality = _quality(predictions, split, self.bundle.task)
        torch.save({"ids": list(split.ids), "targets": split.y, "predictions": predictions}, output_path)
        return quality, {"path": str(Path(output_path).resolve()), "sha256": file_hash(output_path)}

    def _finalize(self):
        from .search import archive_summary, cost_selection_comparison
        summary = archive_summary(self.manifest["candidates"], self.baseline_record, self.protocol)
        self.manifest["selection"] = summary
        selected = tuple(summary["archive_ids"])
        self.manifest["test_gate"] = {"status": "frozen", "selected_ids": list(selected),
                                     "selection_sha256": stable_hash(summary), "selection_role": "validation"}
        self._event({"event": "selection_frozen", **self.manifest["test_gate"]})
        self._save()
        self.manifest["test_gate"]["status"] = "open"
        for record in self.manifest["candidates"]:
            if record["status"] == "succeeded" and (record["candidate_id"] in selected or record["configuration"]["method"] == "baseline"):
                start = time.perf_counter()
                artifact = record["measurement"]["artifact"]
                quality, predictions = self._artifact_quality(artifact, self.bundle.test,
                                                              self.output_dir / f"test-{record['candidate_id']}.pt")
                record["test"] = quality
                record["test_predictions"] = predictions
                record["test_wall_seconds"] = time.perf_counter() - start
        self.manifest["test_gate"]["status"] = "completed"
        self.manifest["cost_selection_comparison"] = cost_selection_comparison(self.manifest["candidates"], self.baseline_record, self.protocol)
        self.manifest["status"] = "succeeded"
        self._save()
        from .reporting import write_reports
        write_reports(self.manifest, self.output_dir)
        return self.manifest

    def run(self, candidates):
        candidates = tuple(candidates)
        if any(not isinstance(candidate, CandidateSpec) for candidate in candidates):
            raise ProtocolError("CandidateSpec records are required")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ProtocolError("Duplicate candidate configurations")
        try:
            self._prepare()
            self.manifest["candidate_plan"] = [candidate.to_dict() for candidate in candidates]
            for candidate in candidates:
                if candidate.method != "baseline":
                    self._evaluate(candidate)
            return self._finalize()
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            self._save()
            raise
        except Exception as error:
            self.manifest.update(status="failed", reason=str(error), error_type=type(error).__name__)
            self._save()
            raise

    def search(self, space, config):
        from .search import run_search
        try:
            self._prepare()
            result = run_search(tuple(space), config, self._evaluate, self.baseline_record, self.protocol,
                                start_time=self.started, event=self._event)
            self.manifest["search"] = result
            return self._finalize()
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            self._save()
            raise
        except Exception as error:
            self.manifest.update(status="failed", reason=str(error), error_type=type(error).__name__)
            self._save()
            raise

    @classmethod
    def from_manifest(cls, path, model_factory, output_dir):
        """Replay from safe tensor files with an explicitly supplied architecture."""
        manifest = load_run(path)
        files = manifest["replay_files"]
        data = torch.load(files["data.pt"]["path"], map_location="cpu", weights_only=True)
        model = model_factory()
        model.load_state_dict(torch.load(files["initial_state.pt"]["path"], map_location="cpu", weights_only=True))
        if model_state_hash(model) != manifest["initial_model_sha256"]:
            raise ProtocolError("The supplied replay architecture/weights differ from the original model")
        splits = {}
        for role in ROLES:
            info = manifest["data"]["roles"][role]
            arguments = (tuple(info["ids"]), tuple(info["unit_ids"]), tuple(tuple(pair) for pair in info["intervals"]))
            if "files" in data[role]:
                files = data[role]["files"]
                splits[role] = TensorSplit.from_npy(files["x"]["path"], files["y"]["path"], *arguments,
                                                  expected_hashes=(files["x"]["sha256"], files["y"]["sha256"]))
            else:
                splits[role] = TensorSplit(data[role]["x"], data[role]["y"], *arguments)
        bundle = ExperimentBundle(**splits, task=manifest["data"]["task"], original_model=model,
                                  metadata=manifest["data"]["metadata"])
        checkpoint = files.get("baseline.json", {}).get("path") if manifest.get("baseline_training", {}).get("training_status") == "trained_on_train" else None
        return cls(bundle, ExperimentProtocol.from_dict(manifest["protocol"]), output_dir, baseline_checkpoint=checkpoint)


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json" if path.is_dir() else path).read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ProtocolError("Unsupported run manifest version")
    for info in manifest.get("replay_files", {}).values():
        if file_hash(info["path"]) != info["sha256"]:
            raise ProtocolError("Replay file hash mismatch")
    for record in manifest.get("candidates", ()):
        if record.get("status") == "succeeded":
            load_artifact(record["measurement"]["artifact"])
    return manifest
