"""Frozen CPU masked-language-model experiment; no downloads or model selection.

The only supported adapter is the exact Transformers BERT encoder FFN. Data
planning is pure; this module owns disk/process effects and calls existing
compression, training, checkpoint and artifact-measurement components.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer, BertIntermediate, BertOutput

from fedcore.algorithm.low_rank.coupled_layers import ReferenceMLP
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair
from fedcore.algorithm.pruning.sparse.channels import partition_mlp, slice_mlp, sola, SoLASpec
from fedcore.algorithm.pruning.sparse.linear import (
    hassle_free, prepare_losparse, finalize_losparse, LoSparseTrainingCallback,
)
from fedcore.algorithm.pruning.sparse.layers import PackedNMLinear
from fedcore.algorithm.pruning.sparse.policies import prunenet, PruneNetSpec
from fedcore.algorithm.pruning.sparse.specs import HASSLEFreeSpec, NMStructure, LoSparseSpec
from fedcore.experiments.measurement import file_hash, tensor_state_bytes, measure_artifact
from fedcore.experiments.protocol import TensorSplit, tensor_hash
from fedcore.experiments.runner import train_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager

VERSION = "local-bert-mlm-v1"
BUDGET = 1_892_044
METHODS = ("hassle", "losparse", "sola", "prunenet", "svd", "norm", "random")
DETERMINISTIC = ("sola", "svd", "norm")
SOURCE_DEFAULT = Path("C:/Users/user/Documents/Codex/FedCore-sparse-sources-2026-10-14")
RESULT_DEFAULT = Path("C:/Users/user/Documents/Codex/FedCore-sparse-results/local-bert-mlm-v1")
ROLE_SEEDS = {"calibration": 101, "recovery": 103, "validation": 107, "test": 109}
ROLE_LIMITS = {"calibration": 16, "recovery": 16, "validation": 16, "test": 32}
UNUSED_SOURCE_KEYS = {"bert.pooler.dense.weight", "bert.pooler.dense.bias",
                      "cls.seq_relationship.weight", "cls.seq_relationship.bias"}


class InitialParityError(ValueError):
    def __init__(self, actual, expected):
        mismatch = ~torch.isclose(actual, expected, rtol=1e-4, atol=1e-5)
        difference = (actual-expected).abs()
        zero_references = int((expected[mismatch] == 0).sum())
        self.evidence = {"gate": "losparse_initial_logits", "training_steps": 0,
                         "mismatched_elements": int(mismatch.sum()), "total_elements": actual.numel(),
                         "max_absolute_mismatch": float(difference[mismatch].max()),
                         "max_relative_mismatch": None if zero_references else float((difference[mismatch].double()/expected[mismatch].abs().double()).max()),
                         "zero_reference_mismatches": zero_references,
                         "rtol": 1e-4, "atol": 1e-5}
        super().__init__("LoSparse initial logits failed the frozen numerical parity gate before training")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    text: str
    text_hash: str
    split: str


def parse_articles(lines, split):
    """Top headings own all subsequent subheadings and paragraphs."""
    articles, title, parts = [], "__initial__", []
    def finish():
        text = "".join(parts)
        if text.strip():
            normalized = " ".join(title.casefold().split())
            digest = hashlib.sha256(text.encode()).hexdigest()
            identifier = hashlib.sha256(normalized.encode()).hexdigest()
            articles.append(Article(identifier, normalized, text, digest, split))
    for line in lines:
        match = re.fullmatch(r"\s*=\s*([^=]+?)\s*=\s*", line)
        if match:
            finish()
            title, parts = match.group(1), [line]
        else:
            parts.append(line)
    finish()
    return articles


def assign_articles(articles):
    """Test wins overlaps; then validation; no role borrows another's data."""
    ranked = sorted(enumerate(articles), key=lambda item: ({"test": 0, "validation": 1, "train": 2}[item[1].split], item[0]))
    seen_titles, seen_texts, assignments = {}, {}, {}
    for index, article in ranked:
        role = article.split if article.split != "train" else ("recovery" if int(article.article_id, 16) & 1 else "calibration")
        conflict = seen_titles.get(article.title) or seen_texts.get(article.text_hash)
        if conflict is not None and conflict != role:
            assignments[index] = "excluded_overlap"
            continue
        assignments[index] = role
        seen_titles[article.title], seen_texts[article.text_hash] = role, role
    return tuple((article, assignments[index]) for index, article in enumerate(articles))


def mask_windows(original, mask_id, seed):
    generator = torch.Generator().manual_seed(seed)
    x, labels = original.clone(), torch.full_like(original, -100)
    for row in range(len(x)):
        positions = torch.randperm(62, generator=generator)[:9] + 1
        labels[row, positions] = original[row, positions]
        x[row, positions] = mask_id
    return x, labels


def verify_sources(root):
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in entries:
        path = Path(entry["path"])
        if file_hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source hash/size mismatch: {path}")
    return entries


def prepare_data(root, destination):
    tokenizer = BertTokenizer(vocab_file=str(root / "model/vocab.txt"), do_lower_case=True)
    articles = []
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(root / f"data/wikitext-2-raw-v1/{split}-00000-of-00001.parquet")
        articles.extend(parse_articles(frame["text"].tolist(), split))
    assigned = assign_articles(articles)
    write_json(destination / "articles.json", [dict(article_id=a.article_id, title=a.title,
               text_sha256=a.text_hash, source_split=a.split, role=role) for a, role in assigned])
    roles = {}
    for role, limit in ROLE_LIMITS.items():
        windows, ids, units = [], [], []
        for article_index, (article, actual) in enumerate(assigned):
            if actual != role:
                continue
            tokens = tokenizer.encode(article.text, add_special_tokens=False, truncation=False)
            for start in range(0, len(tokens) - 61, 62):
                windows.append([tokenizer.cls_token_id, *tokens[start:start+62], tokenizer.sep_token_id])
                ids.append(f"{article.article_id}:{article.text_hash}:{article_index}:{start}")
                units.append(article.article_id)
                if len(windows) == limit:
                    break
            if len(windows) == limit:
                break
        if len(windows) != limit:
            raise ValueError(f"Insufficient independent {role} windows: {len(windows)}/{limit}")
        original = torch.tensor(windows, dtype=torch.long)
        x, y = mask_windows(original, tokenizer.mask_token_id, ROLE_SEEDS[role])
        payload = {"original": original, "x": x, "y": y, "ids": ids, "unit_ids": units}
        torch.save(payload, destination / f"{role}.pt")
        roles[role] = {"original_hash": tensor_hash(original), "x_hash": tensor_hash(x),
                       "y_hash": tensor_hash(y), "mask_seed": ROLE_SEEDS[role],
                       "masked_tokens": int((y != -100).sum()), "window_count": len(x),
                       "article_ids": sorted(set(units))}
    for left in roles:
        for right in roles:
            if left != right and set(roles[left]["article_ids"]) & set(roles[right]["article_ids"]):
                raise ValueError("Article leakage between roles")
    write_json(destination / "data_manifest.json", roles)
    return roles


def load_role(root, role, limit=None):
    frozen = json.loads((root.parent / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(root / "data_manifest.json") != frozen["data_manifest_sha256"]:
        raise ValueError("Frozen data manifest hash mismatch")
    payload = torch.load(root / f"{role}.pt", map_location="cpu", weights_only=True)
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))[role]
    for key in ("original", "x", "y"):
        if tensor_hash(payload[key]) != manifest[key + "_hash"]:
            raise ValueError(f"Frozen {role} {key} hash mismatch")
    if int((payload["y"] != -100).sum()) != len(payload["x"])*9:
        raise ValueError("Invalid masked-token count")
    original, x, y = payload["original"], payload["x"], payload["y"]
    if (any(value.dtype != torch.long or value.shape != (ROLE_LIMITS[role], 64) for value in (original,x,y))
            or len(payload["ids"]) != len(x) or len(set(payload["ids"])) != len(x)
            or len(payload["unit_ids"]) != len(x) or not bool(((y != -100).sum(1) == 9).all())
            or not bool((original[:,0] == 101).all()) or not bool((original[:,-1] == 102).all())
            or not bool((y[:,[0,63]] == -100).all()) or not torch.equal(original[y != -100],y[y != -100])
            or not torch.equal(original[y == -100],x[y == -100]) or not bool((x[y != -100] == 103).all())):
        raise ValueError("Frozen token/mask/ID structure invalid")
    count = len(payload["x"]) if limit is None else limit
    return TensorSplit(payload["x"][:count], payload["y"][:count], tuple(payload["ids"][:count]),
                       tuple(payload["unit_ids"][:count]))


def tied_head(model):
    if model.cls.predictions.decoder.weight is not model.bert.embeddings.word_embeddings.weight:
        raise ValueError("MLM decoder and embeddings are not tied")
    if model.cls.predictions.decoder.bias is not model.cls.predictions.bias:
        raise ValueError("MLM decoder bias is not tied")


def runtime_tensor_bytes(model):
    """Count every registered tensor storage, including nonpersistent buffers."""
    storages = {}
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def load_source(root):
    config = BertConfig.from_json_file(str(root / "model/config.json"))
    model = BertForMaskedLM(config).float().eval()
    state = torch.load(root / "model/pytorch_model.bin", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or set(result.unexpected_keys) != UNUSED_SOURCE_KEYS:
        raise ValueError(f"Unadmitted source keys: {result}")
    if not all(name in state for name in model.state_dict() if name.startswith("cls.predictions.")):
        raise ValueError("Missing pretrained MLM tensor")
    tied_head(model)
    evidence = {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
                "unique_parameters": sum(p.numel() for p in model.parameters()),
                "mlm_tensor_hashes": {name: tensor_hash(state[name]) for name in state if name.startswith("cls.predictions.")}}
    if evidence["unique_parameters"] != 11_201_594:
        raise ValueError("Unexpected source parameter count")
    return model, evidence


def adapt_bert_ffns(model, layers):
    if type(model) is not BertForMaskedLM:
        raise ValueError("Exact BertForMaskedLM required")
    config = model.config
    if (config.hidden_act != "gelu" or config.is_decoder or config.add_cross_attention
            or config.chunk_size_feed_forward != 0 or config.hidden_size != 256
            or config.intermediate_size != 1024 or config.num_hidden_layers != 4):
        raise ValueError("Unsupported BERT configuration")
    for index in layers:
        layer = model.bert.encoder.layer[index]
        if (type(layer) is not BertLayer or type(layer.intermediate) is not BertIntermediate
                or type(layer.output) is not BertOutput or type(layer.intermediate.dense) is not nn.Linear
                or type(layer.output.dense) is not nn.Linear):
            raise ValueError("Exact original BERT FFN required")
        mlp = ReferenceMLP(256, 1024, gated=False, activation="gelu", bias=True)
        mlp.up_proj.load_state_dict(layer.intermediate.dense.state_dict())
        mlp.down_proj.load_state_dict(layer.output.dense.state_dict())
        x = torch.linspace(-2, 2, 64*256).reshape(64, 256)
        with torch.inference_mode():
            torch.testing.assert_close(mlp(x), layer.output.dense(layer.intermediate(x)), rtol=1e-5, atol=1e-6)
        layer.intermediate, layer.output.dense = mlp, nn.Identity()
    model.eval()
    tied_head(model)
    return model


def logits(model, x):
    return model(input_ids=x, attention_mask=torch.ones_like(x), token_type_ids=torch.zeros_like(x)).logits


def evaluate_mlm(model, split, return_logits=False):
    model.eval()
    losses, correct, outputs = [], 0, []
    with torch.inference_mode():
        for x, y in zip(split.x, split.y):
            scores = logits(model, x[None])[0]
            loss = F.cross_entropy(scores, y, ignore_index=-100, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite MLM loss")
            losses.append(float(loss))
            correct += int(((scores.argmax(-1) == y) & (y != -100)).sum())
            if return_logits:
                outputs.append(scores.cpu())
    tokens = int((split.y != -100).sum())
    nll = math.fsum(losses) / tokens
    result = {"loss_sum": math.fsum(losses), "masked_tokens": tokens, "mlm_nll": nll,
              "exp_masked_token_loss": math.exp(nll), "top1_accuracy": correct / tokens,
              "window_nll": [loss / 9 for loss in losses], "unit_ids": list(split.unit_ids)}
    return (result, torch.stack(outputs)) if return_logits else result


def save_bundle(model, destination, layers, source_root):
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.config.to_json_file(str(destination / "config.json"))
    BertTokenizer(vocab_file=str(source_root / "model/vocab.txt"), do_lower_case=True).save_pretrained(destination / "tokenizer")
    manager = CheckpointManager(str(destination), auto_cleanup=False)
    prefixes = tuple(f"bert.encoder.layer.{i}.intermediate." for i in layers)
    state = {name: value for name, value in model.state_dict().items() if not name.startswith(prefixes)}
    torch.save(state, destination / "surrounding.pt")
    for index in layers:
        manager.save_to_file(manager.serialize_to_bytes(model.bert.encoder.layer[index].intermediate),
                             str(destination / f"ffn_{index}.pt"), cleanup_after_save=False)
    write_json(destination / "bundle.json", {"version": VERSION, "layers": list(layers),
               "excluded_prefixes": prefixes, "tied_decoder": True})
    files = {str(p.relative_to(destination)): {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "files.json", files)
    return {"files": files, "file_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
            "save_seconds": time.perf_counter() - started}


def load_bundle(destination):
    files = json.loads((destination / "files.json").read_text(encoding="utf-8"))
    for name, description in files.items():
        if file_hash(destination / name) != description["sha256"]:
            raise ValueError("Bundle hash mismatch")
    metadata = json.loads((destination / "bundle.json").read_text(encoding="utf-8"))
    config = BertConfig.from_json_file(str(destination / "config.json"))
    if (config.hidden_size != 256 or config.intermediate_size != 1024 or config.num_hidden_layers != 4
            or config.vocab_size != 30522 or config.hidden_act != "gelu" or config.is_decoder
            or config.max_position_embeddings != 512 or config.type_vocab_size != 2
            or config.num_attention_heads != 4
            or config.add_cross_attention or config.chunk_size_feed_forward != 0
            or metadata["version"] != VERSION or len(set(metadata["layers"])) != len(metadata["layers"])
            or any(type(i) is not int or not 0 <= i < 4 for i in metadata["layers"])
            or metadata["excluded_prefixes"] != [f"bert.encoder.layer.{i}.intermediate." for i in metadata["layers"]]):
        raise ValueError("Unsupported or unbounded bundle configuration")
    model = BertForMaskedLM(config).float()
    manager = CheckpointManager(str(destination), auto_cleanup=False)
    for index in metadata["layers"]:
        layer = model.bert.encoder.layer[index]
        layer.intermediate = manager.load_from_file(str(destination / f"ffn_{index}.pt"), device="cpu")
        layer.output.dense = nn.Identity()
    state = torch.load(destination / "surrounding.pt", weights_only=True, map_location="cpu")
    old_prefixes = tuple(f"bert.encoder.layer.{i}.{suffix}" for i in metadata["layers"]
                         for suffix in ("intermediate.", "output.dense."))
    if any(name.startswith(old_prefixes) for name in state):
        raise ValueError("Replaced FFN state remains in surrounding checkpoint")
    expected = {name for name in model.state_dict() if name.startswith(tuple(metadata["excluded_prefixes"]))}
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError(f"Invalid surrounding state keys: {result}")
    model.tie_weights()
    tied_head(model)
    BertTokenizer.from_pretrained(str(destination / "tokenizer"), local_files_only=True)
    return model.eval()


def collect_inputs(model, split, layers):
    rows = {index: [] for index in layers}
    handles = [model.bert.encoder.layer[index].intermediate.register_forward_pre_hook(
        lambda module, args, index=index: rows[index].append(args[0].detach().reshape(-1, 256).clone())) for index in layers]
    try:
        with torch.inference_mode():
            for batch in split.x.split(2):
                logits(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return {index: torch.cat(values) for index, values in rows.items()}


def build_candidate(model, layers, method, seed, calibration, recovery):
    started, evidence = time.perf_counter(), {}
    dense_rows = None if method.startswith("hassle") else collect_inputs(model, calibration, layers)
    if method.startswith("losparse"):
        modules = []
        spec = LoSparseSpec(64, 494, 8, warmup_steps=0 if method.endswith("immediate") else 2,
                            final_step=1 if method.endswith("immediate") else 8, delta_t=1,
                            ema_beta=.85, allow_reentry=False, initial_scale=1.)
        model.requires_grad_(False)
        before = evaluate_mlm(model, calibration, True)[1]
        for index in layers:
            mlp = model.bert.encoder.layer[index].intermediate
            mlp.up_proj = prepare_losparse(mlp.up_proj, spec)
            mlp.up_proj.requires_grad_(True)
            modules.append(mlp.up_proj)
        after = evaluate_mlm(model, calibration, True)[1]
        if not torch.allclose(after, before, rtol=1e-4, atol=1e-5):
            raise InitialParityError(after, before)
        def loss_fn(model, x, y):
            model.eval()  # dropout disabled; gradients remain enabled
            return F.cross_entropy(logits(model, x).reshape(-1, model.config.vocab_size),
                                   y.reshape(-1), ignore_index=-100)
        callback = LoSparseTrainingCallback(modules, loss_fn)
        evidence["training"] = train_model(model, recovery, "language_model", epochs=1, batch_size=2,
                               learning_rate=1e-4, seed=seed, batch_objective=callback, max_updates=8)
        evidence["callback"] = callback.evidence
        for index in layers:
            mlp = model.bert.encoder.layer[index].intermediate
            mlp.up_proj = finalize_losparse(mlp.up_proj)
        evidence["spec"] = asdict(spec)
    else:
        for index in layers:
            mlp = model.bert.encoder.layer[index].intermediate
            if method.startswith("hassle"):
                rows = collect_inputs(model, calibration, (index,))[index]
                if method.endswith("diagonal"):
                    diagonal = rows.double().square().mean(0)
                    rows = (torch.diag((256*diagonal).sqrt())).float()
                spec = HASSLEFreeSpec(NMStructure(1, 4, 10), outer_steps=2, inner_steps=4,
                                     learning_rate=.01, damping=.01, seed=seed)
                mlp.up_proj, details = hassle_free(mlp.up_proj, rows, spec)
                evidence[str(index)] = {"spec": asdict(spec), "evidence": asdict(details)}
            elif method == "svd":
                u, s, vh = torch.linalg.svd(mlp.up_proj.weight.detach().double(), full_matrices=False)
                root = s[:163].sqrt()
                pair = FactorPair((u[:, :163]*root).float(), (root[:, None]*vh[:163]).float())
                mlp.up_proj = FactorizedLinear.from_factors(pair, mlp.up_proj.bias)
            elif method in ("norm", "random"):
                if method == "norm":
                    chosen = mlp.up_proj.weight.detach().norm(dim=1).argsort(descending=True, stable=True)[:914].sort().values.tolist()
                else:
                    chosen = torch.randperm(1024, generator=torch.Generator().manual_seed(seed+index))[:914].sort().values.tolist()
                compact, details = slice_mlp(mlp, tuple(chosen))
                model.bert.encoder.layer[index].intermediate = compact
                evidence[str(index)] = details
            elif method == "sola":
                spec = SoLASpec(prime_count=128, activity_point="post_activation", parameter_budget=470963,
                                 rank_multiple=1, rank=None, allow_dense=True, ridge=1e-6,
                                 max_workspace_bytes=256*1024**2)
                compact, details = sola(mlp, dense_rows[index], spec)
                model.bert.encoder.layer[index].intermediate = compact
                evidence[str(index)] = {"spec": asdict(spec), "evidence": details}
            elif method == "prunenet":
                spec = PruneNetSpec(keep_count=914, episodes=4, seed=seed, max_workspace_bytes=256*1024**2)
                compact, details = prunenet(mlp, spec)
                model.bert.encoder.layer[index].intermediate = compact
                evidence[str(index)] = {"spec": asdict(spec), "evidence": details}
            elif method != "dense":
                raise ValueError("Unknown frozen method")
            if time.perf_counter() - started > 600:
                raise TimeoutError("Candidate construction exceeds 10 minutes")
    model.eval()
    return evidence, time.perf_counter() - started


def time_full_model(model, x, repeats):
    raw = []
    with torch.inference_mode():
        for _ in range(3):
            logits(model, x)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            logits(model, x)
            raw.append((time.perf_counter_ns()-start)/1e6)
    return {"raw_ms": raw, "median_ms": float(np.median(raw)), "p95_ms": float(np.percentile(raw, 95)),
            "threads": 4, "batch_size": 1, "sequence_length": 64, "warmup": 3, "repeats": repeats}


def reload_worker(bundle, data, result_path, stage):
    started = time.perf_counter()
    model = load_bundle(bundle)
    result = {"load_seconds": time.perf_counter()-started, "model_tensor_bytes": tensor_state_bytes(model)}
    for role in (("validation", "test") if stage == "main" else ("validation",)):
        split = load_role(data, role, 8 if stage == "pilot" else None)
        quality, scores = evaluate_mlm(model, split, True)
        result[role] = quality
        torch.save(scores, result_path.parent / f"reloaded_{role}_logits.pt")
    result["timing"] = time_full_model(model, load_role(data, "validation").x[:1], 5 if stage == "pilot" else 10)
    result["rss_snapshot_bytes"] = psutil.Process().memory_info().rss
    write_json(result_path, result)


def candidate_worker(source, destination, data, method, seed, stage):
    destination.mkdir(parents=True, exist_ok=True)
    result = {"method": method, "seed": seed, "stage": stage, "status": "running",
              "experiment_module_sha256": file_hash(__file__),
              "checkpoint_module_sha256": file_hash(Path(__file__).parents[1]/"tools/registry/checkpoint_manager.py")}
    def progress(step):
        result["last_step"] = step
        result["rss_snapshot_bytes"] = psutil.Process().memory_info().rss
        write_json(destination / "progress.json", result)
    try:
        progress("source_verification")
        verify_sources(source)
        model, source_evidence = load_source(source)
        layers = (1,) if stage == "pilot" else (0, 1, 2, 3)
        split = load_role(data, "validation", 8 if stage == "pilot" else None)
        source_quality, source_scores = evaluate_mlm(model, split, True)
        adapt_bert_ffns(model, layers)
        adapted_quality, adapted_scores = evaluate_mlm(model, split, True)
        torch.testing.assert_close(adapted_scores, source_scores, rtol=1e-4, atol=1e-5)
        if abs(adapted_quality["mlm_nll"] - source_quality["mlm_nll"]) > 1e-5:
            raise ValueError("Dense adapter MLM loss parity failed")
        if source_quality["mlm_nll"] >= math.log(30522) - .5:
            raise ValueError("Pretrained dense MLM quality gate failed")
        result.update(source=source_evidence, dense_validation=source_quality, adapter_gate=True,
                      changed_layers=list(layers), dense_model_tensor_bytes=tensor_state_bytes(model))
        progress("adapter_gate_passed")
        cal, recovery = load_role(data, "calibration"), load_role(data, "recovery")
        if method == "dense":
            controls = {}
            mlp = model.bert.encoder.layer[layers[0]].intermediate
            rows = collect_inputs(model, cal, (layers[0],))[layers[0]]
            exact, _ = partition_mlp(mlp, tuple(range(128)))
            torch.testing.assert_close(exact(rows), mlp(rows), rtol=1e-5, atol=1e-6)
            packed = PackedNMLinear(256, 1024, n=2, m=4, rank=0, bias=True)
            controls["hassle_2_4_ffn_bytes"] = tensor_state_bytes(packed)+tensor_state_bytes(mlp.down_proj)
            controls["sola_exact_partition_parity"] = True
            result["controls"] = controls
        progress("candidate_build")
        evidence, build_seconds = build_candidate(model, layers, method, seed, cal, recovery)
        result.update(method_evidence=evidence, build_seconds=build_seconds)
        ffn_bytes = {str(i): tensor_state_bytes(model.bert.encoder.layer[i].intermediate) for i in layers}
        if method != "dense" and any(value > BUDGET for value in ffn_bytes.values()):
            result["ffn_tensor_bytes"] = ffn_bytes
            raise ValueError("Actual FFN bytes exceed frozen budget")
        result.update(ffn_tensor_bytes=ffn_bytes, model_tensor_bytes=tensor_state_bytes(model),
                      budget_match=method == "dense" or all(abs(BUDGET-v) <= .01*BUDGET for v in ffn_bytes.values()))
        bundle = destination / "bundle"
        progress("bundle_save")
        result["bundle"] = save_bundle(model, bundle, layers, source)
        exports = {}
        progress("ffn_export_reload")
        rows = collect_inputs(model, cal, layers)
        for index in layers:
            exports[str(index)] = measure_artifact(model.bert.encoder.layer[index].intermediate,
                rows[index][:64], destination / f"ffn_{index}.ts", repeats=5 if stage == "pilot" else 10,
                warmup=3, threads=4)
        result["ffn_artifacts"] = exports
        if any(item["status"] != "succeeded" for item in exports.values()):
            raise ValueError("FFN TorchScript export/reload failed")
        # Validate the complete bundle before reading any test tokens for quality.
        progress("bundle_parity")
        local_reload = load_bundle(bundle)
        local_quality, local_scores = evaluate_mlm(local_reload, split, True)
        quality, scores = evaluate_mlm(model, split, True)
        torch.testing.assert_close(local_scores, scores, rtol=1e-4, atol=1e-5)
        if abs(local_quality["mlm_nll"]-quality["mlm_nll"]) > 1e-5:
            raise ValueError("Bundle MLM loss parity failed")
        del local_reload
        result["validation"] = quality
        result["validation_delta"] = quality["mlm_nll"]-source_quality["mlm_nll"]
        torch.save(scores, destination / "expected_validation_logits.pt")
        if stage == "main":
            test = load_role(data, "test")
            quality, scores = evaluate_mlm(model, test, True)
            result["test"] = quality
            torch.save(scores, destination / "expected_test_logits.pt")
        command = [sys.executable, str(Path(__file__).parents[2]/"scripts/run_sparse_pretrained.py"),
                   "--stage", "reload", "--bundle", str(bundle), "--data", str(data),
                   "--output", str(destination / "reload.json"), "--series-stage", stage]
        progress("fresh_process_reload")
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (destination / "reload.log").write_text(process.stdout+process.stderr, encoding="utf-8")
        if process.returncode:
            raise ValueError("Fresh-process bundle reload failed")
        reloaded = json.loads((destination / "reload.json").read_text(encoding="utf-8"))
        for role in (("validation", "test") if stage == "main" else ("validation",)):
            expected = torch.load(destination / f"expected_{role}_logits.pt", weights_only=True)
            actual = torch.load(destination / f"reloaded_{role}_logits.pt", weights_only=True)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            if abs(reloaded[role]["mlm_nll"] - result[role]["mlm_nll"]) > 1e-5:
                raise ValueError("Fresh-process MLM loss parity failed")
        result.update(reload=reloaded, transfer_gate=True, status="succeeded")
        result["rss_snapshot_bytes"] = psutil.Process().memory_info().rss
        if result["rss_snapshot_bytes"] > 15*1024**3:
            raise MemoryError("RSS limit exceeded")
    except Exception as error:
        result.update(status="failed", reason=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        if isinstance(error, InitialParityError):
            result["initial_parity"] = error.evidence
    write_json(destination / "result.json", result)
    return result


def freeze_run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    sources = verify_sources(source)
    data = output / "data"
    if not (data / "data_manifest.json").exists():
        data.mkdir(exist_ok=True)
        prepare_data(source, data)
    protocol_path = Path(__file__).parents[2] / "docs/sparse_compression/LOCAL_EXPERIMENT_PROTOCOL.md"
    frozen = {"version": VERSION, "protocol_sha256": file_hash(protocol_path), "sources": sources,
              "mask_seeds": ROLE_SEEDS, "role_limits": ROLE_LIMITS, "seeds": [17, 29],
              "budget_per_ffn": BUDGET, "quality_delta_limit": .10, "threads": 4,
              "total_seconds_limit": 7200, "candidate_seconds_limit": 600, "rss_limit_bytes": 15*1024**3,
              "methods": METHODS, "ablations": ["losparse_immediate", "hassle_diagonal"],
              "data_manifest_sha256": file_hash(data / "data_manifest.json")}
    frozen_path = output / "protocol.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text(encoding="utf-8")) != json.loads(json.dumps(frozen)):
            raise ValueError("Frozen protocol differs; refusing to overwrite")
    else:
        write_json(frozen_path, frozen)
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "pandas": pd.__version__, "transformers": __import__("transformers").__version__,
                   "pyarrow": __import__("pyarrow").__version__, "platform": platform.platform(),
                   "logical_cpus": psutil.cpu_count(), "ram": dict(psutil.virtual_memory()._asdict()),
                   "tokenizer": {"class": "BertTokenizer", "do_lower_case": True},
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                   "protocol_sha256": file_hash(frozen_path)}
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", environment)
    write_json(output / f"environment_{time.time_ns()}.json", environment)
    return data


def run_bounded_case(command, destination, series_started_unix, method, seed):
    """One process family, sampled RSS guard and a strict construction deadline."""
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "worker.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        candidate_started = time.monotonic()
        reason, observed_family_rss = None, []
        while process.poll() is None:
            try:
                family = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                rss = sum(member.memory_info().rss for member in family)
                observed_family_rss.append(rss)
                if rss > 15*1024**3:
                    reason = "Worker RSS exceeds 15 GiB"
                if time.time()-series_started_unix > 7200:
                    reason = "Two-hour series limit"
                if time.monotonic()-candidate_started > 600:
                    reason = "Candidate worker exceeds 10 minutes"
            except psutil.NoSuchProcess:
                break
            if reason:
                for member in reversed(family):
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(.5)
        process.wait()
    path = destination / "result.json"
    if path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
    else:
        progress_path = destination / "progress.json"
        row = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"method": method, "seed": seed}
        row.update(status="failed", reason=reason or f"Worker exit {process.returncode}")
    row["observed_family_rss_max_bytes"] = max(observed_family_rss, default=0)
    row["rss_kind"] = "Maximum sampled process-family RSS; not model-exclusive or allocator peak"
    write_json(path, row)
    return row


def retry_pilot(source, output, method):
    """Exactly one same-specification technical retry; original evidence stays."""
    if method != "hassle":
        raise ValueError("Only the known HASSLE nested-checkpoint technical refusal is retryable")
    report_path = output / "pilot/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    original = next(row for row in report["results"] if row["method"] == method)
    if original["status"] != "failed" or original.get("error_type") != "CheckpointError":
        raise ValueError("The original result is not the known technical checkpoint failure")
    destination = output / f"pilot/{method}_seed17_retry1"
    if destination.exists():
        raise ValueError("The one permitted technical retry was already used")
    data = freeze_run(source, output)
    series_started = json.loads((output / "series_budget.json").read_text())["started_unix"]
    command = [sys.executable, str(Path(__file__).parents[2]/"scripts/run_sparse_pretrained.py"),
               "--stage", "worker", "--series-stage", "pilot", "--method", method, "--seed", "17",
               "--source", str(source), "--output", str(destination), "--data", str(data)]
    print(f"Starting technical retry {method} seed17", flush=True)
    row = run_bounded_case(command, destination, series_started, method, 17)
    row["technical_attempt"] = 2
    row["retry_of"] = str(output / f"pilot/{method}_seed17/result.json")
    write_json(destination / "result.json", row)
    report["results"].append(row)
    write_json(report_path, report)
    print(f"Finished technical retry {method}: {row['status']} {row.get('reason','')}", flush=True)
    return row


def audit_original_dense(source, output):
    """Evaluate the original BERT architecture once against the admitted control."""
    target = output / "main/source_dense_control.json"
    if target.exists():
        raise ValueError("Original dense control already evaluated; no implicit rerun")
    dense = json.loads((output / "main/dense_seed17/result.json").read_text(encoding="utf-8"))
    if dense["status"] != "succeeded" or not dense.get("transfer_gate"):
        raise ValueError("Dense bundle was not admitted")
    verify_sources(source)
    model, evidence = load_source(source)
    result = {"method": "original_dense", "status": "succeeded", "source": evidence,
              "model_tensor_bytes": tensor_state_bytes(model), "runtime_whole_model_tensor_bytes": runtime_tensor_bytes(model),
              "nonpersistent_buffer_extra_bytes": runtime_tensor_bytes(model)-tensor_state_bytes(model),
              "comparison": "original BERT versus saved adapted dense logits"}
    for role in ("validation", "test"):
        quality, actual = evaluate_mlm(model, load_role(output / "data", role), True)
        expected = torch.load(output / f"main/dense_seed17/expected_{role}_logits.pt", weights_only=True)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        if abs(quality["mlm_nll"] - dense[role]["mlm_nll"]) > 1e-5:
            raise ValueError("Original/adapted dense MLM loss mismatch")
        result[role] = quality
        result[role + "_maximum_logits_difference"] = float((actual-expected).abs().max())
    write_json(target, result)
    return result


def audit_runtime_storage(output):
    """Augment existing records; do not rebuild or reevaluate any candidate."""
    for stage in ("pilot", "main"):
        report_path = output / stage / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report["results"]:
            if row["status"] != "succeeded":
                continue
            suffix = "_retry1" if row.get("technical_attempt") == 2 else ""
            destination = output / stage / f"{row['method']}_seed{row['seed']}{suffix}"
            model = load_bundle(destination / "bundle")
            state_bytes = tensor_state_bytes(model)
            if state_bytes != row["model_tensor_bytes"]:
                raise ValueError("State storage changed during runtime audit")
            runtime_bytes = runtime_tensor_bytes(model)
            row["runtime_whole_model_tensor_bytes"] = runtime_bytes
            row["nonpersistent_buffer_extra_bytes"] = runtime_bytes-state_bytes
            row["storage_kind"] = {"model_tensor_bytes": "unique state_dict storages",
                                   "runtime_whole_model_tensor_bytes": "unique parameters and all registered buffers"}
            row["storage_audit"] = {"procedure": "reload the admitted bundle; no source factory, reconstruction search or quality reevaluation",
                                    "experiment_module_sha256": file_hash(__file__),
                                    "checkpoint_module_sha256": file_hash(Path(__file__).parents[1]/"tools/registry/checkpoint_manager.py")}
            del model
            gc.collect()
            write_json(destination / "result.json", row)
        write_json(report_path, report)


def time_bundle_worker(bundle, data, destination):
    """Fresh-process timing only: never load source weights or evaluate quality."""
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = load_bundle(bundle)
    load_seconds = time.perf_counter()-started
    split = load_role(data, "validation")
    result = {"status": "succeeded", "timing": time_full_model(model, split.x[:1], 10),
              "load_seconds": load_seconds, "bundle_files_manifest_sha256": file_hash(bundle / "files.json"),
              "input_sha256": tensor_hash(split.x[:1]), "experiment_module_sha256": file_hash(__file__),
              "checkpoint_module_sha256": file_hash(Path(__file__).parents[1]/"tools/registry/checkpoint_manager.py"),
              "runtime_whole_model_tensor_bytes": runtime_tensor_bytes(model), "quality_evaluated": False,
              "source_factory_used": False, "reconstruction_or_training": False}
    write_json(destination / "result.json", result)


def audit_timing(output):
    """One corrective pass for the dense-first ordering deviation; keep both."""
    report_path = output / "main/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    destination = output / "main/timing_audit"
    if destination.exists():
        raise ValueError("The one permitted corrective timing audit was already used")
    destination.mkdir()
    rows = [row for row in report["results"] if row["status"] == "succeeded"]
    np.random.default_rng(211).shuffle(rows)
    audit = {"reason": "Original compact order was shuffled but dense was measured first; all variants must be shuffled",
             "original_raw_retained_in": "main/report.json results[].reload.timing and candidate reload.json",
             "seed": 211, "order": [{"method":row["method"],"seed":row["seed"]} for row in rows],
             "quality_evaluated": False, "rebuilt_or_trained": False, "timings": [],
             "protocol_sha256": file_hash(output / "protocol.json"), "started_unix": time.time()}
    write_json(destination / "audit.json", audit)
    series_started = json.loads((output / "series_budget.json").read_text())["started_unix"]
    for row in rows:
        if time.time()-series_started > 7200:
            audit["stop_reason"] = "Original two-hour budget exhausted"
            break
        name = f"{row['method']}_seed{row['seed']}"
        bundle = output / "main" / name / "bundle"
        case_destination = destination / name
        command = [sys.executable, str(Path(__file__).parents[2]/"scripts/run_sparse_pretrained.py"),
                   "--stage", "time-bundle", "--bundle", str(bundle), "--data", str(output / "data"),
                   "--output", str(case_destination)]
        print(f"Corrected timing {name}", flush=True)
        measured = run_bounded_case(command, case_destination, series_started, row["method"], row["seed"])
        measured.update(method=row["method"], seed=row["seed"])
        audit["timings"].append(measured)
        write_json(case_destination / "result.json", measured)
        write_json(destination / "audit.json", audit)
        if measured["status"] != "succeeded":
            audit["stop_reason"] = measured.get("reason", "Timing worker failed")
            break
    audit["status"] = "succeeded" if len(audit["timings"]) == len(rows) and "stop_reason" not in audit else "failed"
    if audit["status"] == "succeeded":
        baseline = next(item for item in audit["timings"] if item["method"] == "dense")
        for row in report["results"]:
            match = next(item for item in audit["timings"] if (item["method"],item["seed"]) == (row["method"],row["seed"]))
            row["corrected_timing"] = match
            dense_times, compact_times = baseline["timing"]["raw_ms"], match["timing"]["raw_ms"]
            generator = np.random.default_rng(211)
            ratios = [np.median(generator.choice(dense_times,len(dense_times)))/np.median(generator.choice(compact_times,len(compact_times))) for _ in range(2000)]
            row["corrected_full_speed_ratio"] = float(np.median(dense_times)/np.median(compact_times))
            row["corrected_speed_ratio_bootstrap_95"] = np.percentile(ratios,[2.5,97.5]).tolist()
            row["corrected_local_acceleration_pass"] = bool(row["corrected_full_speed_ratio"] >= 1/.9 and row["corrected_speed_ratio_bootstrap_95"][0] > 1)
            row["original_timing_numeric_acceleration_pass"] = row["local_acceleration_pass"]
            row["local_acceleration_pass"] = row["corrected_local_acceleration_pass"]
            row["original_timing_order_deviation"] = "dense fixed first; retained as provisional observations"
            write_json(output / f"main/{row['method']}_seed{row['seed']}/result.json", row)
    report["timing_audit"] = {"status": audit["status"], "path": str(destination / "audit.json"),
                              "reason": audit["reason"], "order": audit["order"]}
    audit["elapsed_seconds"] = time.time()-audit["started_unix"]
    write_json(destination / "audit.json", audit)
    write_json(report_path, report)
    return audit


def run_study(source, output, stage):
    started = time.monotonic()
    data = freeze_run(source, output)
    budget_path = output / "series_budget.json"
    if budget_path.exists():
        series_started_unix = json.loads(budget_path.read_text())["started_unix"]
    else:
        series_started_unix = time.time()
        write_json(budget_path, {"started_unix": series_started_unix, "limit_seconds": 7200})
    if stage == "main":
        pilot = json.loads((output / "pilot/report.json").read_text(encoding="utf-8"))
        dense = next(row for row in pilot["results"] if row["method"] == "dense")
        if dense["status"] != "succeeded":
            raise ValueError("Pilot dense adapter/bundle gate failed")
        admitted = [row["method"] for row in pilot["results"] if row["status"] == "succeeded" and row.get("budget_match", False)]
        cases = [(method, seed) for seed in (17, 29) for method in METHODS if method in admitted
                 and not (seed == 29 and method in DETERMINISTIC)]
        cases += [(method, seed) for seed in (17, 29) for method in ("losparse_immediate", "hassle_diagonal")
                  if method.split("_")[0] in admitted]
    else:
        cases = [(method, 17) for method in METHODS]
    np.random.default_rng(211).shuffle(cases)
    cases.insert(0, ("dense", 17))
    series = output / stage
    series.mkdir(exist_ok=True)
    report = {"stage": stage, "results": [], "deterministic_reused_seed29": list(DETERMINISTIC),
              "timing_order_seed": 211, "started_unix": time.time()}
    for method, seed in cases:
        if time.time()-series_started_unix > 7200:
            report["stop_reason"] = "Two-hour series limit"
            break
        destination = series / f"{method}_seed{seed}"
        if (destination / "result.json").exists():
            raise ValueError("Existing candidate evidence; no implicit reruns")
        print(f"Starting {stage} {method} seed={seed}", flush=True)
        command = [sys.executable, str(Path(__file__).parents[2]/"scripts/run_sparse_pretrained.py"),
                   "--stage", "worker", "--series-stage", stage, "--method", method, "--seed", str(seed),
                   "--source", str(source), "--output", str(destination), "--data", str(data)]
        row = run_bounded_case(command, destination, series_started_unix, method, seed)
        report["results"].append(row)
        write_json(series / "report.json", report)
        print(f"Finished {method}: {row['status']} {row.get('reason','')}", flush=True)
        if method == "dense" and row["status"] != "succeeded":
            report["stop_reason"] = "Dense adapter/artifact gate failed"
            break
    dense = next((row for row in report["results"] if row["method"] == "dense"), None)
    for row in report["results"]:
        if stage == "main" and row.get("test") and dense and dense.get("test"):
            row["test_delta"] = row["test"]["mlm_nll"] - dense["test"]["mlm_nll"]
            row["quality_pass"] = row["test_delta"] <= .10
            row["paired_test_window_delta"] = [a-b for a,b in zip(row["test"]["window_nll"], dense["test"]["window_nll"])]
            row["useful_storage"] = bool(row["quality_pass"] and row.get("transfer_gate") and row.get("budget_match")
                and row["model_tensor_bytes"] < dense["model_tensor_bytes"])
            compact_times = row["reload"]["timing"]["raw_ms"]
            dense_times = dense["reload"]["timing"]["raw_ms"]
            generator = np.random.default_rng(211)
            ratios = [np.median(generator.choice(dense_times, len(dense_times)))/np.median(generator.choice(compact_times, len(compact_times))) for _ in range(2000)]
            row["observed_full_speed_ratio"] = float(np.median(dense_times)/np.median(compact_times))
            row["speed_ratio_bootstrap_95"] = np.percentile(ratios,[2.5,97.5]).tolist()
            row["local_acceleration_pass"] = bool(row["observed_full_speed_ratio"] >= 1/.9 and row["speed_ratio_bootstrap_95"][0] > 1)
    report["elapsed_seconds"] = time.monotonic()-started
    write_json(series / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main", "worker", "reload", "retry", "source-audit", "storage-audit", "time-bundle", "timing-audit"), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=RESULT_DEFAULT)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--method", choices=(*METHODS, "dense", "losparse_immediate", "hassle_diagonal"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--series-stage", choices=("pilot", "main"), default="pilot")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.stage == "time-bundle":
        time_bundle_worker(args.bundle, args.data, args.output)
    elif args.stage == "timing-audit":
        audit_timing(args.output)
    elif args.stage == "reload":
        reload_worker(args.bundle, args.data, args.output, args.series_stage)
    elif args.stage == "storage-audit":
        audit_runtime_storage(args.output)
    elif args.stage == "source-audit":
        audit_original_dense(args.source, args.output)
    elif args.stage == "retry":
        retry_pilot(args.source, args.output, args.method)
    elif args.stage == "worker":
        result = candidate_worker(args.source, args.output, args.data, args.method, args.seed, args.series_stage)
        print(json.dumps({key:result.get(key) for key in ("method", "status", "reason", "validation_delta", "ffn_tensor_bytes")}))
    else:
        run_study(args.source, args.output, args.stage)
