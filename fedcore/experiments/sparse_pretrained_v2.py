"""Independent-article v2 series using the unchanged BERT bundle schema v1.

No runtime monkeypatches or v1 global changes. Logit checks visit every element
one window at a time; only pre-transfer reference windows are written to disk.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import torch
from torch.nn import functional as F
from transformers import BertTokenizer

from fedcore.experiments import sparse_pretrained as v1
from fedcore.experiments.protocol import TensorSplit, tensor_hash
from fedcore.experiments.measurement import file_hash, tensor_state_bytes, measure_artifact
from fedcore.algorithm.pruning.sparse.linear import prepare_losparse, finalize_losparse, LoSparseTrainingCallback
from fedcore.algorithm.pruning.sparse.specs import LoSparseSpec
from fedcore.experiments.runner import train_model

VERSION = "local-bert-mlm-v2"
ROLE_PLAN = {"calibration": (32, 2), "recovery": (16, 1), "validation": (32, 2), "test": (48, 2)}
ROLE_SEEDS = v1.ROLE_SEEDS
RESULT_DEFAULT = Path("D:/data_old/WORK/obsidian/Papers_2025/Проекты для разработки/Исследовательские проекты/Fedcore/work/sparse_experiments/local-bert-mlm-v2")
REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/run_sparse_pretrained_v2.py"
PROTOCOL = REPO / "docs/sparse_compression/LOCAL_EXPERIMENT_PROTOCOL_V2.md"
write_json = v1.write_json


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan_article_groups(articles):
    """Return one record per input; transitive title/text components own roles.

    Canonical ID is the minimum ID of the ENTIRE component. Owner split has
    test > validation > train priority. Exactly one canonical representative in
    that split is eligible; all aliases retain their group/owner and a reason.
    Choice is invariant to input permutation, except indistinguishable aliases.
    """
    parent = list(range(len(articles)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        left, right = find(i), find(j)
        if left != right:
            parent[max(left, right)] = min(left, right)
    titles, texts = {}, {}
    for i, article in enumerate(articles):
        for key, table in ((article.title, titles), (article.text_hash, texts)):
            if key in table:
                union(i, table[key])
            else:
                table[key] = i
    components = {}
    for i in range(len(articles)):
        components.setdefault(find(i), []).append(i)
    records = [None] * len(articles)
    priority = {"test": 0, "validation": 1, "train": 2}
    for members in components.values():
        canonical_id = min(articles[i].article_id for i in members)
        owner = min((articles[i].split for i in members), key=priority.get)
        role = owner if owner != "train" else ("recovery" if int(canonical_id, 16) & 1 else "calibration")
        eligible = [i for i in members if articles[i].split == owner]
        representative = min(eligible, key=lambda i: (articles[i].article_id, articles[i].text_hash,
                                                      articles[i].title, articles[i].text))
        for i in members:
            article = articles[i]
            reason = None if i == representative else ("excluded_overlap" if article.split != owner else "excluded_duplicate")
            records[i] = {"article_id": article.article_id, "canonical_id": canonical_id,
                          "title": article.title, "text_sha256": article.text_hash,
                          "source_split": article.split, "owner_split": owner, "role": role,
                          "eligible": i == representative, "exclusion_reason": reason,
                          "group_titles": sorted({articles[j].title for j in members}),
                          "group_text_hashes": sorted({articles[j].text_hash for j in members})}
    return tuple(records)


def select_role_windows(articles, records, role, tokenize, cls_id, sep_id, article_count, windows_per_article):
    """Pure deterministic selection given a tokenizer function and fixed plan."""
    eligible = [(article, record) for article, record in zip(articles, records)
                if record["eligible"] and record["role"] == role]
    eligible.sort(key=lambda item: (digest(f"{VERSION}|article|{role}|{item[1]['canonical_id']}"), item[1]["canonical_id"]))
    chosen = []
    for article, record in eligible:
        tokens = tokenize(article.text)
        starts = list(range(0, len(tokens) - 61, 62))
        if len(starts) < windows_per_article:
            continue
        starts.sort(key=lambda start: (digest(f"{VERSION}|window|{role}|{record['canonical_id']}|{start}"), start))
        chosen.append((article, record, tokens, sorted(starts[:windows_per_article])))
        if len(chosen) == article_count:
            break
    if len(chosen) != article_count:
        raise ValueError(f"Insufficient independent {role} articles: {len(chosen)}/{article_count}")
    windows, ids, units, descriptions = [], [], [], []
    for round_index in range(windows_per_article):
        for article, record, tokens, starts in chosen:
            start = starts[round_index]
            window = [cls_id, *tokens[start:start+62], sep_id]
            unit = record["canonical_id"]
            identifier = f"{unit}:{article.text_hash}:{start}"
            windows.append(window)
            ids.append(identifier)
            units.append(unit)
            descriptions.append({"id": identifier, "article_id": unit, "source_article_id": article.article_id,
                                 "start": start, "title": article.title, "text_sha256": article.text_hash,
                                 "token_sha256": tensor_hash(torch.tensor(window, dtype=torch.long))})
    return windows, ids, units, descriptions


def prepare_data(root, destination):
    tokenizer = BertTokenizer(vocab_file=str(root / "model/vocab.txt"), do_lower_case=True)
    articles = []
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(root / f"data/wikitext-2-raw-v1/{split}-00000-of-00001.parquet")
        articles.extend(v1.parse_articles(frame["text"].tolist(), split))
    records = plan_article_groups(articles)
    roles = {}
    selected = set()
    for role, (article_count, per_article) in ROLE_PLAN.items():
        windows, ids, units, descriptions = select_role_windows(
            articles, records, role, lambda text: tokenizer.encode(text, add_special_tokens=False, truncation=False),
            tokenizer.cls_token_id, tokenizer.sep_token_id, article_count, per_article)
        original = torch.tensor(windows, dtype=torch.long)
        x, y = v1.mask_windows(original, tokenizer.mask_token_id, ROLE_SEEDS[role])
        for i, description in enumerate(descriptions):
            description.update(x_sha256=tensor_hash(x[i]), y_sha256=tensor_hash(y[i]))
        payload = {"original": original, "x": x, "y": y, "ids": ids, "unit_ids": units}
        torch.save(payload, destination / f"{role}.pt")
        group_records = [r for r in records if r["eligible"] and r["canonical_id"] in set(units)]
        roles[role] = {"original_hash": tensor_hash(original), "x_hash": tensor_hash(x), "y_hash": tensor_hash(y),
                       "mask_seed": ROLE_SEEDS[role], "masked_tokens": len(x)*9, "window_count": len(x),
                       "article_count": len(set(units)), "windows_per_article": per_article,
                       "article_ids": list(dict.fromkeys(units)), "windows": descriptions,
                       "all_group_titles": sorted({t for r in group_records for t in r["group_titles"]}),
                       "all_group_text_hashes": sorted({t for r in group_records for t in r["group_text_hashes"]})}
        selected.update(units)
    for record in records:
        record["selected"] = record["eligible"] and record["canonical_id"] in selected
        if record["eligible"] and not record["selected"]:
            record["exclusion_reason"] = "not_selected_by_fixed_hash_plan_or_insufficient_full_windows"
    for left in roles:
        for right in roles:
            if left != right:
                for key in ("article_ids", "all_group_titles", "all_group_text_hashes"):
                    if set(roles[left][key]) & set(roles[right][key]):
                        raise ValueError(f"Article leakage: {left}/{right}/{key}")
    write_json(destination / "articles.json", records)
    write_json(destination / "data_manifest.json", roles)
    return roles


def load_role(root, role, limit=None):
    frozen = json.loads((root.parent / "protocol.json").read_text(encoding="utf-8"))
    if (frozen["version"] != VERSION or file_hash(root / "data_manifest.json") != frozen["data_manifest_sha256"]
            or file_hash(root / "articles.json") != frozen["articles_manifest_sha256"]):
        raise ValueError("Frozen v2 data manifest mismatch")
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))[role]
    payload = torch.load(root / f"{role}.pt", map_location="cpu", weights_only=True)
    article_count, per_article = ROLE_PLAN[role]
    count = article_count * per_article
    for key in ("original", "x", "y"):
        value = payload[key]
        if value.dtype != torch.long or value.shape != (count, 64) or tensor_hash(value) != manifest[key + "_hash"]:
            raise ValueError(f"Frozen {role}/{key} changed")
    original, x, y = (payload[key] for key in ("original", "x", "y"))
    units, ids = payload["unit_ids"], payload["ids"]
    if (len(ids) != count or len(set(ids)) != count or len(units) != count
            or list(dict.fromkeys(units)) != manifest["article_ids"] or len(set(units)) != article_count
            or any(units.count(unit) != per_article for unit in set(units))
            or len(set(units[:article_count])) != article_count
            or not bool(((y != -100).sum(1) == 9).all())
            or not bool((original[:, 0] == 101).all()) or not bool((original[:, -1] == 102).all())
            or not bool((y[:, [0, 63]] == -100).all())
            or not torch.equal(original[y != -100], y[y != -100])
            or not torch.equal(original[y == -100], x[y == -100]) or not bool((x[y != -100] == 103).all())):
        raise ValueError("Frozen token/mask/article structure invalid")
    for i, desc in enumerate(manifest["windows"]):
        if desc["id"] != ids[i] or desc["article_id"] != units[i]:
            raise ValueError("Window/article identity mismatch")
        for key, expected in (("original", "token_sha256"), ("x", "x_sha256"), ("y", "y_sha256")):
            if tensor_hash(payload[key][i]) != desc[expected]:
                raise ValueError("Per-window checksum mismatch")
    stop = count if limit is None else limit
    if stop > count or stop < 1:
        raise ValueError("Invalid role limit")
    return TensorSplit(x[:stop], y[:stop], tuple(ids[:stop]), tuple(units[:stop]))


def summarize_quality(losses, correct, split):
    tokens = int((split.y != -100).sum())
    grouped = {}
    for unit, loss in zip(split.unit_ids, losses):
        grouped.setdefault(unit, []).append(loss / 9)
    nll = math.fsum(losses) / tokens
    return {"loss_sum": math.fsum(losses), "masked_tokens": tokens, "mlm_nll": nll,
            "exp_masked_token_loss": math.exp(nll), "top1_accuracy": correct/tokens,
            "window_nll": [loss/9 for loss in losses], "unit_ids": list(split.unit_ids),
            "article_nll": {unit: math.fsum(values)/len(values) for unit, values in grouped.items()}}


def stream_evaluate(model, split, reference_dir=None, write_dir=None, other_model=None):
    """Full-vocabulary, all-position comparison with O(one window) logit RAM."""
    model.eval()
    if other_model is not None:
        other_model.eval()
    if write_dir is not None:
        write_dir.mkdir(parents=True, exist_ok=False)
    reference_manifest = None
    if reference_dir is not None:
        reference_manifest = json.loads((reference_dir/"manifest.json").read_text(encoding="utf-8"))
        if len(reference_manifest["windows"]) != len(split.x):
            raise ValueError("Reference window count changed")
    losses, correct, comparisons = [], 0, []
    total, mismatches, maximum, relative, reference_losses = 0, 0, 0., 0., []
    with torch.inference_mode():
        for i, (x, y) in enumerate(zip(split.x, split.y)):
            actual = v1.logits(model, x[None])[0]
            if not bool(torch.isfinite(actual).all()):
                raise ValueError("Nonfinite full logits")
            loss = F.cross_entropy(actual, y, ignore_index=-100, reduction="sum")
            losses.append(float(loss))
            correct += int(((actual.argmax(-1) == y) & (y != -100)).sum())
            item = {"id": split.ids[i], "shape": list(actual.shape), "sha256": tensor_hash(actual)}
            if write_dir is not None:
                file = write_dir / f"{i:04d}.npy"
                np.save(file, actual.numpy(), allow_pickle=False)
                item.update(file=file.name, file_sha256=file_hash(file), bytes=file.stat().st_size)
            expected = None
            if other_model is not None:
                expected = v1.logits(other_model, x[None])[0]
            elif reference_dir is not None:
                reference_file = reference_dir/f"{i:04d}.npy"
                description = reference_manifest["windows"][i]
                if description["id"] != split.ids[i] or file_hash(reference_file) != description["file_sha256"]:
                    raise ValueError("Reference window identity/hash changed")
                expected = torch.from_numpy(np.load(reference_file, allow_pickle=False))
            if expected is not None:
                if expected.shape != actual.shape or not bool(torch.isfinite(expected).all()):
                    raise ValueError("Reference logit shape/finite check failed")
                difference = (actual-expected).abs()
                mismatch = ~torch.isclose(actual, expected, rtol=1e-4, atol=1e-5)
                n = int(mismatch.sum())
                maximum = max(maximum, float(difference.max()))
                relative = max(relative, float((difference.double() / expected.abs().double().clamp_min(1e-300)).clamp_max(1e300).max()))
                total += actual.numel()
                mismatches += n
                reference_losses.append(float(F.cross_entropy(expected, y, ignore_index=-100, reduction="sum")))
                item.update(reference_sha256=tensor_hash(expected), mismatched_elements=n,
                            max_absolute_error=float(difference.max()), total_elements=actual.numel())
            comparisons.append(item)
    quality = summarize_quality(losses, correct, split)
    evidence = {"windows": comparisons, "all_positions_and_vocabulary": True, "streamed_window_limit": 1,
                "total_elements": total, "mismatched_elements": mismatches, "max_absolute_error": maximum,
                "max_relative_error": relative, "rtol": 1e-4, "atol": 1e-5}
    if reference_losses:
        delta = quality["mlm_nll"]-math.fsum(reference_losses)/int((split.y != -100).sum())
        evidence.update(nll_difference=delta, passed=mismatches == 0 and abs(delta) <= 1e-5)
    if write_dir is not None:
        write_json(write_dir / "manifest.json", evidence)
    return quality, evidence


def require_parity(evidence, name):
    if not evidence.get("passed", False):
        error = ValueError(f"{name}: full-logit or MLM loss parity failed")
        error.evidence = evidence
        raise error


def build_candidate(model, layers, method, seed, calibration, recovery):
    if not method.startswith("losparse"):
        return v1.build_candidate(model, layers, method, seed, calibration, recovery)
    started = time.perf_counter()
    before = copy.deepcopy(model)
    model.requires_grad_(False)
    frozen = {name: tensor_hash(p) for name, p in model.named_parameters()
              if not any(name.startswith(f"bert.encoder.layer.{i}.intermediate.up_proj.") for i in layers)}
    spec = LoSparseSpec(64, 494, 8, warmup_steps=0 if method.endswith("immediate") else 2,
                        final_step=1 if method.endswith("immediate") else 8, delta_t=1,
                        ema_beta=.85, allow_reentry=False, initial_scale=1.)
    modules = []
    for index in layers:
        mlp = model.bert.encoder.layer[index].intermediate
        mlp.up_proj = prepare_losparse(mlp.up_proj, spec).requires_grad_(True)
        modules.append(mlp.up_proj)
    _, initial = stream_evaluate(model, calibration, other_model=before)
    require_parity(initial, "LoSparse initial")
    del before
    def objective(model, x, y):
        model.eval()
        return F.cross_entropy(v1.logits(model, x).reshape(-1, model.config.vocab_size), y.reshape(-1), ignore_index=-100)
    callback = LoSparseTrainingCallback(modules, objective)
    training = train_model(model, recovery, "language_model", epochs=1, batch_size=2,
                           learning_rate=1e-4, seed=seed, batch_objective=callback, max_updates=8)
    if training["training_steps"] != 8 or any(int(module.completed_steps) != 8 for module in modules):
        raise ValueError("LoSparse did not complete exactly eight real MLM steps")
    before_pack = copy.deepcopy(model)
    for index in layers:
        mlp = model.bert.encoder.layer[index].intermediate
        mlp.up_proj = finalize_losparse(mlp.up_proj)
    _, packed = stream_evaluate(model, calibration, other_model=before_pack)
    require_parity(packed, "LoSparse packing")
    if any(tensor_hash(dict(model.named_parameters())[name]) != value for name, value in frozen.items()):
        raise ValueError("Frozen source parameter changed during LoSparse training")
    evidence = {"spec": asdict(spec), "training": training, "callback": callback.evidence,
                "initial_logits": initial, "training_to_packed_logits": packed, "frozen_parameters_unchanged": True,
                "precision": {"residual": "float64", "factors_bias_inputs_outputs": "float32", "final": "float32"},
                "temporary_effective_weight_float64_bytes_per_layer": 1024*256*8,
                "temporary_effective_weight_float32_bytes_per_layer": 1024*256*4}
    del before_pack, modules, callback
    gc.collect()
    return evidence, time.perf_counter()-started


def code_hashes():
    # Hash the entire runtime package: solvers, specs, bundle and export code
    # are part of the execution contract just as much as this entry point.
    paths = [SCRIPT, *sorted((REPO/"fedcore").rglob("*.py"))]
    return {str(path.relative_to(REPO)): file_hash(path) for path in paths}


def verify_frozen(data):
    if data.name != "data" or data.parent.name != VERSION:
        raise ValueError("Only a v2 series data directory is admitted")
    frozen = json.loads((data.parent / "protocol.json").read_text(encoding="utf-8"))
    if frozen["version"] != VERSION or frozen["code_sha256"] != code_hashes() or frozen["protocol_sha256"] != file_hash(PROTOCOL):
        raise ValueError("Frozen v2 code/protocol changed")
    return frozen


def validate_worker_destination(data, destination, *, file=False):
    """Validate the frozen owner before creating or writing any worker output."""
    frozen = verify_frozen(data)
    series, target = data.parent.resolve(), destination.resolve()
    if not target.is_relative_to(series):
        raise ValueError("Worker destination must remain inside its frozen v2 series")
    relative = target.relative_to(series)
    if len(relative.parts) < 2 or relative.parts[0] not in ("pilot", "main", "timing"):
        raise ValueError("Worker destination must be a candidate or timing directory")
    if file:
        if target.exists():
            raise ValueError("Existing worker evidence; refusing overwrite")
    elif target.exists() and any(child.name != "worker.log" for child in target.iterdir()):
        raise ValueError("Existing candidate evidence; refusing overwrite")
    return frozen


def freeze_run(source, output):
    if output.resolve() == v1.RESULT_DEFAULT.resolve() or output.name != VERSION:
        raise ValueError("Refusing non-v2 output directory")
    if output.exists():
        raise ValueError("Existing series evidence; no implicit overwrite or timer reset")
    free_disk = shutil.disk_usage(output.parent if output.parent.exists() else output.anchor).free
    ram = psutil.virtual_memory()
    if free_disk < 60*1024**3 or ram.available < 4*1024**3:
        raise MemoryError("Require 60 GiB free disk and 4 GiB available RAM before preparation; family RSS ceiling is 15 GiB")
    output.mkdir(parents=True)
    write_json(output/"series_budget.json", {"started_unix": time.time(), "limit_seconds": 7200,
                                          "free_disk_bytes": free_disk, "available_ram_bytes": ram.available})
    sources = v1.verify_sources(source)
    data = output / "data"
    data.mkdir()
    prepare_data(source, data)
    frozen = {"version": VERSION, "bundle_schema_version": v1.VERSION, "protocol_sha256": file_hash(PROTOCOL),
              "sources": sources, "code_sha256": code_hashes(), "role_plan": ROLE_PLAN, "mask_seeds": ROLE_SEEDS,
              "source_manifest_sha256": file_hash(source/"manifest.json"),
              "seeds": [17, 29], "budget_per_ffn": v1.BUDGET, "quality_delta_limit": .10,
              "threads": 4, "interop_threads": 1, "total_seconds_limit": 7200, "candidate_seconds_limit": 600,
              "rss_limit_bytes": 15*1024**3, "data_manifest_sha256": file_hash(data/"data_manifest.json"),
              "articles_manifest_sha256": file_hash(data/"articles.json"),
              "model_revision": "387825ce42dbb39b87911cdf8e383ee3b25184f8",
              "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3"}
    write_json(output/"protocol.json", frozen)
    write_json(output/"environment.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "transformers": __import__("transformers").__version__,
        "pyarrow": __import__("pyarrow").__version__, "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(), "cpu_frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "ram": ram._asdict(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "code_sha256": code_hashes(), "protocol_json_sha256": file_hash(output/"protocol.json")})
    for role in ROLE_PLAN:
        load_role(data, role)
    return data


def reload_worker(bundle, data, destination, stage):
    validate_worker_destination(data, destination, file=True)
    if not bundle.resolve().is_relative_to(data.parent.resolve()):
        raise ValueError("Bundle must belong to the active v2 series")
    started = time.perf_counter()
    # load_bundle constructs from saved config only, never calls load_source or
    # from_pretrained, and validates strict admitted surrounding state/aliases.
    model = v1.load_bundle(bundle)
    result = {"load_seconds": time.perf_counter()-started, "source_factory_used": False,
              "source_weights_used": False, "model_tensor_bytes": tensor_state_bytes(model),
              "runtime_whole_model_tensor_bytes": v1.runtime_tensor_bytes(model), "parity": {}}
    for role in (("validation", "test") if stage == "main" else ("validation",)):
        split = load_role(data, role, 16 if stage == "pilot" else None)
        quality, parity = stream_evaluate(model, split, reference_dir=bundle.parent/f"expected_{role}")
        result[role] = quality
        result["parity"][role] = parity
        require_parity(parity, "Fresh-process transfer")
    if stage == "pilot":
        result["timing"] = v1.time_full_model(model, split.x[:1], 5)
    write_json(destination, result)


def candidate_worker(source, destination, data, method, seed, stage):
    frozen = validate_worker_destination(data, destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = {"version": VERSION, "method": method, "seed": seed, "stage": stage, "status": "running"}
    def progress(step):
        result["last_step"] = step
        write_json(destination/"progress.json", result)
    try:
        result["protocol_json_sha256"] = file_hash(data.parent/"protocol.json")
        result["code_sha256"] = frozen["code_sha256"]
        progress("source_verification")
        if file_hash(source/"manifest.json") != frozen["source_manifest_sha256"] or v1.verify_sources(source) != frozen["sources"]:
            raise ValueError("Source manifest changed after freeze")
        model, source_evidence = v1.load_source(source)
        layers = (1,) if stage == "pilot" else (0, 1, 2, 3)
        split = load_role(data, "validation", 16 if stage == "pilot" else None)
        original = copy.deepcopy(model)
        v1.adapt_bert_ffns(model, layers)
        dense_quality, adapter = stream_evaluate(model, split, other_model=original)
        result["initial_gate"] = {"adapter": adapter}
        require_parity(adapter, "Dense adapter")
        if dense_quality["mlm_nll"] >= math.log(30522)-.5:
            raise ValueError("Pretrained dense MLM quality gate failed")
        del original
        result.update(source=source_evidence, dense_validation=dense_quality, adapter_gate=True,
                      changed_layers=list(layers), dense_model_tensor_bytes=tensor_state_bytes(model))
        cal, recovery = load_role(data, "calibration"), load_role(data, "recovery")
        if method == "dense":
            mlp = model.bert.encoder.layer[layers[0]].intermediate
            rows = v1.collect_inputs(model, cal, (layers[0],))[layers[0]]
            exact, _ = v1.partition_mlp(mlp, tuple(range(128)))
            torch.testing.assert_close(exact(rows), mlp(rows), rtol=1e-5, atol=1e-6)
            packed = v1.PackedNMLinear(256, 1024, n=2, m=4, rank=0, bias=True)
            negative = tensor_state_bytes(packed)+tensor_state_bytes(mlp.down_proj)
            result["controls"] = {"hassle_2_4_ffn_bytes": negative, "hassle_2_4_storage_increase": negative > 2_102_272,
                                  "sola_exact_partition_parity": True, "sola_partition_windows": len(cal.x)}
            del exact, packed, rows
        progress("candidate_build")
        evidence, seconds = build_candidate(model, layers, method, seed, cal, recovery)
        result.update(method_evidence=evidence, build_seconds=seconds)
        if method.startswith("losparse"):
            result["initial_gate"].update(losparse=evidence["initial_logits"], packing=evidence["training_to_packed_logits"])
        result["initial_gate"]["passed"] = True
        ffn_bytes = {str(i): tensor_state_bytes(model.bert.encoder.layer[i].intermediate) for i in layers}
        result["ffn_tensor_bytes"] = ffn_bytes
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Final model has non-float32 parameters")
        if method != "dense" and any(value > v1.BUDGET for value in ffn_bytes.values()):
            raise ValueError("Actual FFN bytes exceed frozen budget")
        result.update(model_tensor_bytes=tensor_state_bytes(model), runtime_whole_model_tensor_bytes=v1.runtime_tensor_bytes(model),
                      budget_match=method == "dense" or all(abs(v1.BUDGET-value) <= .01*v1.BUDGET for value in ffn_bytes.values()))
        progress("bundle_save")
        bundle = destination/"bundle"
        result["bundle"] = v1.save_bundle(model, bundle, layers, source)
        write_json(destination/"series_specification.json", {"version": VERSION, "bundle_schema": v1.VERSION,
            "method": method, "seed": seed, "layers": layers, "method_evidence": evidence,
            "protocol_json_sha256": result["protocol_json_sha256"], "bundle_manifest_sha256": file_hash(bundle/"files.json")})
        result["frozen_specification_sha256"] = file_hash(destination/"series_specification.json")
        rows = v1.collect_inputs(model, cal, layers)
        exports = {str(i): measure_artifact(model.bert.encoder.layer[i].intermediate, rows[i][:64],
                     destination/f"ffn_{i}.ts", repeats=5 if stage == "pilot" else 10, warmup=3, threads=4) for i in layers}
        result["ffn_artifacts"] = exports
        if any(item["status"] != "succeeded" for item in exports.values()):
            raise ValueError("FFN TorchScript export/reload failed")
        progress("immutable_specification_saved_quality_evaluation")
        roles = ("validation", "test") if stage == "main" else ("validation",)
        for role in roles:
            selected_split = split if role == "validation" else load_role(data, "test")
            quality, written = stream_evaluate(model, selected_split, write_dir=destination/f"expected_{role}")
            result[role] = quality
            result.setdefault("reference_logits", {})[role] = {"manifest_sha256": file_hash(destination/f"expected_{role}/manifest.json"),
                "bytes": sum(p.stat().st_size for p in (destination/f"expected_{role}").iterdir()), "windows": len(written["windows"])}
        result["validation_delta"] = result["validation"]["mlm_nll"]-dense_quality["mlm_nll"]
        del model, rows
        gc.collect()
        progress("fresh_process_reload")
        command = [sys.executable, str(SCRIPT), "--stage", "reload", "--bundle", str(bundle), "--data", str(data),
                   "--output", str(destination/"reload.json"), "--series-stage", stage]
        with (destination/"reload.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if completed.returncode:
            raise ValueError("Fresh-process bundle reload failed")
        reload = json.loads((destination/"reload.json").read_text(encoding="utf-8"))
        result.update(reload=reload, transfer_gate=True, status="succeeded")
    except Exception as error:
        result.update(status="failed", reason=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        if hasattr(error, "evidence"):
            result["failure_parity"] = error.evidence
    write_json(destination/"result.json", result)
    return result


def paired_quality(row, dense):
    for role in ("validation", "test"):
        if role not in row or role not in dense:
            continue
        delta = row[role]["mlm_nll"]-dense[role]["mlm_nll"]
        row[f"{role}_delta"] = delta
        values = {unit: value-dense[role]["article_nll"][unit] for unit, value in row[role]["article_nll"].items()}
        differences = np.array(list(values.values()))
        generator = np.random.default_rng(223)
        samples = [float(generator.choice(differences, len(differences), replace=True).mean()) for _ in range(2000)]
        row[f"paired_{role}_article_delta"] = values
        row[f"paired_{role}_article_bootstrap_95"] = np.percentile(samples, [2.5, 97.5]).tolist()
    if "test_delta" in row:
        row["quality_gate"] = {"passed": row["test_delta"] <= .10, "test_delta_limit": .10, "test_delta": row["test_delta"]}
        row["useful_storage"] = bool(row["quality_gate"]["passed"] and row.get("transfer_gate") and row.get("budget_match")
                                      and row["model_tensor_bytes"] < dense["model_tensor_bytes"])


def timing_worker(bundle, data, destination):
    validate_worker_destination(data, destination)
    if not bundle.resolve().is_relative_to(data.parent.resolve()):
        raise ValueError("Bundle must belong to the active v2 series")
    started = time.perf_counter()
    model = v1.load_bundle(bundle)
    load_seconds = time.perf_counter()-started
    split = load_role(data, "validation")
    result = {"status": "succeeded", "timing": v1.time_full_model(model, split.x[:1], 10), "load_seconds": load_seconds,
              "quality_evaluated": False, "source_factory_used": False, "bundle_manifest_sha256": file_hash(bundle/"files.json")}
    # Every loaded FFN is measured again in the same final shuffled pass.
    layers = json.loads((bundle/"bundle.json").read_text())["layers"]
    example = torch.linspace(-2, 2, 64*256).reshape(64, 256)
    ffn = {}
    with torch.inference_mode():
        for i in layers:
            module = model.bert.encoder.layer[i].intermediate
            for _ in range(3):
                module(example)
            raw = []
            for _ in range(10):
                tick = time.perf_counter_ns()
                module(example)
                raw.append((time.perf_counter_ns()-tick)/1e6)
            ffn[str(i)] = {"raw_ms": raw, "median_ms": float(np.median(raw)), "p95_ms": float(np.percentile(raw, 95)),
                           "input_shape": [64, 256], "warmup": 3, "repeats": 10, "threads": 4,
                           "format": "PyTorch eager from loaded bundle"}
    result["ffn_timing"] = ffn
    write_json(destination/"result.json", result)


def run_study(source, output):
    data = freeze_run(source, output)
    started = json.loads((output/"series_budget.json").read_text())["started_unix"]
    reports = {}
    for stage in ("pilot", "main"):
        if stage == "pilot":
            cases = [(method, 17) for method in v1.METHODS]
        else:
            admitted = {row["method"] for row in reports["pilot"]["results"]
                        if row["status"] == "succeeded" and row.get("budget_match") and row.get("transfer_gate")}
            cases = [(method, seed) for seed in (17, 29) for method in v1.METHODS if method in admitted
                     and not (seed == 29 and method in v1.DETERMINISTIC)]
            cases += [(method, seed) for seed in (17, 29) for method in ("losparse_immediate", "hassle_diagonal")
                      if method.split("_")[0] in admitted]
        np.random.default_rng(211).shuffle(cases)
        cases.insert(0, ("dense", 17))
        series = output/stage
        series.mkdir()
        report = {"version": VERSION, "stage": stage, "results": [], "planned_cases": cases,
                  "deterministic_reused_seed29": list(v1.DETERMINISTIC), "construction_order_seed": 211}
        if stage == "main":
            report["skipped_pilot_failures"] = [{"method": row["method"], "status": "skipped",
                "reason": "Pilot initial/budget/transfer gate failed", "pilot_reason": row.get("reason")}
                for row in reports["pilot"]["results"] if row["method"] != "dense" and row["method"] not in admitted]
        reports[stage] = report
        for method, seed in cases:
            if time.time()-started > 7200:
                report["stop_reason"] = "Two-hour series limit"
                break
            destination = series/f"{method}_seed{seed}"
            print(f"Starting {stage} {method} seed={seed}", flush=True)
            command = [sys.executable, str(SCRIPT), "--stage", "worker", "--series-stage", stage, "--method", method,
                       "--seed", str(seed), "--source", str(source), "--output", str(destination), "--data", str(data)]
            row = v1.run_bounded_case(command, destination, started, method, seed)
            report["results"].append(row)
            dense = next((r for r in report["results"] if r["method"] == "dense"), None)
            if dense and row["status"] == "succeeded":
                paired_quality(row, dense)
                write_json(destination/"result.json", row)
            write_json(series/"report.json", report)
            print(f"Finished {stage} {method}: {row['status']} {row.get('reason', '')}", flush=True)
            if method == "dense" and row["status"] != "succeeded":
                report["stop_reason"] = "Dense control failed"
                break
        write_json(series/"report.json", report)
        if report.get("stop_reason"):
            break
    # All variants including dense are shuffled before the final timing pass.
    if "main" in reports:
        rows = [row for row in reports["main"]["results"] if row["status"] == "succeeded"]
        np.random.default_rng(211).shuffle(rows)
        timing = {"seed": 211, "order": [[r["method"], r["seed"]] for r in rows], "results": []}
        for row in rows:
            if time.time()-started > 7200:
                timing["stop_reason"] = "Two-hour series limit"
                break
            name = f"{row['method']}_seed{row['seed']}"
            destination = output/"timing"/name
            command = [sys.executable, str(SCRIPT), "--stage", "time", "--bundle", str(output/"main"/name/"bundle"),
                       "--data", str(data), "--output", str(destination)]
            measured = v1.run_bounded_case(command, destination, started, row["method"], row["seed"])
            measured.update(method=row["method"], seed=row["seed"])
            timing["results"].append(measured)
            write_json(output/"timing/report.json", timing)
        baseline = next((r for r in timing["results"] if r["method"] == "dense" and r["status"] == "succeeded"), None)
        if baseline:
            for row in reports["main"]["results"]:
                measured = next((r for r in timing["results"] if r["method"] == row["method"] and r["seed"] == row["seed"] and r["status"] == "succeeded"), None)
                if measured is None:
                    continue
                dense_raw, raw = baseline["timing"]["raw_ms"], measured["timing"]["raw_ms"]
                generator = np.random.default_rng(211)
                ratios = [np.median(generator.choice(dense_raw, len(dense_raw)))/np.median(generator.choice(raw, len(raw))) for _ in range(2000)]
                interval = np.percentile(ratios, [2.5, 97.5]).tolist()
                speed = float(np.median(dense_raw)/np.median(raw))
                row.update(final_timing=measured, full_speed_ratio=speed, speed_ratio_bootstrap_95=interval,
                           local_acceleration_pass=bool(speed >= 1/.9 and interval[0] > 1))
                write_json(output/f"main/{row['method']}_seed{row['seed']}/result.json", row)
        write_json(output/"main/report.json", reports["main"])
    write_json(output/"series_summary.json", {"version": VERSION, "elapsed_seconds": time.time()-started,
                                              "stages": list(reports), "finished_unix": time.time()})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "worker", "reload", "time"), required=True)
    parser.add_argument("--source", type=Path, default=v1.SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=RESULT_DEFAULT)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--method", choices=(*v1.METHODS, "dense", "losparse_immediate", "hassle_diagonal"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--series-stage", choices=("pilot", "main"), default="pilot")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.stage == "run":
        run_study(args.source, args.output)
    elif args.stage == "worker":
        candidate_worker(args.source, args.output, args.data, args.method, args.seed, args.series_stage)
    elif args.stage == "reload":
        reload_worker(args.bundle, args.data, args.output, args.series_stage)
    else:
        timing_worker(args.bundle, args.data, args.output)
