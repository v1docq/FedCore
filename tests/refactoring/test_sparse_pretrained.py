"""Invariants for the real pretrained adapter, data roles and portable bundle."""
import json
from pathlib import Path

import pytest
import torch
pytest.importorskip("transformers", reason="BERT experiments require the llm extra")
from transformers import BertConfig, BertForMaskedLM

from fedcore.experiments.sparse_pretrained import (
    Article, adapt_bert_ffns, assign_articles, mask_windows, parse_articles,
    load_bundle, load_source, logits, save_bundle, SOURCE_DEFAULT, file_hash,
    InitialParityError,
    runtime_tensor_bytes,
)
from fedcore.experiments.measurement import tensor_state_bytes


def test_article_heading_and_duplicate_role_invariants():
    articles = parse_articles(["lead\n", " = Title = \n", "a\n", " == Section == \n", "b\n",
                               " = Other = \n", "c\n"], "train")
    assert len(articles) == 3
    assert "Section" in articles[1].text
    duplicate = Article(articles[1].article_id, articles[1].title, "different", "new_hash", "test")
    assigned = assign_articles([*articles, duplicate])
    assert assigned[1][1] == "excluded_overlap"
    assert assigned[-1][1] == "test"


@pytest.mark.parametrize("seed", [101, 103, 107, 109])
def test_mask_exact_count_specials_and_determinism(seed):
    original = torch.arange(64).repeat(5, 1)
    x, y = mask_windows(original, 103, seed)
    again_x, again_y = mask_windows(original, 103, seed)
    assert torch.equal(x, again_x) and torch.equal(y, again_y)
    assert ((y != -100).sum(1) == 9).all()
    assert torch.equal(x[:, [0, 63]], original[:, [0, 63]])
    assert (y[:, [0, 63]] == -100).all()
    assert torch.equal(y[y != -100], original[y != -100])


def test_adapter_exact_logits_and_refusal():
    torch.set_num_threads(4)
    config = BertConfig(hidden_size=256, intermediate_size=1024, num_hidden_layers=4,
                        num_attention_heads=4, vocab_size=30522, hidden_act="gelu")
    model = BertForMaskedLM(config).eval()
    x = torch.randint(0, 30522, (1, 64))
    with torch.inference_mode():
        expected = logits(model, x)
        adapt_bert_ffns(model, (0, 1, 2, 3))
        actual = logits(model, x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    config.hidden_act = "relu"
    with pytest.raises(ValueError, match="configuration"):
        adapt_bert_ffns(model, (0,))


def test_initial_parity_failure_is_not_a_trained_quality_result():
    expected = torch.tensor([0., 1.])
    actual = torch.tensor([2e-5, 1.])
    error = InitialParityError(actual, expected)
    assert error.evidence["training_steps"] == 0
    assert error.evidence["mismatched_elements"] == 1
    assert error.evidence["total_elements"] == 2
    assert error.evidence["atol"] == 1e-5
    json.dumps(error.evidence, allow_nan=False)


def test_runtime_storage_includes_nonpersistent_buffers_and_counts_aliases_once():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, 3))
    model.alias = model.weight
    model.register_buffer("persisted", torch.arange(4))
    model.register_buffer("same_buffer", model.persisted, persistent=False)
    model.register_buffer("temporary", torch.ones(5), persistent=False)
    assert tensor_state_bytes(model) == 24 + 32
    assert runtime_tensor_bytes(model) == 24 + 32 + 20


@pytest.mark.skipif(not (SOURCE_DEFAULT / "model/pytorch_model.bin").exists(), reason="Pinned local source unavailable")
def test_real_head_bundle_without_source_and_stale_keys(tmp_path):
    model, evidence = load_source(SOURCE_DEFAULT)
    assert evidence["unique_parameters"] == 11_201_594
    adapt_bert_ffns(model, (1,))
    x = torch.arange(64)[None]+200
    with torch.inference_mode():
        expected = logits(model, x)
    save_bundle(model, tmp_path, (1,), SOURCE_DEFAULT)
    assert not list(tmp_path.rglob("pytorch_model.bin"))
    loaded = load_bundle(tmp_path)
    with torch.inference_mode():
        torch.testing.assert_close(logits(loaded, x), expected, rtol=1e-4, atol=1e-5)
    assert loaded.cls.predictions.decoder.weight is loaded.bert.embeddings.word_embeddings.weight
    state = torch.load(tmp_path / "surrounding.pt", weights_only=True)
    assert not any(name.startswith("bert.encoder.layer.1.intermediate.") for name in state)
    assert not any(name.startswith("bert.encoder.layer.1.output.dense.") for name in state)
    config = json.loads((tmp_path / "config.json").read_text())
    config["hidden_size"] = 100_000
    (tmp_path / "config.json").write_text(json.dumps(config))
    files = json.loads((tmp_path / "files.json").read_text())
    files["config.json"]["sha256"] = file_hash(tmp_path / "config.json")
    (tmp_path / "files.json").write_text(json.dumps(files))
    with pytest.raises(ValueError, match="configuration"):
        load_bundle(tmp_path)
