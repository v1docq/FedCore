"""v2 boundaries: grouping, independent windows, streamed checks and packing."""
from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM

from fedcore.algorithm.pruning.sparse.linear import prepare_losparse, finalize_losparse, LoSparseTrainingCallback
from fedcore.algorithm.pruning.sparse.specs import LoSparseSpec
from fedcore.experiments.protocol import TensorSplit, tensor_hash
from fedcore.experiments.runner import train_model
from fedcore.experiments import sparse_pretrained_v2 as v2


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("rows", [0, 3, 6])
@pytest.mark.parametrize("scale", [.3, 2.])
@pytest.mark.parametrize("bias", [False, True])
def test_finalized_operator_matches_completed_training_exactly(rank, rows, scale, bias):
    torch.manual_seed(97)
    layer = nn.Linear(8, 6, bias=bias)
    module = prepare_losparse(layer, LoSparseSpec(rank, rows, 8, initial_scale=scale))
    assert module.residual.dtype == torch.float64
    assert module.output_dtype == torch.float32
    with torch.no_grad():
        module.support.zero_()
        module.support[:rows] = True
        module.completed_steps.fill_(8)
        module.residual.add_(.0000123456789)
    x = torch.randn(4, 8)
    expected = module(x)
    final = finalize_losparse(module)
    assert all(value.dtype == torch.float32 for value in final.parameters())
    assert final.residual_count == rows
    assert not any("ema" in name or name == "residual" for name in final.state_dict())
    torch.testing.assert_close(final(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [.3, 2.])
@pytest.mark.parametrize("rank", [0, 2])
def test_initial_full_support_recovers_live_source_weight(scale, rank):
    torch.manual_seed(11)
    layer = nn.Linear(8, 6)
    module = prepare_losparse(layer, LoSparseSpec(rank, 3, 8, initial_scale=scale))
    x = torch.randn(100, 8)
    torch.testing.assert_close(module(x), layer(x), rtol=0, atol=0)
    assert set(module.state_dict()) == ({"residual", "bias", "support", "importance_ema", "completed_steps"}
                                      | ({"factors.left.weight", "factors.right.weight"} if rank else set()))


def test_eight_real_mlm_steps_freeze_and_pack():
    torch.manual_seed(17)
    torch.set_num_threads(4)
    config = BertConfig(vocab_size=31, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
                        num_attention_heads=2, max_position_embeddings=16, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    model = BertForMaskedLM(config).float().eval().requires_grad_(False)
    frozen = {name: tensor_hash(parameter) for name, parameter in model.named_parameters()
              if not name.startswith("bert.encoder.layer.0.intermediate.dense.")}
    module = prepare_losparse(model.bert.encoder.layer[0].intermediate.dense,
                             LoSparseSpec(3, 6, 8, warmup_steps=2, final_step=8)).requires_grad_(True)
    model.bert.encoder.layer[0].intermediate.dense = module
    x = torch.randint(1, 31, (16, 8))
    y = torch.full_like(x, -100)
    y[:, 2:4] = x[:, 2:4]
    x[:, 2:4] = 0
    split = TensorSplit(x, y, tuple(map(str, range(16))), tuple(map(str, range(16))))
    def loss(model, x, y):
        model.eval()
        return nn.functional.cross_entropy(model(x).logits.reshape(-1, 31), y.reshape(-1), ignore_index=-100)
    callback = LoSparseTrainingCallback(module, loss)
    evidence = train_model(model, split, "language_model", epochs=1, batch_size=2,
                           learning_rate=1e-4, seed=17, batch_objective=callback, max_updates=8)
    assert evidence["training_steps"] == 8
    assert int(module.completed_steps) == 8 and int(module.support.sum()) == 6
    assert evidence["optimizer_state_bytes"] > module.residual.numel()*16
    with torch.inference_mode():
        expected = model(x).logits
        model.bert.encoder.layer[0].intermediate.dense = finalize_losparse(module)
        torch.testing.assert_close(model(x).logits, expected, rtol=0, atol=0)
    assert all(tensor_hash(dict(model.named_parameters())[name]) == value for name, value in frozen.items())


def test_full_bert_ffn_final_bytes_include_down_and_indices():
    mlp = v2.v1.ReferenceMLP(256, 1024, gated=False, activation="gelu", bias=True)
    training = prepare_losparse(mlp.up_proj, LoSparseSpec(64, 494, 8))
    assert training.residual.numel()*training.residual.element_size() == 2_097_152
    assert training.importance_ema.numel()*training.importance_ema.element_size() == 1_048_576
    with torch.no_grad():
        training.support.zero_()
        training.support[:494] = True
        training.completed_steps.fill_(8)
    mlp.up_proj = finalize_losparse(training)
    assert v2.tensor_state_bytes(mlp) == 1_891_184
    assert all(p.dtype == torch.float32 for p in mlp.parameters())


def article(identifier, title, text, split="train"):
    return v2.v1.Article(f"{identifier:064x}", title, text, v2.digest(text), split)


def test_transitive_components_cannot_leave_an_alias_in_train():
    articles = [article(2, "x", "first"), article(4, "x", "second"), article(9, "y", "second", "test")]
    records = v2.plan_article_groups(articles)
    assert {r["owner_split"] for r in records} == {"test"}
    assert {r["role"] for r in records} == {"test"}
    assert {r["canonical_id"] for r in records} == {f"{2:064x}"}
    assert [r["eligible"] for r in records] == [False, False, True]


def test_group_low_bit_uses_whole_component_and_permutation_is_stable():
    articles = [article(5, "x", "first"), article(2, "x", "second"), article(8, "z", "third")]
    records = v2.plan_article_groups(articles)
    expected = {r["article_id"]: r for r in records}
    assert records[0]["role"] == "calibration"
    assert {r["article_id"]: r for r in v2.plan_article_groups(articles[::-1])} == expected
    assert sum(r["eligible"] for r in records) == 2


def test_hash_selection_round_robin_and_source_permutation():
    articles = [article(i*2, f"title {i}", str(i), "validation") for i in range(1, 40)]
    def selection(items):
        return v2.select_role_windows(items, v2.plan_article_groups(items), "validation",
                                      lambda text: list(range(62*9)), 101, 102, 32, 2)
    windows, ids, units, descriptions = selection(articles)
    assert selection(articles[::-1]) == (windows, ids, units, descriptions)
    assert len(windows) == 64 and len(set(units[:16])) == 16
    assert units[:32] == units[32:] and len(set(units)) == 32
    assert any(d["start"] > 62 for d in descriptions)
    assert all(descriptions[i]["start"] < descriptions[i+32]["start"] for i in range(32))
    with pytest.raises(ValueError, match="Insufficient independent"):
        v2.select_role_windows(articles[:10], v2.plan_article_groups(articles[:10]), "validation",
                               lambda text: list(range(62*9)), 101, 102, 32, 2)


def test_freeze_refuses_v1_and_existing_series(tmp_path):
    with pytest.raises(ValueError, match="non-v2"):
        v2.freeze_run(tmp_path, v2.v1.RESULT_DEFAULT)
    output = tmp_path/v2.VERSION
    output.mkdir()
    marker = output/"original.json"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="Existing series"):
        v2.freeze_run(tmp_path, output)
    assert marker.read_text() == "keep"


def test_workers_reject_wrong_series_and_existing_evidence_before_writing(tmp_path):
    series = tmp_path/v2.VERSION
    data = series/"data"
    data.mkdir(parents=True)
    v2.write_json(series/"protocol.json", {"version": v2.VERSION, "code_sha256": v2.code_hashes(),
                                          "protocol_sha256": v2.file_hash(v2.PROTOCOL)})
    forbidden = tmp_path/"local-bert-mlm-v1"/"main"/"candidate"
    with pytest.raises(ValueError, match="inside its frozen v2 series"):
        v2.candidate_worker(tmp_path, forbidden, data, "dense", 17, "main")
    assert not forbidden.exists()
    valid = series/"pilot"/"dense_seed17"
    valid.mkdir(parents=True)
    original = valid/"result.json"
    original.write_text("keep")
    with pytest.raises(ValueError, match="Existing candidate evidence"):
        v2.candidate_worker(tmp_path, valid, data, "dense", 17, "pilot")
    assert original.read_text() == "keep"
    v2.write_json(series/"protocol.json", {"version": "local-bert-mlm-v1"})
    with pytest.raises(ValueError):
        v2.reload_worker(valid/"bundle", data, forbidden/"reload.json", "main")
    assert not forbidden.exists()


def test_stream_checks_unmasked_positions_and_full_vocabulary(tmp_path, monkeypatch):
    x = torch.zeros(2, 64, dtype=torch.long)
    y = torch.full_like(x, -100)
    y[:, 1:10] = 1
    split = TensorSplit(x, y, ("a", "b"), ("a", "b"))
    base = nn.Linear(1, 1)
    def scores(model, x):
        value = torch.zeros(1, 64, 37)
        if model is not base:
            value[0, 63, 36] = 1
        return value
    monkeypatch.setattr(v2.v1, "logits", scores)
    _, written = v2.stream_evaluate(base, split, write_dir=tmp_path/"reference")
    _, parity = v2.stream_evaluate(nn.Linear(1, 1), split, reference_dir=tmp_path/"reference")
    assert parity["total_elements"] == 2*64*37
    assert parity["mismatched_elements"] == 2 and not parity["passed"]
    assert parity["nll_difference"] == 0
    assert len(written["windows"]) == 2
