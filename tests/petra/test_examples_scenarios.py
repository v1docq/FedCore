"""Independent references for sample isolation, fitted statistics and causal LM."""
import importlib.util
import json
import math
from pathlib import Path
import zipfile

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import torch

from fedcore.experiments.scenarios import (
    TrainPreprocessor, build_tabular, build_cv_digits, build_language_model,
    build_forecasting, build_financial, causal_token_nll, chronological_windows,
    client_split_indices, read_ts_regression, split_indices,
)


@given(st.integers(min_value=0, max_value=10000))
@settings(max_examples=12, deadline=None)
def test_partition_conserves_samples_is_disjoint_and_seeded(seed):
    labels = np.tile(np.arange(4), 40)
    first, second = split_indices(labels, seed), split_indices(labels, seed)
    seen = set()
    for role, indices in first.items():
        assert np.array_equal(indices, second[role])
        assert not seen.intersection(indices)
        seen.update(indices)
        assert set(labels[indices]) == {0, 1, 2, 3}
    assert seen == set(range(len(labels)))


def test_tabular_test_shift_cannot_change_fitted_statistics_or_baseline_state():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(160, 5))
    x[3, 1] = np.nan
    y = np.tile(np.arange(2, dtype=np.int64), 80)
    indices = split_indices(y, 11)
    shifted = x.copy()
    shifted[indices["test"]] += 10000
    original, changed = build_tabular(11, features=x, labels=y), build_tabular(11, features=shifted, labels=y)
    assert original.metadata["preprocessing"] == changed.metadata["preprocessing"]
    assert original.metadata["preprocessing_fit_ids"] == original.train.ids
    assert torch.equal(original.train.x, changed.train.x)
    assert not torch.equal(original.test.x, changed.test.x)
    assert all(torch.equal(value, changed.original_model.state_dict()[name]) for name, value in original.original_model.state_dict().items())


def test_preprocessor_missingness_matches_independent_reference():
    train = np.array([[1., np.nan, 2.], [3., 2., 2.], [9., 8., 2.]])
    pp = TrainPreprocessor.fit(train)
    filled = np.array([[1., 5., 2.], [3., 2., 2.], [9., 8., 2.]])
    np.testing.assert_allclose(pp.median, [3., 5., 2.])
    np.testing.assert_allclose(pp.mean, filled.mean(0))
    np.testing.assert_allclose(pp.transform(train)[:, 2], 0)
    with pytest.raises(ValueError, match="completely missing"):
        TrainPreprocessor.fit(np.full((3, 2), np.nan))


@given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=6))
def test_forecasting_roles_have_no_shared_input_or_target_row(context, horizon):
    series = np.arange(240, dtype=np.float32)
    windows = chronological_windows(series, context=context, horizon=horizon)
    seen = set()
    for role, (x, y, intervals) in windows.items():
        role_rows = set()
        for index, (begin, end) in enumerate(intervals):
            np.testing.assert_array_equal(x[index, 0], series[begin:begin + context])
            np.testing.assert_array_equal(y[index], series[begin + context:end])
            role_rows.update(range(begin, end))
        assert not role_rows.intersection(seen)
        seen.update(role_rows)
    assert len(seen) == len(series)


def test_forecasting_rejects_bad_horizon_and_requires_actual_future_csv(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        chronological_windows(np.arange(30), context=2, horizon=0)
    with pytest.raises(ValueError, match="too short"):
        chronological_windows(np.arange(30), context=20, horizon=2)
    path = tmp_path / "data.csv"
    path.write_text("date,Appliances\n2020-01-01,1\n2020-01-01,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamps"):
        build_forecasting(path)


def test_timestamped_ts_preserves_shapes_and_timestamp_span(tmp_path):
    source = "@timestamps true\n@targetlabel true\n@data\n(2020-01-01 00:00:00,1),(2020-01-01 00:10:00,2):(2020-01-01 00:00:00,3),(2020-01-01 00:10:00,4):9\n"
    path = tmp_path / "series.ts"
    path.write_text(source, encoding="utf-8")
    x, y, intervals = read_ts_regression(path, return_intervals=True)
    np.testing.assert_array_equal(x, [[[1, 2], [3, 4]]])
    np.testing.assert_array_equal(y, [[9]])
    assert intervals[0][1] - intervals[0][0] == 600
    path.write_text(source.replace("00:10:00,4", "00:20:00,4"), encoding="utf-8")
    with pytest.raises(ValueError, match="numeric"):
        read_ts_regression(path)


@given(st.integers(min_value=0, max_value=500))
@settings(max_examples=10, deadline=None)
def test_client_isolation_conserves_all_events(seed):
    clients = [f"client-{i // 3}" for i in range(90)]
    roles = client_split_indices(clients, seed)
    seen, total = set(), set()
    for indices in roles.values():
        units = {clients[i] for i in indices}
        assert not units.intersection(seen)
        seen.update(units)
        total.update(indices)
    assert total == set(range(len(clients)))


def test_financial_refuses_unverified_access(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source": "closed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized-access"):
        build_financial(tmp_path / "missing.npz", manifest)


def test_causal_nll_matches_scalar_reference_shift_and_padding():
    logits = torch.tensor([[[2., 0., -1.], [0., 2., -1.], [-2., 0., 2.], [10., -10., 0.]]], dtype=torch.float64)
    labels = torch.tensor([[0, 1, 2, -100]])
    actual = causal_token_nll(logits, labels)
    expected = sum(math.log(sum(math.exp(v) for v in row)) - row[target] for row, target in zip(logits[0, :2].tolist(), [1, 2])) / 2
    assert actual["token_count"] == 2
    assert actual["nll"] == pytest.approx(expected, abs=1e-12)
    assert actual["perplexity"] == pytest.approx(math.exp(expected))
    unshifted = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100)
    assert abs(actual["nll"] - float(unshifted)) > .5
    with pytest.raises(ValueError, match="No valid"):
        causal_token_nll(logits, torch.full((1, 4), -100))


def test_language_documents_isolated_before_tokenization_and_causal_model(tmp_path):
    path = tmp_path / "corpus.json"
    corpus = {"source": "test-fixture", "revision": "1", "license": "test-only",
              "documents": [{"id": f"doc-{i}", "text": f"Document number {i} has distinct words."} for i in range(24)]}
    path.write_text(json.dumps(corpus), encoding="utf-8")
    bundle = build_language_model(path, sequence_length=16)
    seen = set()
    for role in ("train", "validation", "calibration", "test"):
        split = getattr(bundle, role)
        assert not seen.intersection(split.unit_ids)
        seen.update(split.unit_ids)
    model = bundle.original_model.eval()
    tokens = bundle.train.x[:1]
    altered = tokens.clone()
    altered[:, -3:] = 33
    assert torch.allclose(model(tokens)[:, :-3], model(altered)[:, :-3])
    corpus["documents"][-1]["text"] = corpus["documents"][0]["text"]
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate documents"):
        build_language_model(path)


def test_real_cv_model_shapes_and_builder_rng_isolation():
    torch.manual_seed(31)
    before = torch.get_rng_state()
    bundle = build_cv_digits(23)
    assert torch.equal(before, torch.get_rng_state())
    assert bundle.original_model(bundle.train.x[:3]).shape == (3, 10)


def test_scanner_detects_tokens_in_outputs_and_compressed_zip_without_value(tmp_path, capsys):
    scanner_path = Path(__file__).resolve().parents[2] / "scripts" / "check_example_secrets.py"
    spec = importlib.util.spec_from_file_location("example_secret_scanner", scanner_path)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    secret = "gh" + "p_" + "x" * 36
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text(json.dumps({"outputs": [{"text": secret}]}), encoding="utf-8")
    archive_path = tmp_path / "copied.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.ipynb", json.dumps({"text": secret}))
    url = tmp_path / "clone.txt"
    url.write_text("https://user:password@example.org/repository.git", encoding="utf-8")
    escaped = tmp_path / "escaped_output.ipynb"
    escaped.write_text(json.dumps({"text": secret}).replace("gh", "\\u0067\\u0068"), encoding="utf-8")
    assert scanner.main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "output.ipynb" in output and "notebook.ipynb" in output and "clone.txt" in output and "escaped_output.ipynb" in output
    assert secret not in output and "password" not in output


def test_original_inventory_complete_with_explicit_status_and_consolidation():
    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "examples/petra/inventory.json").read_text(encoding="utf-8"))
    assert registry["audit_file_count"] == len(registry["files"]) == 75
    paths = {entry["path"] for entry in registry["files"]}
    assert len(paths) == 75
    for entry in registry["files"]:
        assert entry["status"] in {"supported_example", "educational_demonstration", "historical_artifact", "experimental_prototype"}
        if not (root / entry["path"]).exists():
            assert entry.get("consolidated_into")
        if entry["path"].endswith((".pt", ".pth")):
            assert entry["status"] == "historical_artifact"
            assert len(entry["current_sha256"]) == 64


def test_published_zip_exactly_matches_current_sources_and_checksums():
    import hashlib
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(root / "examples/fedcore_examples_export_onnx_and_docker.zip") as archive:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        assert set(archive.namelist()) == set(manifest["source_files"]) | {"PACKAGE_MANIFEST.json"}
        assert manifest["contains_measured_results"] is False
        for filename, expected in manifest["source_files"].items():
            data = archive.read(filename)
            assert data == (root / filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected
