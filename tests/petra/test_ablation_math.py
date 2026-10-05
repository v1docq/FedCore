from copy import deepcopy
import json
import pytest
import torch
from torch import nn

from fedcore.experiments.math_checks import (activation_rows, approximate_layer, factor_diagnostics,
    orthogonal_penalty, relative_error, second_moment, truncated_svd, weighted_svd)
from fedcore.experiments.ablations import (RegularizationVariant, compare_validation_timing,
    run_rank_ablation, run_regularization_ablation, run_training_cost_controls, run_order_ablation, run_validity_ablation)
from fedcore.experiments.protocol import ExperimentBundle, ExperimentProtocol, TensorSplit, CandidateSpec


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDCORE_MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    from fedcore.tools.registry.model_registry import ModelRegistry
    ModelRegistry._instance = None
    ModelRegistry._initialized = False
    torch.manual_seed(27)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    splits = []
    for role in ("train", "validation", "calibration", "test"):
        x = torch.randn(8, 4)
        y = torch.arange(8) % 3
        splits.append(TensorSplit(x, y, tuple(f"{role}-{i}" for i in range(8))))
    return ExperimentBundle(*splits, "classification", model)


def test_weighted_isotropic_full_rank_and_input_moment_reference():
    w = torch.tensor([[3., 0., 0.], [0., 2., 0.]], dtype=torch.float64)
    isotropic = 7 * torch.eye(3, dtype=w.dtype)
    torch.testing.assert_close(weighted_svd(w, isotropic, 1)["approximation"], truncated_svd(w, 1), rtol=1e-12, atol=1e-12)
    singular = torch.diag(torch.tensor([1., 0., 0.], dtype=w.dtype))
    torch.testing.assert_close(weighted_svd(w, singular, 2)["approximation"], w)
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=w.dtype)
    torch.testing.assert_close(second_moment(x), sum(row[:, None] @ row[None, :] for row in x) / 2)
    assert not torch.equal(second_moment(x), torch.cov(x.T))


def test_small_weight_direction_can_be_important_and_distribution_shift_reverses():
    w = torch.diag(torch.tensor([10., 1.], dtype=torch.float64))
    moment = torch.diag(torch.tensor([1., 1000.], dtype=w.dtype))
    ordinary = truncated_svd(w, 1)
    weighted = weighted_svd(w, moment, 1)["approximation"]
    assert float(((w - weighted) @ torch.diag(torch.tensor([1., 1000. ** .5], dtype=w.dtype))).square().sum()) < 101.
    assert float(((w - ordinary) @ torch.diag(torch.tensor([1., 1000. ** .5], dtype=w.dtype))).square().sum()) == pytest.approx(1000.)
    # Independent shifted inputs invalidate a universal improvement claim.
    shifted = torch.tensor([[20., 0.]], dtype=w.dtype)
    assert relative_error(shifted @ w.T, shifted @ ordinary.T)["relative"] == 0.
    assert relative_error(shifted @ w.T, shifted @ weighted.T)["relative"] == 1.


def test_psd_conditioning_zero_denominator_and_invalid_moments():
    w = torch.eye(2, dtype=torch.float64)
    result = weighted_svd(w, torch.diag(torch.tensor([1., 1e-12], dtype=w.dtype)), 1)
    assert result["condition_number"] == pytest.approx(1e12)
    assert relative_error(torch.zeros(2), torch.zeros(2))["status"] == "zero_reference_exact"
    assert relative_error(torch.zeros(2), torch.ones(2))["relative"] is None
    with pytest.raises(ValueError, match="positive semidefinite"):
        weighted_svd(w, torch.diag(torch.tensor([1., -1.], dtype=w.dtype)), 1)
    with pytest.raises(ValueError, match="symmetric"):
        weighted_svd(w, torch.tensor([[1., 1.], [0., 1.]], dtype=w.dtype), 1)


@pytest.mark.parametrize("layer,x", [
    (nn.Linear(4, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)),
    (nn.Conv1d(4, 6, 3, padding=2, stride=2, dilation=2, groups=2, dtype=torch.float64), torch.randn(5, 4, 13, dtype=torch.float64)),
    (nn.Conv2d(4, 6, 3, padding=1, groups=2, padding_mode="reflect", dtype=torch.float64), torch.randn(3, 4, 5, 6, dtype=torch.float64))])
def test_real_linear_grouped_conv_full_rank_and_calibration_patches(layer, x):
    before = deepcopy(layer.state_dict())
    rank = min(layer.out_features, layer.in_features) if isinstance(layer, nn.Linear) else min(layer.out_channels // layer.groups, layer.weight[0].numel())
    actual, _ = approximate_layer(layer, x, rank, weighted=True)
    torch.testing.assert_close(actual(x), layer(x), rtol=1e-10, atol=1e-10)
    rows = activation_rows(layer, x)
    assert len(rows) == getattr(layer, "groups", 1)
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_real_normalizations_match_independent_formula_and_production(rank):
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    from fedcore.losses.low_rank_loss import OrthogonalLoss
    layer = DecomposedLinear(nn.Linear(4, 3, dtype=torch.float64))
    u, s, vh = layer.get_U_S_Vh()
    layer.set_U_S_Vh(2 * u[:, :rank], s[:rank], .5 * vh[:rank])
    u, _, vh = layer.get_U_S_Vh()
    identity = torch.eye(rank, dtype=u.dtype)
    numerator = ((u.T @ u - identity) ** 2).sum() + ((vh @ vh.T - identity) ** 2).sum()
    one = orthogonal_penalty(layer, normalization="rank", coefficient=.7)
    squared = orthogonal_penalty(layer, normalization="rank_squared", coefficient=.7)
    torch.testing.assert_close(one, .7 * numerator / rank)
    torch.testing.assert_close(squared, .7 * numerator / rank ** 2)
    torch.testing.assert_close(one, OrthogonalLoss(.7)(layer))
    squared.backward()
    assert torch.isfinite(layer.U.grad).all() and torch.isfinite(layer.Vh.grad).all()
    layer.requires_grad_(False)
    assert torch.isfinite(orthogonal_penalty(layer, normalization="rank_squared", coefficient=.7))


def test_trained_equivalent_factors_use_actual_operator_spectrum():
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    layer = DecomposedLinear(nn.Linear(4, 3, dtype=torch.float64))
    u, s, vh = layer.get_U_S_Vh()
    layer.set_U_S_Vh(u * 100., s / 100., vh)
    evidence = factor_diagnostics(layer)[0]
    assert evidence["operator_spectrum"][0] == pytest.approx(torch.linalg.svdvals(layer.factor_matrix()).tolist())
    assert evidence["operator_spectrum"][0] != pytest.approx(layer.S.tolist())


def test_rank_and_regularization_ablation_public_real_path(bundle, tmp_path):
    protocol = ExperimentProtocol(baseline_epochs=0, finetune_epochs=1, batch_size=4, measurement_repeats=2, warmup=0)
    before = deepcopy(bundle.original_model.state_dict())
    ranks = run_rank_ablation(bundle, protocol, tmp_path / "rank", layer_path="0", ranks=(1, 4))
    assert len(ranks["records"]) == 4
    assert all(record["training_steps"] == 2 and record["measurement"]["status"] == "succeeded" for record in ranks["records"])
    regularization = run_regularization_ablation(bundle, protocol, tmp_path / "regularization", layer_path="0", rank=2,
        variants=[RegularizationVariant("none", 0), RegularizationVariant("hoyer", .001),
                  RegularizationVariant("orthogonal", .01, "rank"), RegularizationVariant("orthogonal", .02, "rank_squared"), RegularizationVariant("norm", .001)])
    assert len({record["starting_sha256"] for record in regularization["records"]}) == 1
    assert all(record["steps"] == 2 and record["measurement"]["status"] == "succeeded" for record in regularization["records"])
    for key, value in bundle.original_model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    json.dumps(ranks, allow_nan=False)


def test_training_cost_controls_real_teacher_lora_merge_and_qat(bundle, tmp_path):
    protocol = ExperimentProtocol(baseline_epochs=0, finetune_epochs=1, batch_size=4, measurement_repeats=2, warmup=0)
    result = run_training_cost_controls(bundle, protocol, tmp_path / "training")
    records = {r["variant"]: r for r in result["records"]}
    assert all(r["steps"] == 2 and r["measurement"]["status"] == "succeeded" for r in records.values())
    assert records["distillation"]["teacher_forwards"] == 2
    assert records["distillation"]["teacher_forward_seconds"] > 0
    assert 0 < records["lora"]["updated_parameters"] < records["full_finetune"]["updated_parameters"]
    assert records["lora"]["adapter_file_bytes"] > 0
    assert records["lora"]["merge_error"]["relative"] < 1e-5
    assert records["qat"]["updated_parameters"] > 0
    assert records["qat_float_control"]["steps"] == records["qat"]["steps"]
    assert not result["test_used"]


def test_ordering_real_operations_and_common_final_updates(bundle, tmp_path):
    protocol = ExperimentProtocol(baseline_epochs=0, finetune_epochs=1, batch_size=4, measurement_repeats=2, warmup=0)
    result = run_order_ablation(bundle, protocol, tmp_path, rank=1)
    assert len(result["records"]) == 2
    assert all(r["steps"] == 2 for r in result["records"])
    assert result["records"][0]["chain"][0]["method"] == "pruning"
    assert result["records"][1]["chain"][0]["method"] == "svd"


def test_early_late_preserves_proposals_and_detects_false_rejections():
    calls = []
    def execute(value):
        calls.append(value)
        if value == "bad":
            raise ValueError("real operator rejects invalid state")
        return {"status": "succeeded", "quality": 1.}
    result = compare_validation_timing(("good", "bad", "good"), lambda item: item != "bad", execute)
    assert calls == ["good", "good", "good", "bad", "good"]
    assert result["status"] == "completed"
    mismatch = compare_validation_timing(("good",), lambda item: False, execute)
    assert mismatch["status"] == "stop_validity_mismatch"


def test_validity_replay_uses_real_quantizer_checker_and_real_artifacts(bundle, tmp_path):
    from fedcore.algorithm.quantization.quantizers import validate_quantization_request
    protocol = ExperimentProtocol(baseline_epochs=0, finetune_epochs=0, measurement_repeats=2, warmup=0)
    good = CandidateSpec("ptq", {"mode": "dynamic", "backend": "fbgemm"})
    bad = CandidateSpec("ptq", {"mode": "dynamic", "backend": "none"})
    def checker(candidate):
        try:
            validate_quantization_request(bundle.original_model, bundle.calibration.x[:1],
                                          candidate.parameters["mode"], candidate.parameters["backend"], torch.qint8)
            return True
        except ValueError:
            return False
    report = run_validity_ablation(bundle, protocol, tmp_path / "validity", proposals=(good, bad), validator=checker)
    assert report["status"] == "completed"
    assert report["runs"]["early"]["successful_fraction"] == .5
    assert report["runs"]["late"]["failed_execution_seconds"] > 0
    assert "hypervolume" in report["runs"]["early"]["archive"]
    assert report["runs"]["late"]["records"][1]["executed"]
