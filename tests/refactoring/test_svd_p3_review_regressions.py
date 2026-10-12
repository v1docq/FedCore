"""Independent numerical witnesses for the bounded P3 profiles.

References use explicit sample loops, direct dense products or PyTorch lstsq;
they intentionally do not reuse the solvers being verified.
"""
from dataclasses import replace
import io
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.projection_profiles import (
    absorb_ctr_linear_blocks, espace_statistics, pela_regularizer,
    select_espace_projector, solve_ctr_field, solve_espace)
from fedcore.algorithm.low_rank.projection_specs import ESPACE
from fedcore.algorithm.low_rank.statistics import StatisticsPassport
from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention
from fedcore.algorithm.low_rank.coupled_profiles import mode_product_factors, mlp_channel_fit
from fedcore.algorithm.low_rank.coupled_execution import (
    CoupledProfileError, plan_coupled_profile, prepare_coupled_replacements)
from fedcore.algorithm.low_rank.coupled_specs import FlatValueOutput


def _passport():
    return StatisticsPassport("review-checkpoint", "review-graph", "review-predecessor",
                              "review-input", "review-data")


def _outer(vector):
    return vector[:, None] * vector[None, :]


def _unit(vector):
    return vector / vector.norm() if vector.norm() else torch.zeros_like(vector)


def _explicit_espace(rows, weights, gradients, criterion):
    result = torch.zeros(rows.shape[1], rows.shape[1], dtype=rows.dtype)
    for sample, gradient in zip(rows, gradients):
        x = _unit(sample) if criterion in ("nmse", "go_normalized", "nl_normalized") else sample
        xx = _outer(x)
        if criterion.startswith("go_"):
            for row in weights:
                w = _unit(row) if criterion == "go_normalized" else row
                ww = _outer(w)
                result += (xx @ ww + ww @ xx) / (len(rows) * len(weights))
        elif criterion.startswith("nl_"):
            g = _unit(gradient) if criterion == "nl_normalized" else gradient
            gg = _outer(g)
            result += (xx @ gg + gg @ xx) / len(rows)
        else:
            result += xx / len(rows)
    return result


@pytest.mark.parametrize("criterion", ["mse", "nmse", "go_mse_bound", "go_normalized",
                                       "nl_first_order_bound", "nl_normalized"])
def test_espace_six_criteria_match_paired_sample_and_weight_row_reference(criterion):
    x = torch.tensor([[2., 1.], [0., 0.], [0., 3.], [-1., 2.]], dtype=torch.float64)
    w = torch.tensor([[3., 0.], [1., 2.], [0., 0.]], dtype=torch.float64)
    g = torch.tensor([[1., -2.], [0., 1.], [-2., 1.], [0., 0.]], dtype=torch.float64)
    observed = espace_statistics(x, _passport(), ESPACE(criterion=criterion), weight=w, gradients=g)
    expected = _explicit_espace(x, w, g, criterion)
    torch.testing.assert_close(observed.matrix, expected, rtol=1e-12, atol=1e-12)
    # Scale-invariance applies only to the named normalized calibration inputs.
    if criterion in ("nmse", "go_normalized", "nl_normalized"):
        rescaled = espace_statistics(7*x, _passport(), ESPACE(criterion=criterion), weight=3*w, gradients=5*g)
        torch.testing.assert_close(observed.matrix, rescaled.matrix, rtol=1e-12, atol=1e-12)


def test_espace_indefinite_matrix_selects_algebraic_eigenvalue_and_compile_preserves_function():
    x = torch.eye(2, dtype=torch.float64)
    stats = espace_statistics(x, _passport())
    stats = replace(stats, matrix=torch.diag(torch.tensor([1., -100.], dtype=torch.float64)))
    chosen = select_espace_projector(stats, 1)
    torch.testing.assert_close(chosen.basis @ chosen.basis.T,
                               torch.diag(torch.tensor([1., 0.], dtype=torch.float64)))
    weight = torch.tensor([[1., 3.], [2., -1.], [4., 2.]], dtype=torch.float64)
    pair = solve_espace(weight, stats, 1)
    inputs = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64)
    expected = inputs @ (weight @ chosen.basis @ chosen.basis.T).T
    actual = (inputs @ pair.right.T) @ pair.left.T
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("norm_name, norm_value, expected_gradient", [
    ("max_element", 2., [[0., 1.], [0., 0.]]),
    ("max_row_sum", 3., [[1., 1.], [0., 0.]])])
def test_pela_epsilon_subtraction_is_constant_while_hinge_changes_gradient(norm_name, norm_value, expected_gradient):
    left = torch.eye(2, dtype=torch.float64)
    right = torch.tensor([[1., 2.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    reference = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    a = pela_regularizer(left, right, reference, infinity_norm=norm_name, epsilon=0.)
    b = pela_regularizer(left, right, reference, infinity_norm=norm_name, epsilon=10.)
    c = pela_regularizer(left, right, reference, infinity_norm=norm_name, epsilon=10., variant="hinge_variant")
    grad_a = torch.autograd.grad(a, right, retain_graph=True)[0]
    grad_b = torch.autograd.grad(b, right, retain_graph=True)[0]
    grad_c = torch.autograd.grad(c, right)[0]
    assert float(a) == norm_value and float(b) == norm_value - 10 and float(c) == 0
    torch.testing.assert_close(grad_a, torch.tensor(expected_gradient, dtype=torch.float64))
    torch.testing.assert_close(grad_b, grad_a)
    assert torch.count_nonzero(grad_c) == 0
    assert reference.grad is None


def test_ctr_centered_codes_affine_block_sum_and_decoded_interactions():
    table1 = torch.tensor([[2., 5.], [4., 5.], [6., 5.]], dtype=torch.float64)
    table2 = torch.tensor([[7., -1.], [7., 3.], [7., 8.]], dtype=torch.float64)
    fields = tuple(solve_ctr_field(table, 1, _passport()) for table in (table1, table2))
    blocks = (torch.tensor([[1., 2.], [-3., 4.]], dtype=torch.float64),
              torch.tensor([[5., -6.], [7., 8.]], dtype=torch.float64))
    bias = torch.tensor([.5, -2.], dtype=torch.float64)
    compact, intercept = absorb_ctr_linear_blocks(blocks, (bias, None), fields)
    actual = fields[0].coefficients @ compact[0].T + fields[1].coefficients @ compact[1].T + intercept
    expected = torch.cat((table1, table2), dim=1) @ torch.cat(blocks, dim=1).T + bias
    torch.testing.assert_close(actual, expected)
    decoded = tuple(field.projected_table() for field in fields)
    torch.testing.assert_close((decoded[0]*decoded[1]).sum(1), (table1*table2).sum(1))
    assert not torch.allclose((fields[0].coefficients*fields[1].coefficients).sum(1),
                              (table1*table2).sum(1))


def _direct_attention(module, inputs, positions, padding, mask):
    """Scalar/head loops; no runtime RoPE, attention, or cache helper."""
    batch, tokens, _ = inputs.shape
    query = inputs @ module.q_proj.weight.T + module.q_proj.bias
    key = inputs @ module.k_proj.weight.T + module.k_proj.bias
    value = inputs @ module.v_proj.weight.T + module.v_proj.bias
    query = query.reshape(batch, tokens, module.num_heads, module.qk_dim).clone()
    key = key.reshape(batch, tokens, module.num_kv_heads, module.qk_dim).clone()
    value = value.reshape(batch, tokens, module.num_kv_heads, module.value_dim)
    for states, heads, mapping in ((query, module.num_heads, module.head_map),
                                   (key, module.num_kv_heads, torch.arange(module.num_kv_heads))):
        for b in range(batch):
            for t in range(tokens):
                for head in range(heads):
                    for pair in range(module.qk_dim//2):
                        coordinate = int(module.group_coordinates[int(mapping[head]), pair*2])
                        angle = positions[b,t].to(inputs.dtype) * module.rope_base ** (-coordinate/module.head_dim)
                        cosine, sine = angle.cos(), angle.sin()
                        a, c = states[b,t,head,pair*2:pair*2+2].clone()
                        states[b,t,head,pair*2] = a*cosine-c*sine
                        states[b,t,head,pair*2+1] = a*sine+c*cosine
    logits = inputs.new_zeros(batch, module.num_heads, tokens, tokens)
    probabilities = torch.zeros_like(logits)
    contexts = inputs.new_zeros(batch, tokens, module.num_heads, module.value_dim)
    for b in range(batch):
        for head in range(module.num_heads):
            group = int(module.head_map[head])
            for t in range(tokens):
                for k in range(tokens):
                    logits[b,head,t,k] = (query[b,t,head]*key[b,k,group]).sum()*module.logits_scale
                valid = torch.tensor([bool(padding[b,k] and mask[t,k] and k<=t) for k in range(tokens)])
                if valid.any():
                    probabilities[b,head,t,valid] = torch.softmax(logits[b,head,t,valid], 0)
                for k in range(tokens):
                    contexts[b,t,head] += probabilities[b,head,t,k]*value[b,k,group]
    outputs = contexts.flatten(-2) @ module.o_proj.weight.T + module.o_proj.bias
    return outputs, logits, probabilities


def test_attention_gqa_rope_scale_masks_and_cache_match_independent_head_loops():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        module = ReferenceAttention(5, 4, 2, 4, head_map=(1,0,1,0),
                                    logits_scale=.37, dtype=torch.float64).eval()
        inputs = torch.randn(2, 4, 5, dtype=torch.float64)
    positions = torch.tensor([[1,3,4,8], [0,1,5,7]])
    padding = torch.tensor([[True,True,False,True], [False,True,True,True]])
    mask = torch.tensor([[False,False,False,False], [True,True,False,False],
                         [True,False,True,False], [True,True,True,True]])
    expected, logits, probabilities = _direct_attention(module, inputs, positions, padding, mask)
    actual = module.attention(inputs, positions=positions, padding_mask=padding,
                              attention_mask=mask, use_cache=True)
    torch.testing.assert_close(actual.logits, logits, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.probabilities, probabilities, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.output, expected, rtol=1e-12, atol=1e-12)
    prefill = module.attention(inputs[:,:3], positions=positions[:,:3], padding_mask=padding[:,:3],
                               attention_mask=mask[:3,:3], use_cache=True)
    decode = module.attention(inputs[:,3:], positions=positions[:,3:], padding_mask=padding,
                              attention_mask=mask[3:], cache=prefill.cache, use_cache=True)
    torch.testing.assert_close(decode.logits, logits[:,:,3:], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(decode.output, expected[:,3:], rtol=1e-12, atol=1e-12)
    assert tuple(decode.cache.keys.shape) == (2,2,4,4)


def test_mode_thin_product_matches_direct_weighted_low_rank_svd_and_differs_from_value_pca():
    left = torch.diag(torch.tensor([.01,100.], dtype=torch.float64))
    right = torch.diag(torch.tensor([10.,1.], dtype=torch.float64))
    inputs = torch.eye(2, dtype=torch.float64)
    pair = mode_product_factors(left, right, inputs, 1)
    direct_output = inputs @ (left@right).T
    u, singular, vh = torch.linalg.svd(direct_output, full_matrices=False)
    optimal = u[:,:1] @ torch.diag(singular[:1]) @ vh[:1]
    actual = (inputs@pair.right.T)@pair.left.T
    torch.testing.assert_close(actual, optimal, rtol=1e-12, atol=1e-12)
    assert float((direct_output-actual).square().sum()) == pytest.approx(.01)
    value_pca_product = left @ torch.diag(torch.tensor([1.,0.], dtype=torch.float64)) @ right
    assert float((direct_output-inputs@value_pca_product.T).square().sum()) == pytest.approx(10000.)


@pytest.mark.parametrize("family", ["flat", "modegpt"])
def test_selected_nonlinear_mlp_channels_fit_singular_design_with_direct_lstsq(family):
    features = torch.tensor([[1.,2.,1.],[2.,1.,2.],[-1.,3.,-1.],[0.,2.,0.]], dtype=torch.float64)
    down = torch.tensor([[1.,3.,-2.],[4.,1.,2.]], dtype=torch.float64)
    observed = mlp_channel_fit(features, down, 2, indices=(0,2), family=family)
    design, target = features[:,[0,2]], features@down.T
    coefficients = torch.linalg.lstsq(design, target).solution
    torch.testing.assert_close(observed.down_weight.T, coefficients, rtol=1e-12, atol=1e-12)
    error = (design@observed.down_weight.T-target).square().sum()
    baseline = (design@down[:,[0,2]].T-target).square().sum()
    assert error <= baseline


def test_coupled_attention_reserves_quadratic_logits_workspace_before_forward():
    module = ReferenceAttention(2, 4, 2, 2, dtype=torch.float64).eval()
    observations = torch.ones(1, 100, 2, dtype=torch.float64)
    specification = FlatValueOutput(observation_point="post_softmax")
    plan = plan_coupled_profile(module, specification, rank=1, target_paths=("",))
    forward_calls = []
    handle = module.register_forward_hook(lambda *args: forward_calls.append(True))
    try:
        with pytest.raises(CoupledProfileError, match="ResourceLimitExceeded"):
            prepare_coupled_replacements(module, observations, specification, plan,
                                         batch_size=1, max_workspace_bytes=5000)
    finally:
        handle.remove()
    assert not forward_calls


def _physical_tensor_bytes(model):
    storages = {}
    for value in (*model.parameters(), *model.buffers()):
        storage = value.untyped_storage()
        storages[(str(value.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(storages.values())


def test_laser_diagnosis_and_separate_export_include_retained_buffers():
    from fedcore.algorithm.low_rank.projection_execution import (
        DiagnoseIntervention, diagnose_laser, export_laser_compact)
    module = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64)).eval()
    module.register_buffer("retained", torch.arange(100, dtype=torch.float64))
    with torch.no_grad():
        module[0].weight.copy_(torch.arange(1., 5., dtype=torch.float64)[:,None].expand(4,4))
    interventions = tuple(DiagnoseIntervention("0", rank, "frozen-baseline", "validation-only", "mse")
                          for rank in (1, 0))
    observed_weights = []
    def evaluate(candidate):
        observed_weights.append(candidate[0].weight.detach().clone())
        return float(candidate[0].weight.square().sum())
    diagnosis = diagnose_laser(module, interventions, evaluate)
    assert all(candidate.tensor_bytes == _physical_tensor_bytes(module)
               for candidate in diagnosis.candidates)
    # Both candidates start from the baseline; the second is the independent zero-rank intervention.
    torch.testing.assert_close(observed_weights[0], module[0].weight, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(observed_weights[1], torch.zeros_like(module[0].weight))
    compact, evidence = export_laser_compact(module, "0", 1)
    assert evidence["tensor_bytes_dense"] == _physical_tensor_bytes(module)
    assert evidence["tensor_bytes_compact"] == _physical_tensor_bytes(compact)
    inputs = torch.tensor([[1., -2., 3., 4.]], dtype=torch.float64)
    torch.testing.assert_close(compact(inputs), module(inputs), rtol=1e-12, atol=1e-12)


def test_ctr_all_consumers_full_rank_affine_sum_buffer_budget_and_portable_roundtrip(tmp_path):
    from fedcore.algorithm.low_rank.method_execution import transform_method
    from fedcore.algorithm.low_rank.projection_execution import ProjectionProfileError
    from fedcore.algorithm.low_rank.projection_layers import CTRFieldGraph, CTRLinearSum
    from fedcore.algorithm.low_rank.projection_specs import CTRAFM
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    tables = {"a": torch.tensor([[3.,1.], [7.,2.], [4.,6.]], dtype=torch.float64),
              "b": torch.tensor([[2.,8.], [6.,3.], [1.,5.]], dtype=torch.float64)}
    linear = CTRLinearSum({"a":2, "b":2}, 2, dtype=torch.float64)
    weights = {"a": torch.tensor([[1.,3.], [-2.,1.]], dtype=torch.float64),
               "b": torch.tensor([[2.,-1.], [4.,2.]], dtype=torch.float64)}
    bias = torch.tensor([.25, -.5], dtype=torch.float64)
    with torch.no_grad():
        for name in tables:
            linear.blocks[name].weight.copy_(weights[name])
        linear.bias.copy_(bias)
    model = CTRFieldGraph({name: nn.Embedding.from_pretrained(table.clone(), freeze=False)
                           for name,table in tables.items()}, linear, (("a","b"),)).eval()
    tokens = torch.tensor([[0,2], [2,0], [1,1]], dtype=torch.long)
    expected_linear = torch.stack([sum(weights[name]@tables[name][row[index]]
        for index,name in enumerate(tables))+bias for row in tokens])
    expected_dots = torch.stack([(tables["a"][row[0]]*tables["b"][row[1]]).sum()
                                 for row in tokens])[:,None]
    specification = CTRAFM((("a",2),("b",2)))
    result = transform_method(model, None, specification)
    actual = result.model(tokens)
    torch.testing.assert_close(actual["linear"], expected_linear, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual["interactions"], expected_dots, rtol=1e-12, atol=1e-12)
    physical = _physical_tensor_bytes(result.model)
    assert result.evidence["tensor_bytes_after"] == physical
    assert result.evidence["buffer_bytes_after"] > 0
    with pytest.raises(ProjectionProfileError, match="BudgetInfeasible"):
        transform_method(model, None, specification, tensor_byte_budget=physical-1)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    checkpoint = tmp_path/"ctr.ckpt"
    manager.save_to_file(manager.serialize_to_bytes(result.model), str(checkpoint))
    restored = manager.load_from_file(str(checkpoint))
    assert restored._fedcore_method_evidence == json.loads(json.dumps(result.evidence))
    assert _physical_tensor_bytes(restored) == physical
    for key, expected in actual.items():
        torch.testing.assert_close(restored(tokens)[key], expected, rtol=0, atol=0)


def test_espace_recovery_trains_full_weight_against_fixed_projector_then_compiles_exactly():
    from fedcore.algorithm.low_rank.method_execution import transform_method
    from fedcore.algorithm.low_rank.projection_layers import FixedProjectorLinear
    from fedcore.experiments.protocol import TensorSplit
    from fedcore.experiments.runner import train_model
    source = nn.Linear(3, 2, dtype=torch.float64).eval()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[1.,-2.,3.], [4.,1.,-1.]], dtype=torch.float64))
        source.bias.copy_(torch.tensor([.5,-.25], dtype=torch.float64))
    x = torch.tensor([[1.,2.,0.], [0.,3.,1.], [2.,1.,-1.], [-1.,0.,2.]], dtype=torch.float64)
    y = torch.tensor([[1.,0.], [0.,1.], [2.,-1.], [1.,3.]], dtype=torch.float64)
    _, eigenvectors = torch.linalg.eigh(x.T@x)
    p = eigenvectors[:,-2:]
    projected = x@p@p.T
    weight, bias = source.weight.detach().clone(), source.bias.detach().clone()
    residual = projected@weight.T+bias-y
    weight_gradient = 2*residual.T@projected/y.numel()
    bias_gradient = 2*residual.sum(0)/y.numel()
    learning_rate = .01
    # First Adam step with default betas, zero initial moments and epsilon=1e-8.
    expected_weight = weight-learning_rate*weight_gradient/(weight_gradient.abs()+1e-8)
    expected_bias = bias-learning_rate*bias_gradient/(bias_gradient.abs()+1e-8)
    calls = []
    def recover(student, stage, updates, objective):
        assert isinstance(student, FixedProjectorLinear)
        assert student.weight.shape == source.weight.shape
        assert stage == "espace_full_weight" and updates == 1
        assert "projection" in dict(student.named_buffers())
        assert "projection" not in dict(student.named_parameters())
        fixed = student.projection.detach().clone()
        split = TensorSplit(x, y, tuple(f"train-{i}" for i in range(len(x))))
        result = train_model(student, split, "regression", epochs=1, batch_size=len(x),
                             learning_rate=learning_rate, batch_objective=objective, max_updates=1)
        torch.testing.assert_close(student.projection, fixed, rtol=0, atol=0)
        torch.testing.assert_close(student.weight.grad, weight_gradient, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(student.bias.grad, bias_gradient, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(student.weight, expected_weight, rtol=1e-12, atol=1e-12)
        calls.append(True)
        return result
    result = transform_method(source, x, ESPACE(updates=1), rank=2, target_paths=("",),
                              recovery_executor=recover)
    assert calls == [True]
    torch.testing.assert_close(result.model(x), projected@expected_weight.T+expected_bias,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(source.weight, weight, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in source.parameters())


@pytest.mark.parametrize("name, corrupt", [
    ("head_map", torch.tensor([2,2,2,2])),
    ("head_map", torch.tensor([1,1,0,0])),
    ("head_map", torch.tensor([0.,0.,1.,1.])),
    ("group_coordinates", torch.tensor([[0,2], [0,1]])),
    ("group_coordinates", torch.tensor([[2,3], [0,1]])),
])
def test_checkpoint_attention_rejects_state_layout_conflicting_with_descriptor(tmp_path, name, corrupt):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError
    model = ReferenceAttention(4, 4, 2, 4, qk_dim=2, dtype=torch.float64).eval()
    original = model.state_dict()[name].clone()
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(model)), weights_only=True)
    payload["state_dict"][name] = corrupt
    with pytest.raises(CheckpointError):
        manager.restore(payload, model=model)
    torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["espace_projector", "ctr_basis", "ctr_affine_correction"])
def test_checkpoint_rejects_invalid_fixed_projection_geometry(tmp_path, kind):
    from fedcore.algorithm.low_rank.projection_layers import FixedProjectorLinear, CenteredPCAEmbedding
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError
    model = (FixedProjectorLinear(3,2,2,dtype=torch.float64) if kind == "espace_projector" else
             CenteredPCAEmbedding(3,2,1,dtype=torch.float64))
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(model)), weights_only=True)
    if kind == "espace_projector":
        payload["state_dict"]["projection"] *= 2
    elif kind == "ctr_basis":
        payload["state_dict"]["basis"] *= 2
    else:
        payload["state_dict"]["mean"] = torch.ones_like(model.mean)
    with pytest.raises(CheckpointError):
        manager.restore(payload)
