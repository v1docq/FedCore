"""Public P5 construction, checkpoint and boundary contracts."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.architecture_profiles import (
    ArchitectureProfileError, architecture_payload, build_architecture,
    parse_architecture, plan_architecture,
)
from fedcore.algorithm.low_rank.slicegpt_specs import SliceGPT
from fedcore.algorithm.low_rank.slicegpt_layers import SliceGPTModel
from fedcore.algorithm.low_rank.lightformer_specs import LightFormer
from fedcore.algorithm.low_rank.lightformer_layers import LightFormerTransformer
from fedcore.algorithm.low_rank.lrt_specs import LRT
from fedcore.tools.registry.checkpoint_manager import CheckpointError, CheckpointManager


@pytest.fixture(autouse=True)
def bounded_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(140)
        yield
    torch.set_num_threads(old)


def cases(dtype=torch.float64):
    tokens = torch.randint(0, 13, (3, 5))
    slice_source = SliceGPTModel(13, 8, 1, 2, 1, 4, 12, dtype=dtype).eval()
    light_source = LightFormerTransformer(13, 13, 8, 2, 12, (0, 1), (0, 1),
                                          max_positions=8, dtype=dtype).eval()
    return (
        (SliceGPT(), dict(source=slice_source, calibration=tokens), (tokens,)),
        (SliceGPT(widths=(6, 6, 4)), dict(source=slice_source, calibration=tokens), (tokens,)),
        (LightFormer(2, (0, 0, 1), (0, 0), (0, 1), (1,)),
         dict(source=light_source), (tokens, tokens)),
        (LRT(vocab_size=13, d_model=8, num_heads=2, d_k=4, d_v=4, d_ff=12,
             max_seq_len=8), dict(dtype=dtype), (tokens,)),
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_facade_roundtrip_costs_source_independence_and_checkpoint(tmp_path, dtype):
    manager = CheckpointManager(str(tmp_path / 'registry'), auto_cleanup=False)
    for spec, options, inputs in cases(dtype):
        source = options.get('source')
        original = deepcopy(source.state_dict()) if source is not None else None
        rng = torch.get_rng_state().clone()
        plan = plan_architecture(spec, **options)
        assert torch.equal(torch.get_rng_state(), rng)
        result = build_architecture(spec, **options)
        assert torch.equal(torch.get_rng_state(), rng)
        assert result.evidence['parameters_after'] == sum(p.numel() for p in result.model.parameters())
        assert plan.parameter_count == result.evidence['parameters_after']
        assert plan.tensor_bytes == result.evidence['tensor_bytes_after']
        json.dumps(result.evidence, allow_nan=False)
        if source is not None:
            for key, value in original.items():
                torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)
            pointers = {p.untyped_storage().data_ptr() for p in source.parameters()}
            assert not any(p.untyped_storage().data_ptr() in pointers for p in result.model.parameters())
        payload = manager.serialize_to_bytes(result.model)
        restored = manager.deserialize_from_bytes(payload).eval()
        result.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(*inputs), result.model(*inputs), rtol=0, atol=0)
        if isinstance(spec, LightFormer):
            assert restored.encoder_layers[0] is restored.encoder_layers[1]
            assert restored.decoder_layers[0] is restored.decoder_layers[1]
        if isinstance(spec, SliceGPT):
            assert restored.embedding.weight is restored.lm_head.weight


def test_strict_json_config_and_inapplicable_inputs():
    for spec, _, _ in cases():
        payload = architecture_payload(spec)
        assert parse_architecture(payload['architecture'], json.loads(json.dumps(payload['options']))) == spec
        with pytest.raises(ValueError):
            parse_architecture(payload['architecture'], dict(payload['options'], unexpected=True))
    for name, options in [('lrt', {'d_model': True}), ('unknown', {}), ('slicegpt', {'widths': '8'})]:
        with pytest.raises(ValueError):
            parse_architecture(name, options)
    with pytest.raises(ArchitectureProfileError):
        build_architecture(LRT(), source=nn.Linear(2, 2))
    with pytest.raises(ArchitectureProfileError):
        build_architecture(LRT(), calibration=torch.ones(1, 2))
    with pytest.raises(ArchitectureProfileError):
        build_architecture(SliceGPT())
    with pytest.raises(ArchitectureProfileError):
        build_architecture(LRT(), device='cuda')


def test_preflight_budgets_refuse_before_execution(monkeypatch):
    import fedcore.algorithm.low_rank.architecture_profiles as facade
    def forbidden(*args, **kwargs):
        pytest.fail('execution happened after a metadata budget refusal')
    monkeypatch.setattr(facade, '_execute', forbidden)
    for spec, options, _ in cases():
        for budget in ({'parameter_budget': 0}, {'tensor_byte_budget': 0}):
            with pytest.raises(ArchitectureProfileError, match='BudgetInfeasible'):
                build_architecture(spec, **options, **budget)


def test_existing_low_rank_facade(tmp_path, monkeypatch):
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH', str(tmp_path / 'facade-registry'))
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    facade = LowRankModel()
    result = facade.build_architecture(LRT())
    assert facade.model_after is result.model
    assert result.model._fedcore_requires_optimizer_rebuild


def test_checkpoint_refuses_nonfinite_state_unknown_descriptor_and_runtime_hook(tmp_path):
    manager = CheckpointManager(str(tmp_path / 'registry'), auto_cleanup=False)
    for spec, options, _ in cases():
        model = build_architecture(spec, **options).model
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float('nan')
        with pytest.raises(CheckpointError):
            manager.serialize_to_bytes(model)
    model = build_architecture(LRT()).model
    hook = model.register_forward_hook(lambda m, args, out: out)
    with pytest.raises(CheckpointError):
        manager.serialize_to_bytes(model)
    hook.remove()
    from fedcore.algorithm.low_rank.architecture_checkpoint import build_p5
    with pytest.raises(ValueError):
        build_p5({'type': 'P5Architecture', 'architecture': 'subprocess',
                  'version': 1, 'config': {}, 'dtype': 'torch.float32'})
