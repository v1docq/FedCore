"""Bounded CPU mechanisms; not reproductions of paper-quality experiments.

python -m examples.low_rank_ann.p5_profiles --output local-p5-demo
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from fedcore.algorithm.low_rank.architecture_profiles import build_architecture
from fedcore.algorithm.low_rank.slicegpt_specs import SliceGPT
from fedcore.algorithm.low_rank.slicegpt_layers import SliceGPTModel
from fedcore.algorithm.low_rank.lightformer_specs import LightFormer
from fedcore.algorithm.low_rank.lightformer_layers import LightFormerTransformer
from fedcore.algorithm.low_rank.lrt_specs import LRT
from fedcore.algorithm.low_rank.lrt_profiles import build_dense_lrt, lrt_costs
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model, task_loss
from fedcore.tools.registry.checkpoint_manager import CheckpointManager
from fedcore.tools.export import export_model


class PackedEncoderDecoder(nn.Module):
    """Fixed export input [batch, 2, tokens]: source then decoder tokens."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model(tokens[:, 0], tokens[:, 1])


class NextTokenObjective:
    def __init__(self, packed=False):
        self.packed = packed
        self.evidence = {'objective': 'causal_next_token_cross_entropy', 'packed_encoder_decoder': packed,
                         'label_shift': 'logits[:, :-1] versus target[:, 1:]'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __call__(self, model, x, target):
        output = model(x[:, 0], x[:, 1]) if self.packed else model(x)
        return task_loss(output, target, 'language_model')

    def before_step(self, model, optimizer):
        live = [id(p) for p in model.parameters() if p.requires_grad]
        owned = [id(p) for group in optimizer.param_groups for p in group['params']]
        if len(owned) != len(set(owned)) or set(owned) != set(live):
            raise RuntimeError('Optimizer must own exactly the unique live Parameters')


def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _train(model, tokens, *, packed=False):
    x = torch.stack((tokens.flip(1), tokens), dim=1) if packed else tokens
    split = TensorSplit(x, tokens, tuple(f'train-{i}' for i in range(len(tokens))))
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    evidence = train_model(model, split, 'language_model', epochs=2, batch_size=4,
        learning_rate=.003, seed=53, batch_objective=NextTokenObjective(packed), max_updates=6)
    changed = [name for name, value in model.named_parameters() if not torch.equal(value, original[name])]
    if not changed:
        raise RuntimeError('Training did not change any Parameter')
    return dict(evidence, changed_parameters=changed, changed_parameter_count=len(changed))


def _persist(output, name, result, validation, *, packed=False, source=None, training=None):
    root = output / name
    root.mkdir(exist_ok=True)
    model = result.model.eval()
    inputs = (validation.flip(1), validation) if packed else (validation,)
    manager = CheckpointManager(str(root / 'registry'), auto_cleanup=False)
    checkpoint = root / 'checkpoint.pt'
    manager.save_to_file(manager.serialize_to_bytes(model), str(checkpoint))
    restored = manager.load_from_file(str(checkpoint)).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(*inputs), model(*inputs), rtol=0, atol=0)
        loss = float(task_loss(restored(*inputs), validation, 'language_model'))
        mse = None if source is None else float((restored(*inputs)-source(*inputs)).square().mean())
    export_input = torch.stack(inputs, dim=1) if packed else validation
    export_graph = PackedEncoderDecoder(restored) if packed else restored
    artifact = export_model(export_graph, 'torchscript', root / 'deployed.pt', export_input[:1])
    with artifact.open('rb') as stream:
        deployed = torch.jit.load(stream)
    torch.testing.assert_close(deployed(export_input[:1]), export_graph(export_input[:1]))
    report = dict(result.evidence, case=name, training=training,
        validation_next_token_ce=loss, validation_source_logit_mse=mse,
        validation_data_sha256=_hash(validation), checkpoint_bytes=checkpoint.stat().st_size,
        artifact_bytes=artifact.stat().st_size,
        export_contract=('fixed [batch,2,tokens] source/decoder input' if packed else 'fixed token tensor input'),
        export_limit='dynamic lengths, padding masks and cached decoding are not certified by this trace',
        status='checkpoint_and_fixed_input_export_verified')
    (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(20261012)
        calibration = torch.randint(0, 17, (8, 6))
        validation = torch.randint(0, 17, (5, 7))
        # Synthetic learnable cycle. No downloaded corpus or acoustic data.
        train = (torch.randint(0, 17, (12, 1)) + torch.arange(6)) % 17
        slice_source = SliceGPTModel(17, 8, 2, 2, 1, 4, 12, dtype=torch.float64).eval()
        for block in slice_source.blocks:
            block.norm.weight.copy_(torch.linspace(.6, 1.4, 8))
        slice_source.final_norm.weight.copy_(torch.linspace(.7, 1.3, 8))
        light_source = LightFormerTransformer(17, 17, 8, 2, 12, (0, 1), (0, 1, 2),
                                              max_positions=12, dtype=torch.float64).eval()
    reports = []
    for name, widths in (('slicegpt_full', ()), ('slicegpt_uniform', (6, 6, 6, 6, 6)),
                         ('slicegpt_mixed', (6, 6, 5, 5, 4))):
        result = build_architecture(SliceGPT(widths), source=slice_source, calibration=calibration,
                                    verification_inputs=validation)
        report = _persist(output, name, result, validation, source=slice_source)
        if not widths:
            torch.testing.assert_close(result.model(validation), slice_source(validation), rtol=1e-10, atol=1e-10)
        reports.append(report)
    for name, spec in (
        ('lightformer_encoder_shared', LightFormer(2, (0, 0, 0, 1, 1, 1), (0, 1, 2), (0, 1), (0, 1, 2))),
        ('lightformer_decoder_extension', LightFormer(2, (0, 0, 0, 1, 1, 1), (0, 0), (0, 1), (1,))),
    ):
        result = build_architecture(spec, source=light_source)
        training = _train(result.model, train, packed=True)
        reports.append(_persist(output, name, result, validation, packed=True, training=training))
    spec = LRT(vocab_size=17, d_model=8, num_heads=2, d_k=4, d_v=4, d_ff=12,
               max_seq_len=12, seed=37)
    for name, result in (('lrt_factorized', build_architecture(spec, dtype=torch.float64)),
                         ('lrt_dense_control', build_dense_lrt(spec, dtype=torch.float64))):
        training = _train(result.model, train)
        report = _persist(output, name, result, validation, training=training)
        report['comparison'] = {'same_data': _hash(train), 'same_updates': 6, 'same_adam_lr': .003,
            'same_shuffle_seed': 53, 'initialization': spec.initialization,
            'initial_functions_equal': False, 'accuracy_superiority_claimed': False}
        report['analytical_costs'] = lrt_costs(spec, factorized=name == 'lrt_factorized',
                                             dtype=torch.float64, batch_size=1, sequence_length=7)
        (output / name / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        reports.append(report)
    summary = [{'case': row['case'], 'status': row['status'],
        'parameters': row.get('parameters_after', row.get('costs', {}).get('parameter_count')),
        'tensor_bytes': row.get('tensor_bytes_after', row.get('costs', {}).get('tensor_bytes')),
        'validation_next_token_ce': row['validation_next_token_ce'],
        'validation_source_logit_mse': row['validation_source_logit_mse'],
        'training_updates': 0 if row['training'] is None else row['training']['training_steps']}
        for row in reports]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), allow_nan=False))


if __name__ == '__main__':
    main()
