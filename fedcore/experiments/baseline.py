"""Persist and reuse a trained common baseline through data-only checkpoints."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import pickle
from pathlib import Path
import time

import torch

from fedcore.tools.atomic_json import write_atomic_json
from .measurement import file_hash
from .protocol import ProtocolError, role_content_identity, stable_hash, validate_roles


def _identity(bundle, protocol, environment):
    from .runner import model_state_hash
    roles = role_content_identity({role: getattr(bundle, role).manifest()
                                   for role in ('train', 'validation', 'calibration')})
    return {'task': bundle.task, 'initial_model_sha256': model_state_hash(bundle.original_model),
            'roles': roles, 'training': {name: getattr(protocol, name) for name in
                ('seed', 'baseline_epochs', 'batch_size', 'learning_rate', 'device', 'threads')},
            'environment': environment}


def _load(bundle, protocol, checkpoint, identity):
    from .runner import model_state_hash
    path = Path(checkpoint).resolve()
    if path.is_dir():
        path = path / 'baseline.json'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ProtocolError('Cannot read a baseline checkpoint manifest') from error
    if manifest.get('version') != 1 or manifest.get('identity') != identity or manifest.get('identity_sha256') != stable_hash(identity):
        raise ProtocolError('Baseline checkpoint provenance differs from the requested data/model/training/environment')
    record = manifest.get('training', {})
    expected_steps = math.ceil(len(bundle.train.ids) / protocol.batch_size) * protocol.baseline_epochs
    history = record.get('train_loss', [])
    if (record.get('status') != 'trained_on_train' or record.get('training_role') != 'train' or
            expected_steps <= 0 or record.get('training_steps') != expected_steps or
            not isinstance(history, list) or len(history) != expected_steps or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in history)):
        raise ProtocolError('Baseline checkpoint has no verified positive training history')
    seconds = record.get('wall_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ProtocolError('Baseline checkpoint training cost must be finite and nonnegative')
    info = manifest.get('state', {})
    # Always load the sidecar belonging to this manifest, never an executable
    # architecture or an arbitrary path declared inside the JSON payload.
    state_path = path.parent / 'baseline_state.pt'
    if not state_path.is_file() or info.get('filename') != state_path.name or file_hash(state_path) != info.get('sha256'):
        raise ProtocolError('Baseline state file hash mismatch')
    model = deepcopy(bundle.original_model).cpu()
    try:
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError, OSError, EOFError, pickle.UnpicklingError) as error:
        raise ProtocolError('Baseline state is incompatible with the explicit architecture') from error
    state_hash = model_state_hash(model)
    if state_hash != info.get('model_sha256') or state_hash != record.get('state_sha256'):
        raise ProtocolError('Baseline trained state hash mismatch')
    return model.eval(), dict(record), path


def prepare_baseline(bundle, protocol, output_dir, *, checkpoint=None, environment=None):
    """Return an independently owned model and a verified training cost record.

    Training settings/data/initial weights/source/environment must match to
    reuse. Test content, compression/measurement choices and search budgets do
    not select or train this state. Zero epochs remains a provided unverified
    state and cannot be used as a trained checkpoint. Each consumer receives
    its own tensors. Common training cost is retained for paired budget charging;
    actual verification time is reported separately.
    """
    from .runner import _seeded, environment_manifest, model_state_hash, train_model
    started = time.perf_counter()
    validate_roles(bundle)
    environment = environment_manifest() if environment is None else environment
    identity = _identity(bundle, protocol, environment)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'baseline.json').exists():
        raise ProtocolError('A baseline checkpoint already exists in the output directory')
    source_checkpoint = None
    if checkpoint is not None:
        model, record, source_checkpoint = _load(bundle, protocol, checkpoint, identity)
        record.update(reused=True, actual_work_seconds=time.perf_counter() - started,
                      source_checkpoint=str(source_checkpoint), source_checkpoint_sha256=file_hash(source_checkpoint))
    else:
        model = deepcopy(bundle.original_model).cpu()
        training_started = time.perf_counter()
        preparation_seconds = training_started - started
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(protocol.threads)
            with _seeded(protocol.seed, protocol.device):
                if protocol.device.startswith('cuda'):
                    torch.cuda.synchronize(protocol.device)
                training = train_model(model, bundle.train, bundle.task, epochs=protocol.baseline_epochs,
                                       batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
                                       device=protocol.device, seed=protocol.seed)
                if protocol.device.startswith('cuda'):
                    torch.cuda.synchronize(protocol.device)
        finally:
            torch.set_num_threads(previous_threads)
        training_seconds = time.perf_counter() - training_started
        seconds = time.perf_counter() - started
        model.cpu().eval()
        record = {'status': 'trained_on_train' if training['training_steps'] else 'provided_state; pretraining_not_verified_by_this_helper',
                  **training, 'wall_seconds': seconds, 'actual_work_seconds': seconds, 'reused': False,
                  'training_seconds': training_seconds, 'preparation_seconds': preparation_seconds,
                  'initial_sha256': identity['initial_model_sha256'], 'state_sha256': model_state_hash(model)}
    # Compatibility aliases in ablation reports; all refer to the same state.
    record.update(steps=record['training_steps'], seconds=record['wall_seconds'], trained_sha256=record['state_sha256'],
                  identity_sha256=stable_hash(identity), checkpoint=str(root / 'baseline.json'))
    state_path = root / 'baseline_state.pt'
    persistence_started = time.perf_counter()
    torch.save(model.state_dict(), state_path)
    state_info = {'filename': state_path.name, 'sha256': file_hash(state_path), 'model_sha256': model_state_hash(model)}
    record['actual_work_seconds'] = time.perf_counter() - started
    if not record['reused']:
        record.update(wall_seconds=record['actual_work_seconds'], seconds=record['actual_work_seconds'],
                      persistence_seconds=time.perf_counter() - persistence_started)
    else:
        record['reuse_persistence_seconds'] = time.perf_counter() - persistence_started
    record['cost_scope'] = 'common role/source/state verification, training and state persistence; final JSON publication/hash excluded'
    write_atomic_json(root / 'baseline.json', {'version': 1, 'identity': identity,
                      'identity_sha256': stable_hash(identity), 'training': record, 'state': state_info})
    record.update(checkpoint_sha256=file_hash(root / 'baseline.json'), state_file_sha256=state_info['sha256'])
    return model, record
