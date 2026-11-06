"""Owned fixed-recovery evaluator for the Evo-only CV panel.

The caller owns source/data/deadline and supplies batches explicitly. Importing
does not download CIFAR, train a baseline, contact a host or start a search.
"""
from dataclasses import asdict, dataclass
import math
import json
import os
from pathlib import Path
import random
import time


@dataclass(frozen=True)
class RecoveryBatch:
    indices: tuple[int, ...]
    augmentation_seed: int


@dataclass(frozen=True)
class RecoveryConfig:
    updates: int = 200
    batch_size: int = 128
    learning_rate: float = .001
    momentum: float = .9
    weight_decay: float = 5e-4

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.updates,self.batch_size))
                or any(type(v) not in (float,int) or not math.isfinite(v) or v < 0
                       for v in (self.learning_rate,self.momentum,self.weight_decay))
                or self.learning_rate <= 0 or not 0 <= self.momentum < 1):
            raise ValueError('Explicit bounded SGD recovery parameters required')


def recovery_tape(fit_indices, *, model_seed, config=RecoveryConfig()):
    """One immutable tape per M0, independent of search seed, arm or ancestry."""
    if (type(fit_indices) is not tuple or not fit_indices
            or any(type(i) is not int or i < 0 for i in fit_indices)
            or len(set(fit_indices)) != len(fit_indices)
            or type(model_seed) is not int or model_seed < 0 or type(config) is not RecoveryConfig):
        raise ValueError('Distinct fit indices and a model seed required')
    rng, rows, result = random.Random(model_seed), [], []
    for _ in range(config.updates):
        while len(rows) < config.batch_size:
            chunk = list(fit_indices); rng.shuffle(chunk); rows.extend(chunk)
        selected, rows = tuple(rows[:config.batch_size]), rows[config.batch_size:]
        result.append(RecoveryBatch(selected, rng.randrange(2**32)))
    return tuple(result)


def recover_owned(model, tape, load_batch, *, config=RecoveryConfig(), device='cpu',
                  deadline, clock=time.perf_counter, progress=None, synchronize_updates=False):
    """Fresh optimizer on an owned candidate; no parent weight/state inheritance."""
    import torch
    if (type(config) is not RecoveryConfig or type(tape) is not tuple or len(tape) != config.updates
            or any(type(b) is not RecoveryBatch or len(b.indices) != config.batch_size
                   or any(type(i) is not int or i < 0 for i in b.indices)
                   or type(b.augmentation_seed) is not int or b.augmentation_seed < 0 for b in tape)
            or type(deadline) not in (int,float) or not math.isfinite(deadline)
            or not callable(load_batch) or type(synchronize_updates) is not bool):
        raise ValueError('A complete fixed recovery tape, loader and absolute deadline are required')
    if clock() >= deadline:
        raise TimeoutError('Recovery deadline expired before model transfer')
    target = torch.device(device)
    devices = [] if target.type == 'cpu' else [target.index if target.index is not None else torch.cuda.current_device()]
    import numpy as np
    updates = 0
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            model.to(target).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                momentum=config.momentum, weight_decay=config.weight_decay)
            for batch in tape:
                if clock() >= deadline:
                    raise TimeoutError('Recovery deadline expired; no automatic retry')
                torch.manual_seed(batch.augmentation_seed)
                random.seed(batch.augmentation_seed); np.random.seed(batch.augmentation_seed)
                x, y = load_batch(batch.indices, batch.augmentation_seed)
                if (not isinstance(x,torch.Tensor) or not isinstance(y,torch.Tensor)
                        or x.shape[0] != config.batch_size or y.shape != (config.batch_size,)
                        or x.dtype != torch.float32 or y.dtype != torch.int64
                        or not bool(torch.isfinite(x).all())):
                    raise ValueError('Recovery batch must have explicit finite float inputs/int64 labels')
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x.to(target)), y.to(target))
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite recovery loss')
                if progress is not None: progress('sgd_attempted', updates + 1)
                loss.backward(); optimizer.step()
                if synchronize_updates and target.type == 'cuda':
                    torch.cuda.synchronize(target)
                updates += 1
                if progress is not None: progress('sgd_completed', updates)
            if target.type == 'cuda': torch.cuda.synchronize(target)
            if clock() >= deadline:
                raise TimeoutError('Recovery completed after deadline')
        return dict(updates=updates, config=asdict(config), extra_bn_passes=0,
                    semantics='fixed_float_SGD_before_PTQ; new optimizer; common M0 tape')
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
        model.eval().cpu()


def evaluate_loaded(runtime, batches, *, deadline, expected_rows, clock=time.perf_counter):
    """CE/Top1 from the independently loaded deployed artifact, never float proxy."""
    import torch
    if type(expected_rows) is not int or expected_rows < 1:
        raise ValueError('An explicit complete heldout row count is required')
    total, loss, correct = 0, 0., 0
    with torch.no_grad():
        for inputs, labels in batches:
            if clock() >= deadline: raise TimeoutError('Deployed quality deadline')
            output = runtime(inputs)
            logits = output.detach().cpu() if isinstance(output,torch.Tensor) else torch.from_numpy(output)
            targets = labels.detach().cpu()
            if (logits.ndim != 2 or targets.shape != (logits.shape[0],)
                    or targets.dtype != torch.int64 or not bool(torch.isfinite(logits).all())):
                raise ValueError('Invalid deployed classifier output or targets')
            loss += float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
            correct += int((logits.argmax(1) == targets).sum()); total += len(targets)
    if total != expected_rows:
        raise ValueError('Deployed quality evaluation is incomplete or exceeds the sealed heldout role')
    if clock() >= deadline: raise TimeoutError('No timely complete deployed quality evaluation')
    return dict(cross_entropy=loss/total,accuracy=correct/total,rows=total)


class AttemptLedger:
    """Append-only flushed progress survives worker death; final receipt is atomic.

    An attempted step without a completion is explicitly in flight, never zero
    cost. Each search job must own a fresh output_root; old evidence is preserved.
    """
    def __init__(self, directory, workflow, validation_rows, validation_identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.state = dict(schema='heterogeneous-attempt-v1',status='started',
            workflow_fingerprint=workflow,validation_rows=validation_rows,
            validation_identity=validation_identity,sgd_attempted=0,sgd_completed=0,
            phase='structural_build',started_at=time.perf_counter(),phase_seconds={})
        self._phase_started = self.state['started_at']
        self.record('started')

    def record(self,event,**details):
        with (self.directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event,at=time.perf_counter(),**details),allow_nan=False)+'\n')
            stream.flush()

    def progress(self,event,value):
        self.state[event] = value
        self.record(event,updates=value)

    def phase(self,name):
        now = time.perf_counter()
        key = self.state['phase']
        self.state['phase_seconds'][key] = self.state['phase_seconds'].get(key,0.) + now-self._phase_started
        self.state['phase'] = name; self._phase_started = now
        self.record('phase',name=name)

    def close(self,status,**details):
        self.phase('finished')
        self.state.update(status=status,ended_at=time.perf_counter(),**details)
        target = self.directory/'receipt.json'
        temporary = target.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8') as stream:
            json.dump(self.state,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,target)
        self.record('finished',status=status)


@dataclass
class CandidateEvaluator:
    source: object
    calibration: object
    tape: tuple
    load_batch: object
    validation_batches: object  # a fresh iterator factory over Vsearch only
    output_root: str
    deadline: float
    backend_spec: object
    validation_identity: str
    validation_rows: int = 4000
    recovery_config: RecoveryConfig = RecoveryConfig()
    device: str = 'cpu'
    synchronize_updates: bool = False

    def __call__(self, workflow):
        from fedcore.algorithm.heterogeneous.cv import execute_float_workflow
        from fedcore.algorithm.heterogeneous.backend import (
            BackendContractError, DeploymentArtifact, export_onnx_qdq, validate_placement_backend)
        from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
        from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
        from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure, NumericalFailure, StructuralInvalid
        from fedcore.algorithm.low_rank.plans import Violation
        if (type(self.validation_identity) is not str or not self.validation_identity
                or type(self.validation_rows) is not int or self.validation_rows < 1
                or type(self.synchronize_updates) is not bool):
            raise ValueError('The sealed Vsearch identity and row count are mandatory')
        if time.perf_counter() >= self.deadline:
            return ResourceFailure('CandidateDeadline','No candidate may start after its deadline')
        ledger = AttemptLedger(Path(self.output_root)/workflow.fingerprint,workflow.fingerprint,
                               self.validation_rows,self.validation_identity)
        status, details = 'failed', {}
        try:
            try:
                validate_placement_backend(workflow, self.backend_spec)
            except BackendContractError as error:
                failure = StructuralInvalid((Violation(error.code, 'heterogeneous.export', str(error)),))
                details['failure'] = asdict(failure)
                return failure
            if self.backend_spec.runtime == 'tensorrt':
                from .heterogeneous_device import require_cuda_binding
                if self.device not in ('cuda','cuda:0'):
                    raise ValueError('TensorRT recovery must use the same sole bound CUDA device')
                require_cuda_binding(self.backend_spec.device_uuid)
            result = execute_float_workflow(self.source,self.calibration,workflow)
            if type(result) is not CVWorkflowSuccess:
                details['failure'] = asdict(result); return result
            ledger.phase('float_recovery')
            recovery = recover_owned(result.model,self.tape,self.load_batch,
                config=self.recovery_config,device=self.device,deadline=self.deadline,progress=ledger.progress,
                synchronize_updates=self.synchronize_updates)
            if time.perf_counter() >= self.deadline: raise TimeoutError('No late export')
            ledger.phase('calibrate_export_reload')
            artifact = export_onnx_qdq(result.model,self.calibration,workflow,
                ledger.directory/'artifact',spec=self.backend_spec,recovery_evidence=recovery)
            if type(artifact) is not DeploymentArtifact:
                details['failure'] = asdict(artifact); return artifact
            ledger.phase('deployed_Vsearch')
            quality = evaluate_loaded(artifact.load(),self.validation_batches(),deadline=self.deadline,
                                      expected_rows=self.validation_rows)
            status, details = 'succeeded',dict(quality=quality,storage_bytes=artifact.storage_bytes,
                manifest_path=artifact.manifest_path,recovery=recovery)
            # The search archive includes all finite points. Storage feasibility
            # is applied by the sealed selection rule, not a fake cheap penalty.
            return GrammarFitness(quality['cross_entropy'],float(artifact.storage_bytes))
        except TimeoutError as error:
            details['failure'] = str(error)
            return ResourceFailure('CandidateDeadline',str(error))
        except (ValueError,RuntimeError,TypeError) as error:
            details['failure'] = str(error)
            return NumericalFailure(type(error).__name__,str(error))
        finally:
            ledger.close(status,**details)
