"""Effect interpreter for independent-M0 workers; no Torch at module import."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import math
import io
import os
from pathlib import Path
import random
import sys
import time

from .cv_independent_protocol import (AttemptBudget, IndependentProtocol, MODELS,
    Reservation, digest, derived_seed, eligible_finalists, stage_deadline)
from .cv_independent_data import recovery_tape, tape_payload
from .cv_sparse_runtime import (GPUData, PilotFatal, PilotStop, Runtime as OldRuntime,
    checkpoint, classify_cv_candidate_failure, sha256_file)


class Runtime(OldRuntime):
    """Use checked telemetry, with this protocol's stage map and worker clock."""
    def __init__(self, protocol, output, started, deadline):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['OMP_NUM_THREADS'] = str(protocol.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(protocol.cpu_threads)
        import torch
        self.torch, self.protocol, self.output = torch, protocol, output
        self.started, self.deadline = started, deadline
        self.evaluation_progress = {}
        self.stage, self.stage_started, self.stage_limit, self.last_heartbeat = 'manifest', started, deadline, 0.
        if time.perf_counter() >= deadline:
            raise PilotStop('manifest', 'absolute_deadline_before_cuda_context')
        torch.set_num_threads(protocol.cpu_threads)
        torch.set_num_interop_threads(protocol.cpu_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible A100 40 GB is required')
        properties = torch.cuda.get_device_properties(0)
        if 'A100' not in properties.name or not 38*1024**3 <= properties.total_memory <= 41*1024**3:
            raise ValueError('One A100 40 GB is required')
        torch.manual_seed(MODELS[0].initialization)
        torch.cuda.manual_seed_all(MODELS[0].initialization)
        self.environment = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'device': properties.name,
            'total_memory': properties.total_memory, 'cpu_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'deterministic_algorithms': True,
            'tf32': False, 'environment': {k: os.environ.get(k) for k in
                ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'PYTHONHASHSEED')}}
        self.check()

    def check(self, deadline=None, **progress):
        super().check(deadline, **progress)
        label, rows = progress.get('evaluation'), progress.get('rows')
        if type(label) is str and type(rows) is int and rows > self.evaluation_progress.get(label, 0):
            self.evaluation_progress[label] = rows
            self.output.append({'event': 'evaluation_progress', 'label': label,
                                'completed_rows': rows, 'timestamp': time.perf_counter()})

    @contextmanager
    def phase(self, name):
        self.stage, self.stage_started = name, time.perf_counter()
        self.stage_limit = stage_deadline(self.stage_started, self.deadline, name)
        self.output.append({'event': 'stage_started', 'stage': name, 'deadline': self.stage_limit,
                            'timestamp': self.stage_started})
        try:
            self.check()
            yield self.stage_limit
        finally:
            self.output.append({'event': 'stage_ended', 'stage': name, 'timestamp': time.perf_counter()})


class IndependentGPUData(GPUData):
    def training_tape(self, batch):
        torch = self.torch
        ids = torch.tensor(batch.row_ids, dtype=torch.long, device='cuda')
        raw = self.images[ids]
        offsets = torch.tensor(batch.offsets, dtype=torch.long, device='cuda')
        flips = torch.tensor(batch.flips, dtype=torch.bool, device='cuda')
        padded = torch.nn.functional.pad(raw, (4, 4, 4, 4))
        rows = offsets[:, 0, None] + torch.arange(32, device='cuda')[None, :]
        columns = offsets[:, 1, None] + torch.arange(32, device='cuda')[None, :]
        positions = (rows[:, :, None] * 40 + columns[:, None, :]).reshape(len(ids), 1, 1024)
        cropped = padded.flatten(2).gather(2, positions.expand(-1, 3, -1)).reshape(-1, 3, 32, 32)
        return self.normalized(torch.where(flips[:, None, None, None], cropped.flip(-1), cropped)), self.labels[ids]


def evaluate(model, gpu, indices, *, deadline, label, test_data=None):
    """Persist each completed row prefix before the next deadline check.

    A killed/expired job leaves the latest independent NPZ and per-batch journal
    counts. Full metrics are returned only after all requested rows are durable.
    """
    import numpy as np
    runtime, torch = gpu.runtime, gpu.torch
    losses, predictions, targets, row_ids = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            runtime.check(deadline, evaluation=label, rows=start)
            ids = indices[start:start+128]
            device = gpu.images.device
            if test_data is None:
                rows = torch.as_tensor(ids, dtype=torch.long, device=device)
                x, y = gpu.normalized(gpu.images[rows]), gpu.labels[rows]
            else:
                images, labels = test_data
                raw = torch.from_numpy(np.ascontiguousarray(images[np.asarray(ids)])).permute(0, 3, 1, 2).contiguous().to(device)
                x, y = gpu.normalized(raw), torch.from_numpy(np.asarray(labels[np.asarray(ids)], dtype=np.int64)).to(device)
            logits = model(x)
            per_row = torch.nn.functional.cross_entropy(logits, y, reduction='none')
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(per_row).all()):
                raise ValueError('Nonfinite logits or cross entropy')
            losses.extend(per_row.double().cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
            row_ids.extend(int(x) for x in ids)
            content = io.BytesIO()
            np.savez(content, row_ids=np.asarray(row_ids, dtype=np.int64),
                     labels=np.asarray(targets, dtype=np.int64), predictions=np.asarray(predictions, dtype=np.int64),
                     cross_entropy=np.asarray(losses, dtype=np.float64))
            artifact = runtime.output.bytes(f'predictions/{label}.npz', content.getvalue())
            runtime.output.json(f'prediction_progress/{label}.json', {
                'status': 'complete' if len(row_ids) == len(indices) else 'partial',
                'completed_rows': len(row_ids), 'requested_rows': len(indices), **artifact})
            runtime.output.append({'event': 'evaluation_batch', 'label': label, 'batch_rows': len(ids),
                'completed_rows': len(row_ids), 'requested_rows': len(indices),
                'prediction_file': artifact['path'], 'prediction_sha256': artifact['sha256'],
                'timestamp': time.perf_counter()})
            runtime.check(deadline, evaluation=label, rows=len(row_ids))
    if not targets:
        raise ValueError('A complete nonempty evaluation role is required')
    correct = sum(a == b for a, b in zip(predictions, targets))
    return {'rows': len(targets), 'correct': correct, 'top1': correct/len(targets),
            'cross_entropy': sum(losses)/len(losses), 'prediction_file': artifact['path'],
            'prediction_sha256': artifact['sha256']}


def train_baseline(gpu, seeds, deadline):
    import numpy as np
    from fedcore.algorithm.pruning.sparse.cv_channels import make_cifar_resnet18
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
    runtime, torch, protocol = gpu.runtime, gpu.torch, gpu.runtime.protocol
    runtime.check(deadline)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seeds.initialization)
        model = make_cifar_resnet18()
    initial_fingerprint = inventory_fingerprint(capture_inventory(model))
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    runtime.output.json('initialization.json', {'seeds': asdict(seeds), 'fingerprint': initial_fingerprint})
    model.to('cuda').train().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=protocol.baseline_lr, momentum=protocol.momentum,
                                weight_decay=protocol.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    gradients, updates = set(), 0
    for epoch in range(80):
        order_seed = derived_seed('baseline_order', seeds.train_order, epoch)
        ordered = np.random.default_rng(order_seed).permutation(np.asarray(gpu.data.roles.train))
        for batch_index, start in enumerate(range(0, len(ordered), 128)):
            runtime.check(deadline, model=seeds.name, epoch=epoch+1, updates=updates)
            ids = ordered[start:start+128].tolist()
            augmentation_seed = derived_seed('baseline_augmentation', seeds.augmentation, epoch, batch_index)
            x, y = gpu.training(ids, augmentation_seed)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite baseline loss')
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError(f'Baseline missing finite gradient: {name}')
                gradients.add(name)
            runtime.check(deadline, updates=updates, phase='before_optimizer_step')
            optimizer.step()
            updates += 1
            runtime.output.append({'event': 'baseline_update', 'model': seeds.name, 'epoch': epoch+1,
                'update': updates, 'batch_index': batch_index, 'rows': len(ids),
                'order_seed': order_seed, 'augmentation_seed': augmentation_seed,
                'cross_entropy': float(loss.detach()), 'timestamp': time.perf_counter()})
            runtime.check(deadline, updates=updates)
        scheduler.step()
        runtime.output.append({'event': 'baseline_epoch', 'epoch': epoch+1, 'updates': updates,
                               'next_lr': scheduler.get_last_lr()[0], 'timestamp': time.perf_counter()})
    model.eval()
    metrics = evaluate(model, gpu, gpu.data.roles.search_validation, deadline=deadline, label='M0_Vsearch')
    changed = {name: not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters()}
    proof = {'completed_epochs': 80, 'updates': updates, 'initial_fingerprint': initial_fingerprint,
             'seeds': asdict(seeds), 'final_epoch': 80, 'changed_parameter_tensors': changed,
             'all_parameters_received_finite_gradients': set(changed) == gradients,
             'all_parameter_tensors_changed': all(changed.values())}
    accepted = (updates == 25040 and metrics['top1'] >= .90
                and proof['all_parameters_received_finite_gradients'] and proof['all_parameter_tensors_changed'])
    runtime.output.json('baseline_admission.json', {'validation': metrics, 'training': proof, 'accepted': accepted})
    if not accepted:
        raise PilotStop('baseline', 'baseline_quality_or_training_admission')
    return model, metrics, proof


def recover_no_extra_bn(model, gpu, tape, *, deadline, role):
    """The sole recovery implementation; deliberately has no calibration hook."""
    runtime, torch, protocol = gpu.runtime, gpu.torch, gpu.runtime.protocol
    if len(tape) != 200 or any(len(batch.row_ids) != 128 for batch in tape):
        raise ValueError('The frozen 200 x 128 recovery tape is required')
    runtime.check(deadline, recovery=role, updates=0)
    model.to('cuda').train().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=protocol.recovery_lr, momentum=protocol.momentum,
                                weight_decay=protocol.weight_decay, nesterov=True)
    for update, batch in enumerate(tape, 1):
        runtime.check(deadline, recovery=role, updates=update-1)
        x, y = gpu.training_tape(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite recovery loss')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Recovery missing finite gradients')
        runtime.check(deadline, recovery=role, updates=update-1, phase='before_optimizer_step')
        optimizer.step()
        runtime.output.append({'event': 'recovery_update', 'role': role, 'update': update,
            'rows': len(batch.row_ids), 'cross_entropy': float(loss.detach()), 'timestamp': time.perf_counter()})
        runtime.check(deadline, recovery=role, updates=update)
    model.eval()
    evidence = {'updates': 200, 'tape_sha256': digest(tape_payload(tape)),
                'optimizer': 'fresh_SGD', 'lr': .001, 'momentum': .9, 'nesterov': True,
                'weight_decay': .0005, 'grammar_recovery': 'NoRecovery',
                'bn_mode': 'inherited_state_ordinary_SGD_updates_only', 'extra_bn_passes': 0}
    model._fedcore_method_evidence = {**getattr(model, '_fedcore_method_evidence', {}),
                                     'caller_owned_recovery': evidence}
    return evidence


class IndependentEvaluator:
    """Public CV facade adapter with one recovered observation for both objectives."""
    def __init__(self, runtime, data, teacher, seeds):
        from .cv_sparse_pilot import cv_menu, fixed_prefix
        from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
        from fedcore.tools.registry.model_registry import ModelRegistry
        os.environ['FEDCORE_MODEL_REGISTRY_PATH'] = str(runtime.output.root / 'registry')
        ModelRegistry(auto_cleanup=False)
        self.runtime, self.protocol, self.output, self.data = runtime, runtime.protocol, runtime.output, data
        self.facade = LowRankModel({'device': 'cpu', 'fedcore_id': 'independent-m0-comparison'})
        self.gpu = IndependentGPUData(data, runtime)
        self.calibration = data.rows(data.roles.calibration)
        self.menu, self.prefix = cv_menu(), fixed_prefix(cv_menu())
        self.records = []
        self.seeds, self.tape = seeds, recovery_tape(data.roles.train, seeds)
        self.tape_sha256 = digest(tape_payload(self.tape))
        self.output.json('recovery_tape.json', {'sha256': self.tape_sha256, 'updates': tape_payload(self.tape)})
        self.bind_teacher(teacher)

    # These two public-facade adapters neither train nor calibrate BN. The old
    # Pilot.short/recover paths are intentionally never called.
    def bind_teacher(self, model):
        from .cv_sparse_pilot import Pilot
        Pilot.bind_teacher(self, model)

    def build(self, genotype):
        from .cv_sparse_pilot import Pilot
        return Pilot.build(self, genotype)

    def phase_event(self, strategy, phase):
        """Persist nested diagnostic phases; they are never added as overhead."""
        first, last = phase['started_at'], phase['completed_at']
        if (any(type(x) not in (float, int) or not math.isfinite(x) for x in (first, last))
                or last < first):
            raise ValueError('Finite ordered phase timestamps required')
        self.output.append({'event': 'search_phase', 'strategy': strategy, **phase,
                            'seconds': last-first})

    def numerical_callback(self, genotype, strategy, deadline):
        started, callback_id = time.perf_counter(), len(self.records)+1
        self.output.append({'event': 'search_callback_started', 'strategy': strategy,
                            'callback_id': callback_id, 'timestamp': started})
        try:
            return self.evaluate(genotype, strategy, deadline)
        finally:
            self.phase_event(strategy, {'phase': 'numerical_callback', 'callback_id': callback_id,
                'started_at': started, 'completed_at': time.perf_counter()})

    def evaluate(self, genotype, run, deadline):
        from .cv_sparse_pilot import solution_id
        from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
        from fedcore.algorithm.low_rank.cv_grammar import cv_genotype_payload
        from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
        from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
        from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
        from fedcore.algorithm.low_rank.compression_graph import _canonical
        key, started = solution_id(genotype), time.perf_counter()
        if key in {r['solution_id'] for r in self.records}:
            raise ValueError('Evaluator must never retrain a duplicate in the same run')
        record = {'run': run, 'attempt_id': len(self.records)+1, 'solution_id': key,
                  'genotype': cv_genotype_payload(genotype), 'status': 'started', 'charged_attempt': True,
                  'started_at': started, 'deadline': deadline, 'quality_accepted': False,
                  'recovery_updates': 0, 'tape_sha256': self.tape_sha256,
                  'evaluation_key': digest({'model': self.teacher_fingerprint, 'data': self.data.identity,
                    'genotype': key, 'tape': self.tape_sha256, 'protocol': self.protocol.sha256,
                    'environment': self.runtime.environment, 'source': self.runtime.source_identity})}
        self.output.append({'event': 'candidate_charged', **record})
        outcome = None
        try:
            self.runtime.check(deadline, run=run, attempt=record['attempt_id'])
            result = self.build(genotype)
            record['cpu_build_seconds'] = time.perf_counter()-started
            if type(result) is not CVWorkflowSuccess:
                disposition = classify_cv_candidate_failure(result)
                record.update(status='failed' if disposition == 'candidate_failure' else 'fatal',
                              failure_type=type(result).__name__, failure=str(result))
                if disposition != 'candidate_failure':
                    raise PilotFatal(f'Undeclared CV build failure: {result}')
                outcome = result
            else:
                self.output.json(f'workflows/{key}.json', _canonical(result.workflow))
                self.runtime.check(deadline, phase='after_cpu_build')
                recovery_started = time.perf_counter()
                proof = recover_no_extra_bn(result.model, self.gpu, self.tape, deadline=deadline, role=key)
                record.update(recovery_updates=200, recovery_proof=proof,
                              recovery_seconds=time.perf_counter()-recovery_started)
                evaluation_started = time.perf_counter()
                metrics = evaluate(result.model, self.gpu, self.data.roles.search_validation,
                                   deadline=deadline, label=f'{run}_{key}_Vsearch')
                record.update(metrics, evaluation_seconds=time.perf_counter()-evaluation_started)
                saved_at = time.perf_counter()
                artifact, restored = checkpoint(result.model, self.gpu, label=key, solution_id=key,
                                                phase='recovered_200_no_extra_bn', deadline=deadline)
                if artifact['tensor_bytes'] != result.evidence['storage_bytes']:
                    raise ValueError('Recovery/checkpoint changed physical model byte count')
                artifact.update(recovery_updates=200, recovery_tape_sha256=self.tape_sha256,
                                recovery_proof=proof, validation=metrics,
                                source_m0_fingerprint=self.teacher_fingerprint)
                record.update(checkpoint=artifact, tensor_bytes=artifact['tensor_bytes'],
                    save_reload_seconds=time.perf_counter()-saved_at,
                    cost_ratio=artifact['tensor_bytes']/self.space.source_storage_bytes,
                    storage_accepted=artifact['tensor_bytes'] <= self.byte_ceiling,
                    status='succeeded', quality_accepted=True)
                del restored, result
                outcome = GrammarFitness(record['cross_entropy'], record['cost_ratio'], quality_accepted=True)
        except PilotStop as error:
            record.update(status='incomplete', failure=str(error), failure_type='ResourceFailure')
            if error.reason != 'absolute_deadline':
                raise
            outcome = ResourceFailure('EvaluationDeadline', str(error))
        except Exception as error:
            record.update(status='fatal', failure=str(error), failure_type=type(error).__name__)
            raise PilotFatal(f'Unexpected candidate failure: {error}') from error
        finally:
            # This fingerprint includes every storage byte and BN buffer.
            if inventory_fingerprint(capture_inventory(self.teacher)) != self.teacher_fingerprint:
                record.update(status='fatal', failure='M0 mutated')
                self.output.append({'event': 'source_integrity_failure', **record})
                raise PilotFatal('Candidate evaluation mutated immutable M0')
            ended = time.perf_counter()
            record.update(ended_at=ended, wall_seconds=ended-started,
                completed_within_deadline=ended <= min(deadline, self.runtime.deadline, self.runtime.stage_limit))
            if record['status'] == 'succeeded' and not record['completed_within_deadline']:
                record['status'] = 'late'
            self.records.append(record)
            self.output.append({'event': 'candidate_ended', **record})
            self.output.json('attempts.json', self.records)
        # Journal overhead is part of the attempt, including for Random.
        if time.perf_counter() >= min(deadline, self.runtime.deadline, self.runtime.stage_limit):
            record['completed_within_deadline'] = False
            if record['status'] == 'succeeded':
                record['status'] = 'late'
            self.output.json('attempts.json', self.records)
            return ResourceFailure('EvaluationDeadline', 'Late observations are excluded; costs remain charged')
        return outcome

    def random_search(self, seed, deadline):
        started = time.perf_counter()
        self.output.append({'event': 'search_engine_started', 'strategy': 'random', 'timestamp': started})
        try:
            return self._random_search(seed, deadline)
        finally:
            self.output.append({'event': 'search_engine_ended', 'strategy': 'random',
                                'timestamp': time.perf_counter()})

    def _random_search(self, seed, deadline):
        from .cv_sparse_pilot import solution_id
        from fedcore.algorithm.low_rank.cv_grammar import generate_cv_genotype, cv_genotype_payload
        rng, budget = random.Random(seed), AttemptBudget(deadline)
        self.output.append({'event': 'search_engine_ready', 'strategy': 'random',
                            'timestamp': time.perf_counter()})
        while True:
            self.runtime.check(deadline)
            if budget.proposed >= 256 or len(budget.identities) >= 32:
                break
            proposal_started = time.perf_counter()
            genotype = (self.prefix[budget.proposed] if budget.proposed < 4
                        else generate_cv_genotype(self.space, seed=rng.randrange(2**32)))
            key = solution_id(genotype)
            budget, reservation = budget.reserve(key, time.perf_counter())
            if reservation is Reservation.STOPPED:
                break
            self.output.append({'event': 'proposal', 'proposal_id': budget.proposed,
                'solution_id': key, 'genotype': cv_genotype_payload(genotype),
                'cache_hit': reservation is Reservation.DUPLICATE,
                'operator': 'shared_prefix' if budget.proposed <= 4 else 'uniform_independent_sampling'})
            self.phase_event('random', {'phase': 'proposal_graph_creation', 'proposal_id': budget.proposed,
                'started_at': proposal_started, 'completed_at': time.perf_counter()})
            if reservation is Reservation.CHARGED:
                self.numerical_callback(genotype, 'random', deadline)
        return {'engine': 'uniform_random_over_declared_finite_CV_grammar',
                'charged_evaluations': len(budget.identities), 'created_proposals': budget.proposed}

    def evolution(self, seed, deadline):
        started = time.perf_counter()
        self.output.append({'event': 'search_engine_started', 'strategy': 'evolution', 'timestamp': started})
        try:
            return self._evolution(seed, deadline)
        finally:
            self.output.append({'event': 'search_engine_ended', 'strategy': 'evolution',
                                'timestamp': time.perf_counter()})

    def _evolution(self, seed, deadline):
        from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
        from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
        from fedcore.algorithm.low_rank.plans import Violation
        from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig, run_cv_grammar_search
        from .cv_sparse_pilot import _cv_search_report
        source_data_identity = self.data.identity

        def context_guard():
            if (inventory_fingerprint(capture_inventory(self.teacher)) != self.teacher_fingerprint
                    or self.data.identity != source_data_identity):
                return StructuralInvalid((Violation('IndependentContextChanged', 'independent_search',
                    'The immutable M0 or data identity changed'),))
            return None

        def trace_event(event):
            if event['event'] == 'started':
                self.output.append({'event': 'search_engine_ready', 'strategy': 'evolution',
                                    'timestamp': event['timestamp']})
            self.output.append({'event': 'evolution_trace', 'trace': event})

        result = run_cv_grammar_search(self.space,
            GrammarSearchConfig(seed=seed, population_size=4, generations=32,
                max_evaluations=32, max_proposals=256, max_seconds=720),
            lambda workflow: self.numerical_callback(workflow.genotype, 'evolution', deadline),
            fitness_context='4000 Vsearch CE after fixed 200 SGD lr=.001, inherited BN, no extra BN; physical bytes/own M0',
            initial_genotypes=self.prefix, absolute_deadline=deadline,
            charge_before_compile=True, context_guard=context_guard,
            trace_callback=trace_event,
            phase_callback=lambda e: self.phase_event('evolution', e))
        by_solution = {r['solution_id']: r for r in self.records}
        for event in result.get('trace', ()):
            if event.get('event') != 'evaluated':
                continue
            key = event['canonical_solution_id']
            if key in by_solution:
                record = by_solution[key]
                record['attempt_id'] = event['attempt_id']
                record['common_search_completed_at'] = event['ended_at']
                record['completed_within_deadline'] &= event['completed_within_deadline']
                if not record['completed_within_deadline'] and record['status'] == 'succeeded':
                    record['status'] = 'late'
            else:
                self.records.append({'attempt_id': event['attempt_id'], 'solution_id': key,
                    'genotype': event['genotype'], 'status': 'failed', 'charged_attempt': True,
                    'quality_accepted': False, 'recovery_updates': 0, 'failure': event['failure'],
                    'started_at': event['started_at'], 'ended_at': event['ended_at'],
                    'completed_within_deadline': event['completed_within_deadline'],
                    'failure_details': event['failure_details'], 'source': 'common_pre_callback_compilation'})
        self.records.sort(key=lambda r: r['attempt_id'])
        self.output.json('attempts.json', self.records)
        if result.get('status') == 'failed':
            raise PilotFatal(f'Optimizer failure: {result.get("reason")}')
        return _cv_search_report(result)
