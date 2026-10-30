"""CPU fakes exercise guard v2 decision -> owned stop -> durable receipt."""
from dataclasses import replace
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import psutil
import pytest

from fedcore.experiments.cv_independent_guard import (
    GuardEffects, MAX_RSS_BYTES, MAX_SECONDS, MAX_VRAM_BYTES, MINIMUM_DISK_BYTES,
    WORKER_SECONDS, run_guard,
)
from fedcore.experiments.cv_independent_supervision import NamespaceBinding, PidBinding, ProcessIdentity


class FakeTimer:
    def __init__(self, interval, callback):
        self.interval, self.callback, self.started, self.cancelled = interval, callback, False, False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


class Harness:
    def __init__(self, samples):
        self.samples, self.queries, self.launches, self.signals, self.providers = iter(samples), [], [], [], []
        self.now, self.returncode, self.finish_on_poll, self.finish_after = 100., None, False, None
        self.processes, self.timers = {}, []
        self.root = self.add_process(500)
        self.worker = self.add_process(501)
        self.foreign = self.add_process(900)
        self.root.descendants = [self.worker]

    def add_process(self, pid):
        harness = self

        class Process:
            def __init__(self):
                self.pid, self.birth, self.running, self.rss, self.zombie = pid, 1000.+pid, True, 1024, False
                self.descendants = []

            def create_time(self):
                return self.birth

            def status(self):
                return psutil.STATUS_ZOMBIE if self.zombie else (psutil.STATUS_RUNNING if self.running else psutil.STATUS_DEAD)

            def is_running(self):
                return self.running

            def memory_info(self):
                return SimpleNamespace(rss=self.rss)

            def children(self, recursive):
                assert recursive is True
                return self.descendants

            def send_signal(self, value):
                harness.signals.append((self.pid, self.birth, value))
                self.running = False
                if self.pid == 500:
                    harness.returncode = -9

        process = Process()
        self.processes[pid] = process
        return process

    def process(self, pid):
        return self.processes[pid]

    def gpu_rows(self):
        value = next(self.samples)
        self.queries.append(value)
        if isinstance(value, Exception):
            raise value
        return value

    def provider(self, rows, owned):
        self.providers.append((rows, owned))
        return NamespaceBinding(verified=True,
            entries=(PidBinding(80501, ProcessIdentity(501, 1501.)),), evidence='test external observer')

    def popen(self, command, **kwargs):
        self.launches.append((command, kwargs))
        harness = self

        class Child:
            pid = 500

            def poll(self):
                if harness.finish_on_poll or harness.finish_after is not None and len(harness.queries) >= harness.finish_after:
                    harness.returncode = 0
                    harness.root.running = False
                return harness.returncode

            def wait(self, timeout):
                assert timeout >= 0
                return harness.returncode

        return Child()

    def sleep(self, seconds):
        self.now += seconds

    def timer(self, seconds, callback):
        timer = FakeTimer(seconds, callback)
        self.timers.append(timer)
        return timer

    def effects(self):
        return GuardEffects(popen=self.popen, process=self.process, gpu_rows=self.gpu_rows,
            clock=lambda: self.now, sleep=self.sleep, timer=self.timer,
            disk_free=lambda path: MINIMUM_DISK_BYTES+1, guard_rss=lambda: 0)


def run(harness, tmp_path, *, provider=None, effects=None):
    receipt = tmp_path/'guard.json'
    result = run_guard(['cpu-noop-placeholder'], receipt_path=receipt,
        binding_provider=provider or harness.provider, effects=effects or harness.effects(),
        environment={'FEDCORE_PILOT_ABSOLUTE_DEADLINE': '99999999'})
    assert json.loads(receipt.read_text(encoding='utf-8')) == json.loads(json.dumps(result))
    return result


def test_missing_provider_creates_blocked_receipt_and_zero_children_without_gpu_query(tmp_path):
    harness = Harness([])
    effects = replace(harness.effects(), gpu_rows=lambda: pytest.fail('Missing provider queried GPU'))
    receipt = tmp_path/'blocked.json'
    result = run_guard(['cpu-noop-placeholder'], receipt_path=receipt, effects=effects)
    assert result['status'] == 'blocked' and result['reason'] == 'namespace_binding_provider_missing'
    assert result['spawned_child_count'] == 0 and harness.launches == []
    assert json.loads(receipt.read_text()) == result and harness.signals == []


def test_unverified_provider_blocks_even_empty_gpu_before_any_child(tmp_path):
    harness = Harness([()])
    result = run(harness, tmp_path, provider=lambda rows, owned: NamespaceBinding())
    assert result['status'] == 'blocked' and result['reason'] == 'gpu_identity_unverified'
    assert result['spawned_child_count'] == 0 and harness.launches == []


def test_multiple_preflight_rows_skip_provider_and_local_resolver_before_refusing_spawn(tmp_path):
    harness = Harness([('host1, 10', 'host2, 10')])
    effects = replace(harness.effects(), process=lambda pid: pytest.fail('Raw GPU PID was resolved'))
    result = run(harness, tmp_path, provider=lambda *a: pytest.fail('Multiple rows reached provider'), effects=effects)
    assert result['status'] == 'blocked' and result['reason'] == 'multiple_gpu_rows'
    assert harness.launches == [] and harness.signals == []


def test_multiple_runtime_rows_decide_stop_then_signal_only_owned_local_births_and_write_receipt(tmp_path):
    harness = Harness([(), ('80501, 10', '99999, 20')])
    effects = harness.effects()

    def checked_process(pid):
        if len(harness.queries) > 1:
            ledger = tmp_path/'guard.json.jsonl'
            assert ledger.is_file()
            assert json.loads(ledger.read_text().splitlines()[0])['reason'] == 'multiple_gpu_rows'
        return harness.process(pid)

    result = run(harness, tmp_path, effects=replace(effects, process=checked_process))
    assert result['status'] == 'stopped' and result['reason'] == 'multiple_gpu_rows'
    assert len(harness.providers) == 1  # preflight only, before launch
    assert {pid for pid, birth, sig in harness.signals} == {500, 501}
    assert all(sig == getattr(signal, 'SIGKILL', signal.SIGTERM) and birth == 1000.+pid
               for pid, birth, sig in harness.signals)
    assert result['cleanup']['live'] == result['cleanup']['unknown'] == []
    assert len(harness.timers) == 1 and harness.timers[0].cancelled


def test_one_unmapped_runtime_row_stops_after_valid_sample_without_querying_later_empty_row(tmp_path):
    harness = Harness([(), ('80501, 10',), ('99999, 20',), ()])
    result = run(harness, tmp_path)
    assert result['reason'] == 'gpu_identity_unverified' and result['status'] == 'stopped'
    assert result['telemetry_failures'] == 1 and len(harness.queries) == 3
    assert len(harness.launches) == 1 and {pid for pid, _, _ in harness.signals} == {500, 501}


def test_foreign_gpu_lifetime_is_never_a_cleanup_signal_target(tmp_path):
    harness = Harness([(), ('80900, 10',)])

    def provider(rows, owned):
        return NamespaceBinding(verified=True, entries=(PidBinding(80900, ProcessIdentity(900, 1900.)),),
                                evidence='test external observer')

    result = run(harness, tmp_path, provider=provider)
    assert result['reason'] == 'foreign_gpu_process'
    assert {pid for pid, _, _ in harness.signals} == {500, 501}
    assert harness.foreign.running and 80900 not in {pid for pid, _, _ in harness.signals}


def test_provider_losing_verification_is_terminal_even_when_gpu_is_empty(tmp_path):
    harness = Harness([(), ()])
    calls = []

    def provider(rows, owned):
        calls.append(rows)
        return (harness.provider(rows, owned) if len(calls) == 1 else NamespaceBinding())

    result = run(harness, tmp_path, provider=provider)
    assert result['status'] == 'stopped' and result['reason'] == 'gpu_identity_unverified'
    assert result['telemetry_failures'] == 1 and len(harness.launches) == 1


def test_verified_owned_gpu_sample_finishes_with_private_session_and_narrow_deadline(tmp_path):
    harness = Harness([(), ('80501, 10',)])
    harness.finish_on_poll = True
    result = run(harness, tmp_path)
    assert result['status'] == 'finished' and result['returncode'] == 0
    assert result['telemetry_failures'] == 0 and result['spawned_child_count'] == 1
    kwargs = harness.launches[0][1]
    assert kwargs['start_new_session'] is True
    assert float(kwargs['env']['FEDCORE_PILOT_ABSOLUTE_DEADLINE']) == 100.+28790
    assert (result['max_seconds'], result['worker_seconds'], result['max_rss_bytes']) == (28800, 28790, 32*1024**3)
    assert result['within_authorized_window'] and result['automatic_retry'] is False


@pytest.mark.parametrize('resource,reason', [('rss', 'tree_rss_limit'), ('vram', 'vram_limit')])
def test_live_bridge_enforces_frozen_memory_caps_and_stops_owned_tree(tmp_path, resource, reason):
    harness = Harness([(), ('80501, 17000',) if resource == 'vram' else ()])
    if resource == 'rss':
        harness.worker.rss = MAX_RSS_BYTES+1
    result = run(harness, tmp_path)
    assert result['status'] == 'stopped' and result['reason'] == reason
    assert {pid for pid, _, _ in harness.signals} == {500, 501}


def test_absolute_deadline_is_independent_of_gpu_query_success(tmp_path):
    harness = Harness([(), (), ()])

    def sleep(seconds):
        harness.now = 100.+WORKER_SECONDS

    result = run(harness, tmp_path, effects=replace(harness.effects(), sleep=sleep))
    assert result['reason'] == 'absolute_deadline' and result['status'] == 'stopped'
    assert result['within_authorized_window'] and len(harness.launches) == 1


@pytest.mark.parametrize('os_deadline,passed_deadline,expected', [
    ('122', '142', 122.), ('142', '122', 122.), ('122', None, 122.),
])
def test_earlier_external_absolute_deadline_is_preserved_in_timer_child_env_and_stop_receipt(
        tmp_path, monkeypatch, os_deadline, passed_deadline, expected):
    monkeypatch.setenv('FEDCORE_PILOT_ABSOLUTE_DEADLINE', os_deadline)
    harness = Harness([(), (), ()])

    def sleep(seconds):
        harness.now = expected

    environment = {} if passed_deadline is None else {'FEDCORE_PILOT_ABSOLUTE_DEADLINE': passed_deadline}
    result = run_guard(['cpu-noop-placeholder'], receipt_path=tmp_path/'guard.json',
        binding_provider=harness.provider, environment=environment,
        effects=replace(harness.effects(), sleep=sleep))
    assert result['absolute_deadline'] == expected and result['reason'] == 'absolute_deadline'
    assert float(harness.launches[0][1]['env']['FEDCORE_PILOT_ABSOLUTE_DEADLINE']) == expected
    assert harness.timers[0].interval == expected-100.


@pytest.mark.parametrize('external', ['nan', 'inf', '100', '99', 'not-a-deadline'])
def test_invalid_or_expired_external_deadline_blocks_before_spawn(tmp_path, monkeypatch, external):
    monkeypatch.delenv('FEDCORE_PILOT_ABSOLUTE_DEADLINE', raising=False)
    harness = Harness([()])
    result = run_guard(['cpu-noop-placeholder'], receipt_path=tmp_path/'guard.json',
        binding_provider=harness.provider, environment={'FEDCORE_PILOT_ABSOLUTE_DEADLINE': external},
        effects=harness.effects())
    assert result['status'] == 'blocked' and result['spawned_child_count'] == 0
    assert harness.launches == []


@pytest.mark.parametrize('gpu_still_reports_vanished', [False, True])
def test_finished_sequential_owned_child_may_leave_rss_but_never_remain_an_accepted_raw_gpu_identity(
        tmp_path, gpu_still_reports_vanished):
    next_row = '80501, 10' if gpu_still_reports_vanished else '80502, 10'
    harness = Harness([(), ('80501, 10',), (next_row,)])
    successor = harness.add_process(502)
    harness.finish_after = 3
    vanished = []

    def process(pid):
        if pid == 501 and vanished:
            raise psutil.NoSuchProcess(pid)
        return harness.process(pid)

    def sleep(seconds):
        vanished.append(True)
        harness.root.descendants = [successor]
        harness.now += seconds

    def provider(rows, owned):
        return NamespaceBinding(verified=True, evidence='test sequential host observer', entries=(
            PidBinding(80501, ProcessIdentity(501, 1501.)), PidBinding(80502, ProcessIdentity(502, 1502.))))

    result = run(harness, tmp_path, provider=provider,
        effects=replace(harness.effects(), process=process, sleep=sleep))
    if gpu_still_reports_vanished:
        assert result['status'] == 'stopped' and result['reason'] == 'gpu_identity_unverified'
        assert result['telemetry_failures'] == 1
    else:
        assert result['status'] == 'finished' and result['returncode'] == 0
        assert result['telemetry_failures'] == 0
    assert result['cleanup']['unknown'] == []
    assert any(row['owned']['pid'] == 501 for row in result['cleanup']['exited'])


def test_cleanup_receipt_separates_zombie_and_unknown_and_never_signals_them(tmp_path):
    harness = Harness([(), ('80501, 10', '99999, 20')])
    harness.root.descendants = []
    harness.root.zombie = True
    result = run(harness, tmp_path)
    assert result['cleanup']['zombie'][0]['owned']['pid'] == 500
    assert harness.signals == []
    unknown = Harness([(), ('80501, 10', '99999, 20')])
    process_calls = []

    def denied_after_attach(pid):
        process_calls.append(pid)
        if len(process_calls) > 1:
            raise psutil.AccessDenied(pid)
        return unknown.process(pid)

    result = run(unknown, tmp_path/'other', effects=replace(unknown.effects(), process=denied_after_attach))
    assert result['cleanup']['unknown'][0]['owned']['pid'] == 500
    assert unknown.signals == []


def test_freedisk_preflight_and_existing_receipt_never_launch_another_child(tmp_path):
    harness = Harness([()])
    effects = replace(harness.effects(), disk_free=lambda path: MINIMUM_DISK_BYTES-1)
    result = run(harness, tmp_path, effects=effects)
    assert result['status'] == 'blocked' and result['reason'] == 'less_than_100_GiB_free_disk'
    with pytest.raises(FileExistsError):
        run_guard(['cpu-noop-placeholder'], receipt_path=tmp_path/'guard.json',
                  binding_provider=harness.provider, effects=harness.effects())
    assert harness.launches == []
