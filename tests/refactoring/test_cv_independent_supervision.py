"""Fail-closed GPU identity and local birth-bound cleanup; CPU-only fakes."""
from dataclasses import replace
import signal

import psutil
import pytest

from fedcore.experiments.cv_independent_supervision import (
    CleanupState, GpuReason, LocalFailure, LocalObservation, LocalState,
    NamespaceBinding, PidBinding, ProcessIdentity, SignalDisposition,
    SupervisionState, advance_supervision, classify_gpu_rows, observe_gpu_rows,
    observe_owned_cleanup, resolve_local_process, signal_local_owned,
)


OWNED = ProcessIdentity(71, 100.)
FOREIGN = ProcessIdentity(91, 200.)


def verified_binding(*entries):
    return NamespaceBinding(verified=True, entries=tuple(entries),
                            evidence='explicit external host observer identity record')


def live(identity=OWNED):
    return LocalObservation(identity.pid, LocalState.LIVE, identity)


def forbidden_resolver(pid):
    pytest.fail(f'Unsafe local PID lookup was attempted: {pid}')


@pytest.mark.parametrize('rows', [
    ('81234, 100', '81235, 200'),
    ('unmapped garbage', 'another unparsed row'),
    ('81234, 100', '81234, 100'),
])
@pytest.mark.parametrize('binding', [NamespaceBinding(), verified_binding(PidBinding(81234, OWNED))])
def test_two_raw_rows_are_terminal_before_any_local_pid_parsing_or_resolution(rows, binding):
    result = classify_gpu_rows(rows, owned=(OWNED,), binding=binding, resolver=forbidden_resolver)
    assert result.reason is GpuReason.MULTIPLE_ROWS and result.terminal
    assert result.raw_rows == rows and result.local is None


def test_one_unmapped_row_is_terminal_before_local_probe_even_with_verified_binding():
    result = classify_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(), resolver=forbidden_resolver)
    assert result.reason is GpuReason.IDENTITY_UNVERIFIED
    assert result.reported_gpu_pid == 81234 and result.local is None


def test_default_binding_is_unverified_and_equal_integer_owned_pid_is_not_evidence():
    assert NamespaceBinding().verified is False
    binding = NamespaceBinding(entries=(PidBinding(71, OWNED),))
    result = classify_gpu_rows(('71, 100',), owned=(OWNED,), binding=binding, resolver=forbidden_resolver)
    assert result.reason is GpuReason.IDENTITY_UNVERIFIED and not result.namespace_binding_verified


def test_verified_explicit_mapping_resolves_local_pid_instead_of_reported_host_pid():
    calls = []

    def resolver(pid):
        calls.append(pid)
        return live()

    result = classify_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(81234, OWNED)), resolver=resolver)
    assert result.reason is GpuReason.OWNED and not result.terminal
    assert result.reported_gpu_pid == 81234 and result.local.identity == OWNED
    assert calls == [71]


@pytest.mark.parametrize('rows', [(), ('81234, 100', '81235, 200'), ('81235, 100',), ('invalid, 100',)])
@pytest.mark.parametrize('foreign', [False, True])
def test_public_verified_observation_cannot_forge_raw_row_count_or_pid(rows, foreign):
    identity = FOREIGN if foreign else OWNED
    observation = classify_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(81234, identity)), resolver=lambda pid: live(identity))
    assert observation.reason is (GpuReason.FOREIGN if foreign else GpuReason.OWNED)
    with pytest.raises(ValueError):
        replace(observation, raw_rows=rows)


def test_changed_local_birth_cannot_reuse_an_old_owned_pid():
    result = classify_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(81234, OWNED)),
        resolver=lambda pid: live(ProcessIdentity(pid, OWNED.birth+1)))
    assert result.reason is GpuReason.IDENTITY_UNVERIFIED and result.terminal


@pytest.mark.parametrize('state,failure', [
    (LocalState.EXITED, LocalFailure.NO_SUCH_PROCESS),
    (LocalState.UNKNOWN, LocalFailure.ACCESS_DENIED),
    (LocalState.ZOMBIE, LocalFailure.ZOMBIE),
    (LocalState.UNKNOWN, LocalFailure.BIRTH_CHANGED),
])
def test_non_live_mapping_never_becomes_ignored_or_empty(state, failure):
    observation = LocalObservation(OWNED.pid, state, failure=failure)
    result = classify_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(81234, OWNED)), resolver=lambda pid: observation)
    assert result.reason is GpuReason.IDENTITY_UNVERIFIED
    assert result.raw_rows == ('81234, 100',) and result.local == observation


@pytest.mark.parametrize('error', [psutil.NoSuchProcess(71), psutil.AccessDenied(71), psutil.ZombieProcess(71)])
def test_resolver_exceptions_are_terminal_unverified_not_continue(error):
    def resolver(pid):
        raise error

    result = classify_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(81234, OWNED)), resolver=resolver)
    assert result.reason is GpuReason.IDENTITY_UNVERIFIED and result.terminal


def test_verified_foreign_process_is_terminal_and_cannot_enter_owned_cleanup():
    result = classify_gpu_rows(('91234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(91234, FOREIGN)), resolver=lambda pid: live(FOREIGN))
    assert result.reason is GpuReason.FOREIGN
    signals = []
    observed = signal_local_owned(FOREIGN, owned=(OWNED,),
        process_factory=lambda pid: signals.append(pid) or pytest.fail('Foreign process was attached for signalling'))
    assert observed.disposition is SignalDisposition.NOT_OWNED and signals == []


def test_terminal_reason_is_sticky_after_a_later_empty_gpu_sample():
    first = observe_gpu_rows(('hostpid, 100',), owned=(OWNED,), resolver=forbidden_resolver)
    assert first.state.terminal_reason is GpuReason.IDENTITY_UNVERIFIED
    second = observe_gpu_rows((), state=first.state, owned=(OWNED,), resolver=forbidden_resolver)
    assert second.observation.reason is GpuReason.EMPTY
    assert second.state.terminal_reason is GpuReason.IDENTITY_UNVERIFIED and not second.may_continue
    assert second.state.telemetry_failures == first.state.telemetry_failures == 1


def test_resolver_access_denied_increments_and_never_resets_existing_telemetry_failures():
    def denied(pid):
        raise psutil.AccessDenied(pid)

    decision = observe_gpu_rows(('81234, 100',), owned=(OWNED,),
        binding=verified_binding(PidBinding(81234, OWNED)), resolver=denied,
        state=SupervisionState(telemetry_failures=2))
    assert not decision.may_continue
    assert decision.state.telemetry_failures == 3
    assert decision.state.terminal_reason is GpuReason.IDENTITY_UNVERIFIED


def test_only_successful_active_observation_resets_telemetry_failure_counter():
    empty = classify_gpu_rows((), resolver=forbidden_resolver)
    assert advance_supervision(SupervisionState(telemetry_failures=2), empty).telemetry_failures == 0
    missing = classify_gpu_rows(None, resolver=forbidden_resolver)
    stopped = advance_supervision(SupervisionState(telemetry_failures=2), missing)
    assert stopped.telemetry_failures == 3 and stopped.terminal_reason is GpuReason.TELEMETRY_UNAVAILABLE


@pytest.mark.parametrize('row', ['N/A, 100', '-5, 100', '0, 100', '１２３, 100', 'pid, used_memory'])
def test_malformed_nonempty_gpu_row_is_not_empty_or_an_integer_guess(row):
    observation = classify_gpu_rows((row,), owned=(OWNED,), binding=verified_binding(), resolver=forbidden_resolver)
    assert observation.reason is GpuReason.IDENTITY_UNVERIFIED and observation.terminal


class FakeProcess:
    def __init__(self, *, birth=100., status=psutil.STATUS_RUNNING, running=True):
        self.birth, self.current_status, self.running = birth, status, running
        self.signals = []

    def create_time(self):
        return self.birth

    def status(self):
        return self.current_status

    def is_running(self):
        return self.running

    def send_signal(self, value):
        self.signals.append(value)


@pytest.mark.parametrize('error,expected_state,expected_failure', [
    (psutil.NoSuchProcess(71), LocalState.EXITED, LocalFailure.NO_SUCH_PROCESS),
    (psutil.AccessDenied(71), LocalState.UNKNOWN, LocalFailure.ACCESS_DENIED),
    (psutil.ZombieProcess(71), LocalState.ZOMBIE, LocalFailure.ZOMBIE),
])
def test_psutil_boundary_retains_vanished_denied_and_zombie_as_typed_observations(error, expected_state, expected_failure):
    def factory(pid):
        raise error

    result = resolve_local_process(71, process_factory=factory)
    assert result.state is expected_state and result.failure is expected_failure


def test_local_probe_never_accepts_birth_that_changes_inside_the_snapshot():
    process = FakeProcess()
    births = iter((100., 101.))
    process.create_time = lambda: next(births)
    result = resolve_local_process(71, process_factory=lambda pid: process)
    assert result.state is LocalState.UNKNOWN and result.failure is LocalFailure.BIRTH_CHANGED


def test_cleanup_keeps_live_zombie_unknown_exited_and_reused_lifetimes_separate():
    identities = tuple(ProcessIdentity(pid, 100.) for pid in (71, 72, 73, 74, 75))
    observations = {
        71: live(identities[0]),
        72: LocalObservation(72, LocalState.ZOMBIE, identities[1], LocalFailure.ZOMBIE),
        73: LocalObservation(73, LocalState.UNKNOWN, failure=LocalFailure.ACCESS_DENIED),
        74: LocalObservation(74, LocalState.EXITED, failure=LocalFailure.NO_SUCH_PROCESS),
        75: live(ProcessIdentity(75, 101.)),
    }
    result = observe_owned_cleanup(identities, resolver=observations.__getitem__)
    assert tuple(row.state for row in result) == (
        CleanupState.LIVE, CleanupState.ZOMBIE, CleanupState.UNKNOWN, CleanupState.EXITED, CleanupState.PID_REUSED)
    assert tuple(row.owned for row in result if row.may_signal) == (identities[0],)


@pytest.mark.parametrize('process', [
    FakeProcess(birth=101.), FakeProcess(status=psutil.STATUS_ZOMBIE), FakeProcess(running=False),
])
def test_cleanup_signal_skips_changed_zombie_or_exited_owned_pid(process):
    result = signal_local_owned(OWNED, owned=(OWNED,), process_factory=lambda pid: process)
    assert result.disposition is SignalDisposition.SKIPPED and process.signals == []


def test_cleanup_signal_targets_only_freshly_verified_local_owned_pid():
    process, calls = FakeProcess(), []

    def factory(pid):
        calls.append(pid)
        return process

    result = signal_local_owned(OWNED, owned=(OWNED,), signal_number=signal.SIGTERM, process_factory=factory)
    assert result.disposition is SignalDisposition.SENT
    assert calls == [OWNED.pid] and process.signals == [signal.SIGTERM]
    assert result.cleanup.state is CleanupState.LIVE and result.cleanup.may_signal


def test_bare_reported_gpu_pid_cannot_be_used_as_signal_target():
    with pytest.raises(ValueError, match='captured local process lifetime'):
        signal_local_owned(81234, owned=(OWNED,), process_factory=forbidden_resolver)


def test_cleanup_send_failure_is_explicit_and_never_converted_to_success():
    process = FakeProcess()
    process.send_signal = lambda value: (_ for _ in ()).throw(psutil.AccessDenied(71))
    result = signal_local_owned(OWNED, owned=(OWNED,), process_factory=lambda pid: process)
    assert result.disposition is SignalDisposition.FAILED
    assert result.cleanup.state is CleanupState.UNKNOWN
    assert result.cleanup.local.failure is LocalFailure.ACCESS_DENIED


def test_namespace_binding_never_self_verifies_from_entries_or_ambiguous_gpu_pid():
    assert not NamespaceBinding(entries=(PidBinding(71, OWNED),)).verified
    with pytest.raises(ValueError, match='external evidence'):
        NamespaceBinding(verified=True, entries=(PidBinding(71, OWNED),))
    with pytest.raises(ValueError, match='unique mappings'):
        verified_binding(PidBinding(81234, OWNED), PidBinding(81234, FOREIGN))
    with pytest.raises(ValueError):
        replace(OWNED, birth=float('nan'))
