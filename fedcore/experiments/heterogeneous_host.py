"""Explicit, bounded host binding for the existing fail-closed CV guard.

Nothing is discovered or contacted at import. The caller supplies an authenticated
host-side command and an exact Docker container ID. This is not launch approval.
Host evidence is never inferred from equal PID numbers inside a container.
"""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time

from .cv_independent_supervision import NamespaceBinding, PidBinding, ProcessIdentity


def query_device_gpu_rows(device_uuid):
    """Explicit device scope: other GPUs on a shared host are not this job's GPU."""
    if (type(device_uuid) is not str or re.fullmatch(
            r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',device_uuid) is None):
        raise ValueError('An explicit stable GPU UUID is required; numeric indices are not inferred')
    result = subprocess.run(['nvidia-smi','--id='+device_uuid,
        '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=2,shell=False)
    if len(result.stdout) > 65536: raise RuntimeError('GPU process telemetry exceeds its limit')
    return tuple(row for row in result.stdout.splitlines() if row.strip())


def _run_bounded(command, payload, timeout):
    """Bound memory for both pipes; parent deadline also covers writer stalls.

    The provider's private process and observed descendants are lifetime-checked
    before cleanup. An OS cgroup deadline must enclose the full guard/observer
    tree: this transport alone cannot certify remote or escaping descendants.
    """
    import psutil
    data = payload.encode('utf-8')
    if len(data) > 65536:
        raise ValueError('Host request exceeds the bounded transport limit')
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, shell=False)
    owned = {}
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    failed = threading.Event()

    def read(stream, key, maximum):
        try:
            while True:
                block = stream.read(4096)
                if not block: break
                remaining = maximum - len(buffers[key])
                buffers[key].extend(block[:remaining])
                if len(block) > remaining:
                    failed.set(); break
        except (OSError,ValueError): failed.set()

    def write():
        try: child.stdin.write(data); child.stdin.close()
        except (OSError,ValueError): failed.set()

    workers = [threading.Thread(target=read,args=(child.stdout,'stdout',65536),daemon=True),
               threading.Thread(target=read,args=(child.stderr,'stderr',4096),daemon=True),
               threading.Thread(target=write,daemon=True)]
    for worker in workers: worker.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                root = psutil.Process(child.pid)
                for process in (root, *root.children(recursive=True)):
                    birth = process.create_time()
                    if process.pid in owned and owned[process.pid] != birth:
                        raise RuntimeError('Host transport PID was reused')
                    owned[process.pid] = birth
            except psutil.NoSuchProcess: pass
            if failed.is_set(): raise RuntimeError('Host transport exceeded output limit or pipe failed')
            if time.monotonic() >= deadline: raise TimeoutError('Bounded host observer timeout')
            if child.poll() is not None and not any(w.is_alive() for w in workers): break
            time.sleep(.005)
        if child.returncode:
            raise RuntimeError('Host observer returned nonzero status: ' + buffers['stderr'].decode('utf-8','replace'))
        return buffers['stdout'].decode('utf-8')
    finally:
        for pid, birth in reversed(tuple(owned.items())):
            try:
                process = psutil.Process(pid)
                if process.create_time() == birth and process.is_running(): process.kill()
            except psutil.NoSuchProcess: pass
        if child.poll() is None: child.kill()
        try: child.wait(timeout=.5)
        except subprocess.TimeoutExpired: pass
        for worker in workers: worker.join(timeout=.1)


@dataclass(frozen=True)
class HostRequest:
    container_id: str
    nonce: str
    rows: tuple[str, ...]
    owned: tuple[ProcessIdentity, ...]

    def __post_init__(self):
        if (type(self.container_id) is not str or re.fullmatch('[a-f0-9]{12,64}', self.container_id) is None
                or type(self.nonce) is not str or re.fullmatch('[a-f0-9]{32}', self.nonce) is None
                or type(self.rows) is not tuple or any(type(r) is not str for r in self.rows)
                or type(self.owned) is not tuple or any(type(p) is not ProcessIdentity for p in self.owned)):
            raise ValueError('Require exact container ID, request nonce, raw rows and owned identities')


@dataclass(frozen=True)
class BoundedBindingProvider:
    """The command must already be trusted; no shell interpolation or retries."""
    command: tuple[str, ...]
    container_id: str
    timeout: float = 3.

    def __post_init__(self):
        HostRequest(self.container_id, '0' * 32, (), ())
        if (type(self.command) is not tuple or not self.command
                or any(type(c) is not str or not c for c in self.command)
                or type(self.timeout) not in (int,float) or not math.isfinite(self.timeout)
                or not 0 < self.timeout <= 5):
            raise ValueError('Require an explicit command and a 0..5 second probe timeout')

    def __call__(self, rows, owned):
        request = HostRequest(self.container_id, secrets.token_hex(16), tuple(rows), tuple(owned))
        began = time.monotonic()
        result = _run_bounded(self.command, json.dumps(asdict(request)), self.timeout)
        if time.monotonic() - began > self.timeout or len(result) > 65536:
            raise RuntimeError('Host evidence is late or exceeds the bounded response size')
        response = json.loads(result)
        if (type(response) is not dict or set(response) !=
                {'schema','container_id','nonce','rows','entries','verified','evidence'}
                or response['schema'] != 'fedcore-host-binding-v1'
                or response['container_id'] != request.container_id or response['nonce'] != request.nonce
                or response['rows'] != list(request.rows) or response['verified'] is not True
                or type(response['entries']) is not list or len(response['entries']) > 1
                or type(response['evidence']) is not str or not response['evidence']):
            raise RuntimeError('Untrusted, replayed or mismatched host response')
        entries = []
        for row in response['entries']:
            if type(row) is not dict or set(row) != {'gpu_pid','local'} or type(row['local']) is not dict or set(row['local']) != {'pid','birth'}:
                raise RuntimeError('Malformed host PID lifetime evidence')
            binding = PidBinding(row['gpu_pid'], ProcessIdentity(**row['local']))
            if binding.local not in request.owned:
                raise RuntimeError('A host mapping does not belong to the launched local tree')
            entries.append(binding)
        reported = tuple(int(row.split(',')[0].strip()) for row in request.rows if row.strip())
        if tuple(e.gpu_pid for e in entries) != reported:
            raise RuntimeError('Every raw GPU process requires a fresh owned host binding')
        return NamespaceBinding(True, tuple(entries), response['evidence'])


def collect_host_binding(request, *, proc_root='/proc', run=subprocess.run, process_birth=None):
    """Run explicitly ON THE HOST. Empty or foreign mappings never grant ownership.

    Docker identifies the container's host init PID; host procfs establishes its
    namespace, NSpid establishes the local PID and birth protects against reuse.
    The outer authenticated command timeout bounds this whole probe.
    """
    if type(request) is not HostRequest:
        raise ValueError('Require a validated host request')
    root = Path(proc_root)
    result = run(['docker','inspect','--format','{{.State.Pid}}', request.container_id],
                 capture_output=True, text=True, check=True, timeout=1, shell=False)
    init = result.stdout.strip()
    if not init.isascii() or not init.isdecimal() or int(init) < 1:
        raise RuntimeError('Container must be running and visible to the host observer')
    namespace = os.readlink(root / init / 'ns/pid')
    if namespace == os.readlink(root / 'self/ns/pid'):
        raise RuntimeError('Shared/host PID containers are outside this binding profile')
    if process_birth is None:
        import psutil
        process_birth = lambda pid: psutil.Process(pid).create_time()
    entries = []
    raw = tuple(row for row in request.rows if row.strip())
    if len(raw) > 1:
        raise RuntimeError('Multiple raw GPU processes are forbidden by this profile')
    for row in raw:
        fields = row.split(',')
        if len(fields) != 2 or not fields[0].strip().isascii() or not fields[0].strip().isdecimal():
            raise RuntimeError('Malformed raw GPU row')
        host_pid = int(fields[0])
        before = process_birth(host_pid)
        if os.readlink(root / str(host_pid) / 'ns/pid') != namespace:
            raise RuntimeError('GPU process belongs to a different PID namespace')
        status = (root / str(host_pid) / 'status').read_text(encoding='ascii')
        line = next((line for line in status.splitlines() if line.startswith('NSpid:')), '')
        ids = line.split()[1:]
        if len(ids) != 2 or not all(v.isdecimal() for v in ids) or int(ids[0]) != host_pid:
            raise RuntimeError('Require exact host/container NSpid mapping, not nested namespaces')
        after = process_birth(host_pid)
        identity = next((p for p in request.owned if p.pid == int(ids[-1]) and p.birth == before), None)
        if identity is None or before != after:
            raise RuntimeError('Host process birth does not match a live owned local lifetime')
        entries.append({'gpu_pid': host_pid, 'local': asdict(identity)})
    # Recheck init namespace after probing; an init restart cannot silently rebind.
    if os.readlink(root / init / 'ns/pid') != namespace:
        raise RuntimeError('Container namespace changed during observation')
    return dict(schema='fedcore-host-binding-v1', container_id=request.container_id,
        nonce=request.nonce, rows=list(request.rows), entries=entries, verified=True,
        evidence='authenticated host Docker init + procfs PID namespace + NSpid + exact process births')


def main():
    import sys
    raw = sys.stdin.read(65537)
    if len(raw) > 65536:
        raise ValueError('Host request exceeds limit')
    value = json.loads(raw)
    if type(value) is not dict or set(value) != {'container_id','nonce','rows','owned'}:
        raise ValueError('Exact host request fields are required')
    request = HostRequest(value['container_id'], value['nonce'], tuple(value['rows']),
                          tuple(ProcessIdentity(**p) for p in value['owned']))
    print(json.dumps(collect_host_binding(request), allow_nan=False))


if __name__ == '__main__':
    main()
