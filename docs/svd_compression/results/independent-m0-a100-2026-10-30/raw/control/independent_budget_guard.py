"""External one-shot eight-hour boundary; owns only its descendant processes."""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, path)


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    result = []
    for row in output.strip().splitlines():
        if not row.strip():continue
        parts = row.split(',')
        if len(parts) != 2:raise RuntimeError('Unparseable GPU telemetry')
        result.append({'pid':int(parts[0]),'used_bytes':int(parts[1])*1024**2})
    return result


def classify_gpu_rows(rows, tree):
    # A numerical child can start between the tree and nvidia-smi snapshots,
    # or exit before the query returns. Resolve those races before classifying
    # a live PID as foreign; never claim ownership by PID alone.
    tree.capture()
    live=[]
    for row in rows:
        try:
            process=psutil.Process(row['pid'])
            if process.status()==psutil.STATUS_ZOMBIE:continue
        except psutil.NoSuchProcess:continue
        if not tree.owns(row['pid']):return 'foreign_gpu_process'
        live.append(row)
    return 'multiple_gpu_workers' if len(live)>1 else None


class OwnedTree:
    """Birth-checked descendants; private groups remain owned after reparenting."""
    def __init__(self):
        self.identities = {};self.groups = {};self.lock = threading.RLock()

    def capture(self):
        with self.lock:
            for process in psutil.Process().children(recursive=True):
                try:
                    birth = process.create_time()
                    self.identities[process.pid] = birth
                    if os.getpgid(process.pid) == process.pid:self.groups[process.pid] = birth
                except (psutil.NoSuchProcess, ProcessLookupError):pass
            live = []
            for pid, birth in list(self.identities.items()):
                try:
                    process = psutil.Process(pid)
                    if process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE:live.append(process)
                except psutil.NoSuchProcess:pass
            return live

    def owns(self, pid):
        try:return psutil.Process(pid).create_time() == self.identities.get(pid)
        except psutil.NoSuchProcess:return False

    def stop(self, child=None):
        with self.lock:
            for _ in range(3):
                live = self.capture()
                for pid, birth in list(self.groups.items()):
                    try:
                        process = psutil.Process(pid)
                        if process.create_time() == birth and os.getpgid(pid) == pid:os.killpg(pid, signal.SIGKILL)
                    except (ProcessLookupError, psutil.NoSuchProcess):pass
                # Also stop owned descendants that inherited another owned PGID.
                for process in live:
                    try:
                        if process.create_time() == self.identities.get(process.pid):process.kill()
                    except psutil.NoSuchProcess:pass
                if child is not None:
                    try:child.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
            for pid in list(self.identities):
                try:os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);args=parser.parse_args()
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Subreaper admission failed')
    packet=json.loads(args.packet.read_text(encoding='utf-8'));root=args.packet.parent
    assert packet['schema']=='fedcore-independent-cv-launch-v1' and packet['automatic_retry'] is False
    assert packet['max_seconds']==28800 and packet['worker_seconds']==28790 and packet['prior_seconds']==0
    assert packet['one_launch_authorized'] is True and digest(__file__)==packet['guard_sha256']
    assert digest(root/'source-manifest.json')==packet['source_manifest_sha256']
    began=packet['launch_monotonic'];assert 0<=time.perf_counter()-began<60
    output=Path(packet['output_root']);state_path=root/'results/series-1-budget-guard.json'
    assert not output.exists() and not state_path.exists()
    assert not gpu_processes(),'A100 occupied before worker launch'
    child=None;tree=OwnedTree();deadline_fired=threading.Event();timer=None
    state={'schema':'fedcore-independent-cv-guard-v1','status':'starting',
        'automatic_retry':False,'prior_seconds':0,'max_seconds':28800,'worker_seconds':28790,
        'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'peak_tree_rss_bytes':0,
        'peak_gpu_used_bytes':0,'subreaper_enabled':True,'telemetry_failures':0}
    atomic(state_path,state)
    def stop_deadline():
        deadline_fired.set();tree.stop(child)
    try:
        environment=dict(os.environ,**packet['environment'])
        with (root/'logs/series-1-console.log').open('x',encoding='utf-8') as console:
            child=subprocess.Popen(packet['command'],cwd=root,env=environment,stdin=subprocess.DEVNULL,
                stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',controller_pid=child.pid)
            timer=threading.Timer(max(0,began+28790-time.perf_counter()),stop_deadline);timer.daemon=True;timer.start()
            with (root/'logs/series-1-supervision.jsonl').open('x',encoding='utf-8') as log:
                missing=0
                while True:
                    code=child.poll();reason=None;rss=None;gpu=None
                    try:
                        live=tree.capture();rss=psutil.Process().memory_info().rss+sum(p.memory_info().rss for p in live)
                        gpu=gpu_processes();missing=0
                        gpu_reason=classify_gpu_rows(gpu,tree)
                        if gpu_reason:reason=gpu_reason
                        elif rss>32*1024**3:reason='tree_rss_limit'
                        state['peak_tree_rss_bytes']=max(state['peak_tree_rss_bytes'],rss)
                        state['peak_gpu_used_bytes']=max(state['peak_gpu_used_bytes'],sum(x['used_bytes'] for x in gpu))
                    except psutil.NoSuchProcess:
                        # A descendant can exit while the RSS sum is read. Keep
                        # the race visible and retry telemetry, not numerical work.
                        missing+=1
                    except (psutil.AccessDenied,OSError,ValueError,RuntimeError,subprocess.SubprocessError):missing+=1
                    elapsed=time.perf_counter()-began
                    state['telemetry_failures']+=int(rss is None or gpu is None)
                    if missing>=3:reason='telemetry_unavailable'
                    if elapsed>=28790 or deadline_fired.is_set():reason='absolute_deadline'
                    row={'elapsed_seconds':elapsed,'controller_returncode':code,'tree_rss_bytes':rss,
                         'gpu':gpu,'consecutive_telemetry_failures':missing,'stop_reason':reason}
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    state.update(elapsed_seconds=elapsed,last_observation=row);atomic(state_path,state)
                    if reason:
                        tree.stop(child);state.update(status='stopped',reason=reason);break
                    if code is not None:
                        state.update(status='finished',reason='controller_exit',returncode=code);break
                    time.sleep(.5)
    finally:
        if timer is not None:timer.cancel()
        tree.stop(child)
        elapsed=time.perf_counter()-began
        state.update(elapsed_seconds=elapsed,total_charged_seconds=elapsed,
                     within_authorized_window=elapsed<=28800,returncode=child.returncode if child else None,
                     remaining_owned_pids=[p.pid for p in tree.capture()])
        atomic(state_path,state)
    print(json.dumps(state,allow_nan=False),flush=True)
    return 0 if state['status']=='finished' and state['returncode']==0 else 1


if __name__=='__main__':raise SystemExit(main())
