"""Exercise the real guard main on sleeping CPU children, never an ML worker."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
source=Path(__file__).with_name('independent_budget_guard.py')
for case in ('deadline','telemetry_failure','foreign_process'):
    spec=importlib.util.spec_from_file_location('probe_guard_'+case,source)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    with tempfile.TemporaryDirectory(prefix='fedcore-independent-guard-main-') as name:
        root=Path(name);(root/'logs').mkdir();(root/'results').mkdir()
        (root/'source-manifest.json').write_text('{}\n')
        calls=[0]
        def gpu():
            calls[0]+=1
            if calls[0]==1:return []
            if case=='telemetry_failure':raise RuntimeError('injected unavailable telemetry')
            if case=='foreign_process':return [{'pid':guard.os.getpid(),'used_bytes':1}]
            return []
        guard.gpu_processes=gpu
        began=time.perf_counter()
        if case=='deadline':
            ticks=[0]
            def clock():
                ticks[0]+=1
                return began if ticks[0]<=3 else began+28791
            guard.time=SimpleNamespace(perf_counter=clock,sleep=time.sleep)
        packet={'schema':'fedcore-independent-cv-launch-v1','automatic_retry':False,
            'max_seconds':28800,'worker_seconds':28790,'prior_seconds':0,'one_launch_authorized':True,
            'guard_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_manifest_sha256':hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest(),
            'launch_monotonic':began,'output_root':str(root/'results/unused-output'),
            'environment':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
            'command':[sys.executable,'-c','import time;time.sleep(30)']}
        path=root/'launch-packet.json';path.write_text(json.dumps(packet))
        old_argv=sys.argv
        try:
            sys.argv=['independent_budget_guard.py',str(path)];code=guard.main()
        finally:sys.argv=old_argv
        result=json.loads((root/'results/series-1-budget-guard.json').read_text())
        expected={'deadline':'absolute_deadline','telemetry_failure':'telemetry_unavailable',
                  'foreign_process':'foreign_gpu_process'}[case]
        assert code==1 and result['status']=='stopped' and result['reason']==expected,result
        assert result['remaining_owned_pids']==[]
        print(json.dumps({'case':case,'expected_stop':expected,'owned_tree_clean':True,
                          'real_seconds':time.perf_counter()-began}),flush=True)
print(json.dumps({'cpu_only':True,'GPU_queries':0,'full_guard_main_cases':3}))
