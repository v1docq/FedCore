"""Audit this interrupted series from immutable files; no ML/CUDA imports.

Run with Python 3.10+ from any directory. This is an evidence replay, never a
numerical experiment/resume command. It writes only analysis/audit.json and
analysis/reviewed-report.json alongside this script.
"""
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'

def read(name):return json.loads((RAW/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def journal(name):
    rows=[]
    for number,line in enumerate((RAW/name).read_text().splitlines(),1):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError as error:raise ValueError(f'{name}:{number}: torn evidence') from error
    return rows

def audit():
    packet=read('control/launch-packet.json');guard=read('control/series-1-budget-guard.json')
    closure=read('control/interruption-closure.json');budget=read('control/manual-export-budget.json')
    manifest=read('control/source-manifest.json');protocol=read('protocol.json')
    inventory=read('independent-file-inventory.json')['files']
    for name,meta in inventory.items():
        path=RAW/name
        assert path.resolve().is_relative_to(RAW.resolve()) and path.is_file(),name
        assert sha(path)==meta['sha256'] and path.stat().st_size==meta['bytes'],name
    for name,value in packet['external_files_sha256'].items():assert sha(RAW/'control'/name)==value,name
    assert sha(RAW/'control/source-manifest.json')==packet['source_manifest_sha256']
    assert manifest['source_commit']==packet['source_commit']=='a23bc9ecf43d05e468ecf40979a5425873772b18'
    canonical=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert canonical(protocol)==manifest['protocol_sha256']==packet['protocol_sha256']
    roles=read('workers/manifest/data_roles.json')
    assert roles==read('workers/baseline_M1/data_roles.json')
    expected={'train':40000,'calibration':1000,'search_validation':4000,'report_validation':5000}
    seen=set()
    for name,count in expected.items():
        values=roles[name];assert len(values)==len(set(values))==count
        assert not seen.intersection(values);seen.update(values)
    assert seen==set(range(50000)) and roles['seed']==140100
    assert canonical(roles)==read('data_manifest.json')['roles_sha256']
    master=journal('ledger.jsonl');worker=journal('workers/baseline_M1/ledger.jsonl')
    updates=[r for r in worker if r['event']=='baseline_update']
    epochs=[r for r in worker if r['event']=='baseline_epoch']
    assert [r['update'] for r in updates]==list(range(1,len(updates)+1))
    assert [r['epoch'] for r in epochs]==list(range(1,len(epochs)+1))
    for r in updates:
        assert r['epoch']==(r['update']-1)//313+1
        assert r['batch_index']==(r['update']-1)%313
        assert r['rows']==(64 if r['batch_index']==312 else 128)
        assert math.isfinite(r['cross_entropy']) and r['cross_entropy']>=0
        assert r['model']=='M1'
    assert all(r['updates']==313*r['epoch'] for r in epochs)
    assert len(updates)==closure['confirmed_counts']['baseline_M1']['baseline_update']==3521
    assert len(epochs)==11 and updates[-1]['epoch']==12
    assert all(a['timestamp']<b['timestamp'] for a,b in zip(updates,updates[1:]))
    event_counts=Counter(r['event'] for r in master+worker+journal('workers/manifest/ledger.jsonl'))
    for name in ('candidate_charged','recovery_update','latency_pair','evaluation_batch','evaluation_progress'):
        assert event_counts[name]==0,name
    assert not (RAW/'global_seal.json').exists() and not list(RAW.rglob('*.npz')) and not list(RAW.rglob('*.pt'))
    assert {r['worker'] for r in master if r['event']=='child_started'}=={'manifest','baseline'}
    assert guard['returncode']==-15 and guard['within_authorized_window']
    assert closure['remaining_live_owned_processes']==0
    assert closure['new_numerical_launches']==0
    assert budget['additional_numerical_launches']==0 and budget['manual_close_within_14_minutes'] and budget['within_original_8_hours']
    assert not budget['outer_guard_covered_export']
    supervision=journal('control/series-1-supervision.jsonl')
    multi=[r for r in supervision if r.get('gpu') is not None and len(r['gpu'])>1]
    assert multi and multi[0]['elapsed_seconds']==closure['first_multiple_gpu_seconds']
    assert len(multi)==closure['multiple_gpu_observations']==173
    assert all(r['stop_reason'] is None for r in multi)
    preservation=read('control/postrun-preservation.json')
    assert preservation['old_files_verified']==1762 and preservation['new_package_files_verified']==325 and preservation['new_wheel_verified']
    installed=read('control/installed-checks-receipt.json')
    assert installed['installed_cpu_checks_passed'] and installed['dependencies_match'] and not installed['gpu_started']
    assert '468 passed' in (RAW/'control/installed-cpu-tests.log').read_text()
    launch=packet['launch_monotonic']
    progress={int(r['timestamp']-launch):r for r in updates}
    progress_rows=[{'seconds':r['timestamp']-launch,'minutes':(r['timestamp']-launch)/60,
        'updates':r['update'],'epoch':r['epoch'],'training_ce':r['cross_entropy']} for r in progress.values()]
    telemetry_rows=[{'seconds':r['elapsed_seconds'],'minutes':r['elapsed_seconds']/60,
        'gpu_rows':len(r['gpu']) if r.get('gpu') is not None else None,
        'gpu_total_mib':sum(x['used_bytes'] for x in r['gpu'])/1024**2 if r.get('gpu') is not None else None,
        'tree_rss_mib':r['tree_rss_bytes']/1024**2 if r.get('tree_rss_bytes') is not None else None} for r in supervision]
    phases=[{'stage':'Разбиение и допуск','status':'Завершено','done':1,'planned':1,'unit':'стадия'},
        {'stage':'Независимые M0','status':'M1 прервана; M2/M3 не начаты','done':0,'planned':3,'unit':'готовые модели'},
        {'stage':'Обновления исходных моделей','status':'Подтверждённый префикс','done':len(updates),'planned':75120,'unit':'обновления SGD'},
        {'stage':'Фиксированные контроли','status':'Не начаты','done':0,'planned':12,'unit':'траектории'},
        {'stage':'Random / Evo','status':'Не начаты','done':0,'planned':18,'unit':'поиски'},
        {'stage':'Оплаченные кандидаты','status':'Не начаты','done':0,'planned':576,'unit':'попытки'},
        {'stage':'Восстановление','status':'Не начато','done':0,'planned':117600,'unit':'обновления SGD'},
        {'stage':'Измерения скорости','status':'Не начаты','done':0,'planned':7560,'unit':'пары; верхний предел'}]
    summary={'status':'interrupted_infrastructure','source_commit':manifest['source_commit'],
        'protocol_sha256':manifest['protocol_sha256'],'wheel_sha256':manifest['wheel_sha256'],
        'launch_utc':packet['launch_utc'],'guard_seconds':guard['elapsed_seconds'],
        'launch_to_export_closed_seconds':budget['launch_to_export_closed_seconds'],
        'manual_close_seconds':budget['manual_close_after_guard_seconds'],
        'first_multiple_gpu_seconds':multi[0]['elapsed_seconds'],
        'guard_stop_delay_from_multiple_seconds':guard['elapsed_seconds']-multi[0]['elapsed_seconds'],
        'multiple_gpu_observations':len(multi),'confirmed_baseline_updates':len(updates),
        'completed_epochs_M1':len(epochs),'last_started_epoch_M1':updates[-1]['epoch'],
        'training_row_presentations':sum(r['rows'] for r in updates),'completed_independent_M0':0,
        'searches_started':0,'charged_attempts':0,'recovery_updates':0,'evaluation_rows':0,'timing_pairs':0,
        'Random_vs_Evo_delta_ce':None,'test_top1':None,'speedup':None,
        'peak_tree_rss_bytes':guard['peak_tree_rss_bytes'],
        'peak_total_gpu_used_bytes_includes_foreign':guard['peak_gpu_used_bytes'],
        'raw_report_status_is_stale':read('report.json')['status']=='started',
        'postrun_old_files_verified':1762,'postrun_package_files_verified':325,
        'original_inventory_files_verified':len(inventory),'live_owned_processes_after':0,
        'new_numerical_repeats':0,'gpu_exclusivity_verified':False,
        'primary_panel_complete':False,'main_result_available':False,
        'preparation_checks':{'windows_main_passed':395,'windows_skipped':20,'windows_facade_passed':53,
            'installed_linux_passed':468,'modules':18}}
    user_stop=json.loads((ROOT/'validation/user-requested-stop.json').read_text())
    assert user_stop['remaining_live_owned_processes']==[] and user_stop['new_numerical_launches']==0
    summary.update(cluster_stopped_by_user=True,cluster_stop_utc=user_stop['utc'])
    queries={}
    rows_by_id={'progress':progress_rows,'telemetry':telemetry_rows,'stages':phases,
        'epochs':[{'epoch':r['epoch'],'updates':r['updates'],'seconds':r['timestamp']-launch,'next_lr':r['next_lr']} for r in epochs],
        'summary':[summary]}
    paths={'progress':'workers/baseline_M1/ledger.jsonl','epochs':'workers/baseline_M1/ledger.jsonl',
        'telemetry':'control/series-1-supervision.jsonl','stages':'control/interruption-closure.json','summary':'control/manual-export-budget.json'}
    for name,rows in rows_by_id.items():
        relative=paths[name]
        source_files=[relative]
        if name=='summary':
            source_files += ['control/interruption-closure.json','control/source-manifest.json',
                'control/postrun-preservation.json','control/installed-checks-receipt.json',
                'control/installed-cpu-tests.log','control/series-1-budget-guard.json',
                'workers/baseline_M1/ledger.jsonl','control/series-1-supervision.jsonl']
        if name=='stages':source_files += ['protocol.json','workers/baseline_M1/ledger.jsonl']
        queries[name]={'rows':rows,'methods':[{'language':'Python','code':'Run replay.py; every assertion must pass. Progress uses the last confirmed update in each integer second; full raw records remain in ledger.jsonl. No GPU or model execution.'}],
            'source':{'label':f'PR #140 · {relative}','kind':'local-experiment-evidence','classification':'observed',
                'files':[f'results/independent-m0-a100-2026-10-30/raw/{path}' for path in source_files],
                'executedAt':budget['export_closed_utc'],
                'evidenceFlow':[{'title':'Источник '+str(i+1),'detail':path+'; SHA256 '+sha(RAW/path)} for i,path in enumerate(source_files)]+[
                    {'title':'Проверка','detail':'replay.py проверяет хеши, непрерывность SGD, непересечение ролей, отсутствие поиска и оценок; значения без моделей/CUDA.'}],
                'caveats':['Серия прервана; частичное обучение не измеряет итоговое качество.',
                    'NVML использует PID хоста; прежняя проверка ошибочно отбрасывала недоступные PID.',
                    'Нулевые счётчики означают отсутствие запуска, не нулевую точность или нулевой эффект.'],
                'metricDefinitions':[{'label':'Подтверждённые обновления SGD','definition':'Записи baseline_update с непрерывными номерами; только завершённые optimizer.step.',
                    'componentIds':['run-progress','stages','opening','methodology']},
                    {'label':'GPU-процессы','definition':'Число сырых строк NVML без фильтрации через пространство PID контейнера.',
                     'componentIds':['gpu-contexts','incident','opening']} ]}}
        if name=='summary':
            stop_ref='results/independent-m0-a100-2026-10-30/validation/user-requested-stop.json'
            queries[name]['source']['files'].append(stop_ref)
            queries[name]['source']['evidenceFlow'].append({'title':'Остановка владельцем',
                'detail':stop_ref+'; SHA256 '+sha(ROOT/'validation/user-requested-stop.json')})
    snapshot={'surface':'report','title':'Сравнение Random/Evo прервано до поиска: контроль GPU пропустил второй процесс',
        'generatedAt':user_stop['utc'],'status':'reviewed','filters':[],
        'report':{'asOf':'2026-10-30'},'queries':queries}
    dest=ROOT/'analysis';dest.mkdir(exist_ok=True)
    (dest/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (dest/'reviewed-report.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':audit()
