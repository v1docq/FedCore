"""Independent NumPy audit of the fixed five-family recovery/extra-BN matrix."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path)
p.add_argument('--completion-report',type=Path,required=True)
p.add_argument('--diagnostic-root',type=Path,required=True)
args=p.parse_args();raw=args.raw;args.output.mkdir(exist_ok=False,parents=True)
read=lambda name:json.loads((raw/name).read_text(encoding='utf-8'))
r=read('report.json');protocol=read('recovery_protocol.json');source=read('source_manifest.json')['supplied']
completed=r['status']=='completed'
inventory=read('recovery-file-inventory.json')['files'];packet=read('launch-packet.json');receipt=read('launch-receipt.json')
guard=read('recovery-1-budget-guard.json');inner=read('recovery-1-supervision.json')
completion=read('completion_admission.json');diagnostic=read('diagnostic_admission.json')
checks=[]
def check(name,condition):checks.append({'check':name,'passed':bool(condition)})
check('authorized_window',packet['one_recovery_authorized'] and packet['automatic_retry'] is False and packet['repeat_slot_seconds']==1800)
check('prior_and_total_budget',guard['prior_charged_seconds']==3648.862081296742 and guard['elapsed_seconds']<=1800 and math.isclose(guard['total_charged_seconds'],guard['prior_charged_seconds']+guard['elapsed_seconds'],abs_tol=1e-9) and guard['total_charged_seconds']<=14400)
check('fixed_protocol',digest(protocol)==source['recovery_protocol_sha256']==packet['recovery_protocol_sha256']==r['recovery_sha256'])
check('source_commit',source['source_commit']==packet['source_commit'] and len(source['source_commit'])==40)
check('wheel',source['wheel_sha256']==packet['wheel_sha256'])
check('frozen_release',source['source_commit']=='981a048d7d1d6fa57bdf2a725f0f446af03596a3' and source['wheel_sha256']=='a903eab2d1f8d1c64dc7659c0bccfc4ad7315b89ede521df16ec4849def15fa5' and digest(protocol)=='c42d1f75aec83cbe837c9c19c01745c4ba63926e29bf52853a5e936f20b3af97')
check('manifest_bytes',sha(raw/'source-manifest.json')==packet['source_manifest_sha256'])
check('launch_receipt',sha(raw/'launch-packet.json')==receipt['packet_sha256'] and sha(raw/'recovery_budget_guard.py')==packet['budget_guard_sha256'])
check('completion_report',sha(args.completion_report)==completion['report_sha256']==protocol['completion_report_sha256']=='2aa82ef99dd7a3137900c191c2fd9d1890f3cdeca6311fedbefee43156d22303' and json.loads(args.completion_report.read_text())==completion['report'])
check('diagnostic_report',sha(args.diagnostic_root/'report.json')==diagnostic['report_sha256']==protocol['diagnostic_report_sha256']=='a73489d743b541d58e267e9df75f01557499c92f552cd44c77909466a27848a9' and json.loads((args.diagnostic_root/'report.json').read_text())==diagnostic['report'])
check('parent_inventories',sha(raw/'completion-file-inventory.json')==protocol['completion_inventory_sha256'] and sha(raw/'diagnostic-file-inventory.json')==protocol['diagnostic_inventory_sha256'])
check('no_test_search_timing',r['official_test_access'] is r['search_performed'] is r['baseline_training_performed'] is False and r.get('latency_sessions',0)==0)
if 'environment' in r:
    check('scientific_environment',r['environment']==completion['report']['environment']==diagnostic['report']['environment'])
else:check('partial_before_environment',not completed and not r['quality'])
normalize=lambda s:tuple(sorted(line.strip() for line in s.splitlines() if line.strip() and not line.lstrip().startswith('#') and not re.match(r'^fedcore(?:==| @ )',line.strip(),re.IGNORECASE)))
if (raw/'environment_freeze.txt').exists():
    check('scientific_dependencies',normalize((raw/'environment_freeze.txt').read_text())==normalize((args.diagnostic_root/'environment_freeze.txt').read_text()))
else:check('partial_before_dependency_snapshot',not completed and not r['quality'])
for name,row in inventory.items():
    path=raw/name
    check('export_presence:'+name,path.is_file() or path.suffix=='.pt')
    if path.is_file():check('export_hash:'+name,path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
roles=read('data_roles.json') if (raw/'data_roles.json').exists() else None
ids=np.asarray(roles['search_validation'],dtype=np.int64) if roles else np.asarray([],dtype=np.int64)
if roles:
    check('same_data_roles',ids.shape==(5000,) and digest(roles)==r['data']['roles_sha256']==completion['report']['data']['roles_sha256']==diagnostic['report']['data']['roles_sha256'])
else:check('partial_before_data_snapshot',not completed and not r['quality'])
tape=read('recovery_tape.json') if (raw/'recovery_tape.json').exists() else None
if tape:
    encoded=tuple((tuple(v['row_ids']),v['augmentation_seed']) for v in tape['updates'])
    check('original_tape',digest(encoded)==r['recovery_tape_sha256']==tape['sha256']==diagnostic['report']['recovery_tape_sha256'] and len(encoded)==200)
    gen=np.random.default_rng(990032);order=gen.permutation(np.asarray(roles['train'],dtype=np.int64))
    reference=tuple((tuple(map(int,order[start:start+128])),int(gen.integers(0,2**31))) for start in range(0,25600,128))
    check('independent_tape_replay',encoded==reference)
else:check('partial_before_tape_snapshot',not completed and not r['quality'])
ledger=[];ledger_failure=None
for number,line in enumerate((raw/'ledger.jsonl').read_text().splitlines(),1):
    try:
        event=json.loads(line)
        if type(event) is not dict:raise ValueError('Non-object event')
    except ValueError as error:
        ledger_failure=f'line {number}: {error}';break
    ledger.append(event)
check('ledger_parse',ledger_failure is None)
observations=[json.loads(line) for line in (raw/'recovery-1-supervision.jsonl').read_text().splitlines()]
live=[v for v in observations if v['returncode'] is None]
missing_live_telemetry=sum(v['gpu_process_count'] is None for v in live)
check('live_GPU_telemetry_complete',missing_live_telemetry==0)
check('no_foreign_GPU',inner['preflight_gpu_processes']==[] and all(v['gpu_process_count']<=1 for v in live if v['gpu_process_count'] is not None))
steps=[v for v in ledger if v.get('event')=='diagnostic_SGD_update']
expected_roles=[family+'_SGD_lr_'+str(lr) for family,_,_ in protocol['families'] for lr in (.01,.001)]
check('registered_step_total',len(steps)==r['total_SGD_updates'] and len(steps)<=2000 and set(v['trajectory'] for v in steps)<=set(expected_roles))
observed={role:sum(v['trajectory']==role for v in steps) for role in expected_roles if any(v['trajectory']==role for v in steps)}
check('observed_step_counters',observed==r['observed_trajectory_updates'] and r['completed_trajectory_count']==len(r['trajectories']) and r['completed_trajectory_updates']==sum(v['updates'] for v in r['trajectories'].values()))
for role in expected_roles:
    rows=[v for v in steps if v['trajectory']==role];lr=float(role.split('_lr_')[1])
    check('step_sequence:'+role,[v['updates'] for v in rows]==list(range(1,len(rows)+1)) and len(rows)<=200)
    check('step_values:'+role,all(tape is not None and v['lr']==lr and math.isfinite(v['loss']) and v['loss']>=0 and v['tape_sha256']==tape['sha256'] for v in rows))
check('no_timing_events',not any(v.get('event')=='latency_pair' for v in ledger))
families={name:(identity,cost) for name,identity,cost in protocol['families']}
states=('no_SGD','SGD_lr_0.01','SGD_lr_0.001')
expected_labels=[family+'_'+state+'_'+suffix for family in families for state in states for suffix in ('pre_BN','post_BN')]
expected_cells={family+'_'+state+'_'+suffix:(family,state,None if state=='no_SGD' else float(state.split('_lr_')[1]),0 if state=='no_SGD' else 200,suffix=='post_BN') for family in families for state in states for suffix in ('pre_BN','post_BN')}
master=diagnostic['report']['quality']['M0_untouched']['validation']
check('master_labels_SHA',sha(args.diagnostic_root/master['prediction_file'])==master['prediction_sha256'])
with np.load(args.diagnostic_root/master['prediction_file'],allow_pickle=False) as master_npz:master_labels=master_npz['labels'].copy()
quality=[];arrays={};prediction_rows=0
for label,item in r['quality'].items():
    family,state=item['family'],item['state'];identity,cost=families[family]
    check('fixed_cell_identity:'+label,label in expected_cells and (family,state,item['lr'],item['SGD_updates'],item['additional_BN_calibration'])==expected_cells[label])
    metrics=item['validation'];path=raw/metrics['prediction_file']
    check('prediction_SHA:'+label,sha(path)==metrics['prediction_sha256'])
    with np.load(path,allow_pickle=False) as data:values={key:data[key] for key in data.files}
    arrays[label]=values;prediction_rows+=len(values['row_ids'])
    check('prediction_schema:'+label,set(values)=={'row_ids','labels','predictions','cross_entropy'} and all(v.shape==(5000,) for v in values.values()) and np.array_equal(values['row_ids'],ids))
    check('prediction_dtypes_labels:'+label,all(value.dtype==np.dtype('float64' if key=='cross_entropy' else 'int64') for key,value in values.items()) and np.array_equal(values['labels'],master_labels))
    check('prediction_values:'+label,np.isfinite(values['cross_entropy']).all() and np.all(values['cross_entropy']>=0) and np.all((values['labels']>=0)&(values['labels']<10)) and np.all((values['predictions']>=0)&(values['predictions']<10)))
    correct=int((values['predictions']==values['labels']).sum());ce=float(values['cross_entropy'].sum()/5000)
    check('quality_metrics:'+label,metrics['rows']==5000 and metrics['correct']==correct and metrics['top1']==correct/5000 and math.isclose(metrics['cross_entropy'],ce,abs_tol=1e-12))
    stored=inventory[item['path']]
    check('model_identity_cost:'+label,item['solution_id']==identity and item['tensor_bytes']==stored['torch_zip_storage_bytes']==cost and item['sha256']==stored['sha256'] and item['file_bytes']==stored['bytes'] and item['gpu_reload_exact'])
    check('model_artifact_ID:'+label,item['artifact_id']==digest({'solution_id':identity,'checkpoint_sha256':item['sha256'],'phase':item['phase']}))
    check('same_start:'+label,item['initial_checkpoint_sha256']==r['starts'][family]['sha256'] and item['source_M0_sha256']==protocol.get('baseline_sha256','531d02916a6a4ad7341f95a1f7e075c91c7b5350bc7009a09d5b93077c4d5510'))
    if state!='no_SGD':check('quality_training_lineage:'+label,item['training']==r['trajectories'][family+'_'+state])
    if family=='M0':
        old_label=('M0_untouched' if not item['additional_BN_calibration'] else 'M0_post_BN') if state=='no_SGD' else state+('_post_BN' if item['additional_BN_calibration'] else '_pre_BN')
        old=diagnostic['report']['quality'][old_label]['validation']
        old_path=args.diagnostic_root/old['prediction_file']
        check('dense_prediction_parent_SHA:'+label,sha(old_path)==old['prediction_sha256'])
        with np.load(old_path,allow_pickle=False) as old_values:
            check('dense_exact_replay:'+label,all(values[key].dtype==old_values[key].dtype and np.array_equal(values[key],old_values[key]) for key in values))
    quality.append({'label':label,'family':family,'state':state,'extra_BN':item['additional_BN_calibration'],'top1':correct/5000,'cross_entropy':ce,'tensor_bytes':cost,'fraction_M0':cost/44734408,'SGD_updates':item['SGD_updates'],'checkpoint_sha256':item['sha256']})
bn_pairs=[]
for family in families:
    for state in states:
        prefix=family+'_'+state;before=prefix+'_pre_BN';after=prefix+'_post_BN'
        if before not in arrays or after not in arrays:continue
        first,second=arrays[before],arrays[after]
        check('pair_labels:'+prefix,np.array_equal(first['row_ids'],second['row_ids']) and np.array_equal(first['labels'],second['labels']))
        correct0=first['predictions']==first['labels'];correct1=second['predictions']==second['labels']
        if prefix not in r['BN_pairs']:
            check('missing_BN_pair_proof:'+prefix,False)
            continue
        proof=r['BN_pairs'][prefix];pre=r['quality'][before];post=r['quality'][after]
        check('BN_pair_parameters:'+prefix,proof['parameters_exactly_unchanged'] and proof['independent_reloads'] and pre['parameter_fingerprint']==post['parameter_fingerprint']==digest(proof['parameter_evidence']))
        check('BN_pair_buffers:'+prefix,pre['buffer_fingerprint']==digest(proof['buffers_before']) and post['buffer_fingerprint']==digest(proof['buffers_after']) and set(proof['buffers_before'])==set(proof['buffers_after']))
        changed={key:proof['buffers_before'][key]!=proof['buffers_after'][key] for key in proof['buffers_before']}
        check('BN_pair_allowed_changes:'+prefix,changed==proof['buffers_changed'] and not any(value for key,value in changed.items() if not key.endswith(('running_mean','running_var','num_batches_tracked'))))
        check('BN_pair_same_snapshot:'+prefix,pre['shared_pre_BN_checkpoint_sha256']==post['shared_pre_BN_checkpoint_sha256']==proof['shared_pre_BN_checkpoint_sha256'] and proof['calibration_rows']==1000 and proof['reset_running_stats'] and proof['momentum_during_calibration'] is None)
        bn_pairs.append({'family':family,'state':state,'delta_pp':100*float(correct1.mean()-correct0.mean()),'CE_delta':float(second['cross_entropy'].mean()-first['cross_entropy'].mean()),'corrected':int((~correct0&correct1).sum()),'spoiled':int((correct0&~correct1).sum())})
for role,item in r['trajectories'].items():
    family=role.split('_SGD_')[0]
    check('trajectory_start:'+role,item['updates']==200 and item['initial_checkpoint_sha256']==r['starts'][family]['sha256'] and item['initial_parameter_fingerprint']==digest(r['starts'][family]['initial_state']['parameters']))
    check('trajectory_optimizer:'+role,item['lr']==float(role.split('_lr_')[1]) and item['optimizer']=='fresh_SGD_momentum_0.9_nesterov_weight_decay_0.0005' and item['training_BN_updates'] and not item['additional_BN_calibration'] and item['tape_sha256']==tape['sha256'] and item['all_parameters_received_finite_gradients'])
check('reported_validation_rows',prediction_rows==r['total_validation_rows'] and prediction_rows<=150000)
if completed:
    check('successful_termination',guard['status']==inner['status']=='finished' and guard['returncode']==inner['returncode']==0)
    check('complete_workload',set(r['quality'])==set(expected_labels) and len(bn_pairs)==15 and len(steps)==2000 and prediction_rows==150000 and set(r['trajectories'])==set(expected_roles))
    check('no_partial_trajectory',all(sum(v['trajectory']==role for v in steps)==200 for role in expected_roles))
    check('immutable_teacher_and_parents',r['teacher_fingerprint_before']==r['teacher_fingerprint_after'] and r['parents_read_only_verified'])
    check('worker_deadline',r['within_slot_deadline'] and r['ended_at']<=r['absolute_deadline'] and inner['worker_limit_seconds']==1790)
failed=[v for v in checks if not v['passed']]
summary={'schema':'fedcore-recovery-bn-analysis-v1','status':r['status'],'verification_consistent':not failed,'checks':len(checks),'failed_checks':failed,
    'quality':quality,'BN_pairs':bn_pairs,'SGD_updates':len(steps),'new_prediction_rows':prediction_rows,
    'complete_trajectories':len(r['trajectories']),'families':list(families),'states':list(states),
    'prior_seconds':guard['prior_charged_seconds'],'recovery_seconds':guard['elapsed_seconds'],'total_seconds':guard['total_charged_seconds'],
    'source_commit':source['source_commit'],'wheel_sha256':source['wheel_sha256'],'recovery_protocol_sha256':digest(protocol),
    'failure':r.get('failure'),'stop_reason':r.get('stop_reason'),'ledger_failure':ledger_failure,
    'manifest_evidence_available':roles is not None and tape is not None,'missing_live_telemetry_samples':missing_live_telemetry,'peak_allocated_bytes':r.get('peak_allocated_bytes'),
    'peak_reserved_bytes':r.get('peak_reserved_bytes'),'peak_worker_rss_bytes':inner.get('peak_rss_bytes')}
for name,value in (('analysis.json',summary),('checks.json',checks)):
    (args.output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({key:summary[key] for key in ('status','verification_consistent','checks','failed_checks','SGD_updates','new_prediction_rows')}))
raise SystemExit(bool(failed))
