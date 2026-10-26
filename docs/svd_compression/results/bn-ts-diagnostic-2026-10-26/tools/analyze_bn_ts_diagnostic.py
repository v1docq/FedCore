"""Independent NumPy audit of one fixed BN/SGD and TS-loading diagnostic."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def aa_gate(ratios,orders):
    if len(ratios)!=20 or orders!=['AB','BA']*10 or any(not math.isfinite(v) or v<=0 for v in ratios):return {'accepted':False,'reason':'invalid_pairs'}
    median=statistics.median(ratios);inside=sum(.85<=v<=1.15 for v in ratios)
    order=statistics.median(ratios[::2])/statistics.median(ratios[1::2])
    return {'accepted':.95<=median<=1.05 and inside>=18 and .9<=order<=1.1,
            'median_ratio':median,'inside_15_percent':inside,'ab_ba_median_ratio':order}

parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
parser.add_argument('--completion-report',type=Path,required=True)
args=parser.parse_args();raw=args.raw;args.output.mkdir(parents=True,exist_ok=False)
read=lambda name:json.loads((raw/name).read_text(encoding='utf-8'))
r=read('report.json');protocol=read('diagnostic_protocol.json');source=read('source_manifest.json')['supplied']
inventory=read('diagnostic-file-inventory.json')['files'];packet=read('launch-packet.json');receipt=read('launch-receipt.json')
guard=read('diagnostic-1-budget-guard.json');inner=read('diagnostic-1-supervision.json');parent=read('completion_admission.json')
checks=[]
def check(name,condition,details=None):checks.append({'check':name,'passed':bool(condition),'details':details})
check('terminal_supervision',guard['status'] in ('finished','stopped') and inner['status'] in ('finished','stopped'))
check('authorized_window',packet['one_diagnostic_authorized'] and packet['automatic_retry'] is False and packet['repeat_slot_seconds']==1800)
check('external_budget',guard['elapsed_seconds']<=1800 and guard['prior_charged_seconds']==3491.18732554093 and math.isclose(guard['total_charged_seconds'],guard['elapsed_seconds']+3491.18732554093,abs_tol=1e-9) and guard['total_charged_seconds']<=14400)
check('fixed_diagnostic_protocol',digest(protocol)==r['diagnostic_sha256']==packet['diagnostic_protocol_sha256']==source['diagnostic_protocol_sha256']=='2b2996e814fe31e83c2d042b499cf4bcd250fef1199892ce9ae80d8659747166')
check('code_and_wheel',source['source_commit']==packet['source_commit']=='7640681e198e1257cf58e9f21259b1ba4adce46e' and source['wheel_sha256']==packet['wheel_sha256']=='0d49ffcb8ac517fb8f754bee113bb85444db51e104889f6049562c609a41084e')
check('packet_and_guard',sha(raw/'launch-packet.json')==receipt['packet_sha256'] and sha(raw/'diagnostic_budget_guard.py')==packet['budget_guard_sha256'])
check('manifest_bytes',sha(raw/'source-manifest.json')==packet['source_manifest_sha256'])
check('read_only_sources',r['baseline_training_performed'] is r['search_performed'] is r['official_test_access'] is False)
check('parent_identity',parent['report_sha256']==r['completion_report_sha256']=='2aa82ef99dd7a3137900c191c2fd9d1890f3cdeca6311fedbefee43156d22303' and parent['inventory_sha256']==r['completion_inventory_sha256']=='c83086ca9f152969aadac7cc41fb750db10aaf78628d430d277d5bd809738d1f')
check('parent_report_bytes',sha(args.completion_report)==r['completion_report_sha256'] and json.loads(args.completion_report.read_text())==parent['report'])
check('parent_inventory_bytes',sha(raw/'completion-file-inventory.json')==r['completion_inventory_sha256'])
normalize=lambda text:tuple(sorted(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#') and not re.match(r'^fedcore(?:==| @ )',line.strip(),re.IGNORECASE)))
check('scientific_dependencies',normalize((raw/'environment_freeze.txt').read_text())==normalize((args.completion_report.parent/'environment_freeze.txt').read_text()))
check('scientific_runtime',r['environment']==parent['report']['environment'])
for name,expected in inventory.items():
    path=raw/name
    check('export_presence:'+name,path.is_file() or (path.suffix=='.pt' and name.startswith('checkpoints/')))
    if path.is_file():check('published_file:'+name,path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'])
ledger=[json.loads(line) for line in (raw/'ledger.jsonl').read_text().splitlines()]
observations=[json.loads(line) for line in (raw/'diagnostic-1-supervision.jsonl').read_text().splitlines()]
check('no_foreign_GPU',inner['preflight_gpu_processes']==[] and max(row['gpu_process_count'] or 0 for row in observations)<=1)
roles=read('data_roles.json');ids=roles['search_validation']
check('same_roles',len(ids)==5000 and digest(roles)==r['data']['roles_sha256']==parent['report']['data']['roles_sha256'])
tape=read('recovery_tape.json');updates=tape['updates'];encoded=tuple((tuple(v['row_ids']),v['augmentation_seed']) for v in updates)
check('common_200_step_tape',len(updates)==200 and digest(encoded)==tape['sha256']==r['recovery_tape_sha256'] and all(len(v['row_ids'])==128 and set(v['row_ids'])<=set(roles['train']) for v in updates))
generator=np.random.default_rng(990032);order=generator.permutation(np.asarray(roles['train'],dtype=np.int64))
reference=tuple((tuple(map(int,order[start:start+128])),int(generator.integers(0,2**31))) for start in range(0,25600,128))
check('independent_tape_replay',encoded==reference)
step_events=[v for v in ledger if v.get('event')=='diagnostic_SGD_update']
trajectories={name:[v for v in step_events if v['trajectory']==name] for name in ('SGD_lr_0.01','SGD_lr_0.001')}
check('no_third_SGD_trajectory',set(v['trajectory'] for v in step_events)<=set(trajectories))
for name,lr in (('SGD_lr_0.01',.01),('SGD_lr_0.001',.001)):
    rows=trajectories[name]
    check('step_counter:'+name,[v['updates'] for v in rows]==list(range(1,len(rows)+1)) and len(rows)<=200)
    check('SGD_finite_loss:'+name,all(v['lr']==lr and math.isfinite(v['loss']) and v['loss']>=0 and v['tape_sha256']==tape['sha256'] for v in rows))
quality=[];arrays={};prediction_rows=0
for name,item in r['quality'].items():
    metrics=item['validation'];path=raw/metrics['prediction_file']
    check('prediction_sha:'+name,sha(path)==metrics['prediction_sha256'])
    with np.load(path,allow_pickle=False) as data:values={key:data[key] for key in data.files}
    arrays[name]=values;n=len(values['row_ids']);prediction_rows+=n
    correct=int((values['predictions']==values['labels']).sum());ce=float(values['cross_entropy'].sum()/n)
    check('validation_rows:'+name,np.array_equal(values['row_ids'],np.asarray(ids)) and all(v.shape==(5000,) for v in values.values()))
    check('validation_values:'+name,np.isfinite(values['cross_entropy']).all() and np.all((values['labels']>=0)&(values['labels']<10)) and np.all((values['predictions']>=0)&(values['predictions']<10)))
    check('validation_metrics:'+name,metrics['rows']==5000 and correct==metrics['correct'] and correct/5000==metrics['top1'] and math.isclose(ce,metrics['cross_entropy'],abs_tol=1e-12))
    stored=inventory[item['path']]
    check('checkpoint_cost:'+name,stored['sha256']==item['sha256'] and stored['bytes']==item['file_bytes'] and item['tensor_bytes']==stored['torch_zip_storage_bytes']==44734408 and item['gpu_reload_exact'])
    check('artifact_identity:'+name,item['artifact_id']==digest({'solution_id':item['solution_id'],'checkpoint_sha256':item['sha256'],'phase':item['phase']}))
    quality.append({'name':name,'top1':correct/5000,'cross_entropy':ce,'SGD_updates':item['SGD_updates'],
                    'lr':item['lr'],'additional_BN_calibration':item['additional_BN_calibration'],
                    'tensor_bytes':item['tensor_bytes'],'checkpoint_sha256':item['sha256']})
pairs=[]
for before,after in (('M0_untouched','M0_post_BN'),('SGD_lr_0.01_pre_BN','SGD_lr_0.01_post_BN'),('SGD_lr_0.001_pre_BN','SGD_lr_0.001_post_BN')):
    if before not in arrays or after not in arrays:continue
    first,second=arrays[before],arrays[after];check('same_pair_labels:'+before,np.array_equal(first['labels'],second['labels']))
    correct0=first['predictions']==first['labels'];correct1=second['predictions']==second['labels']
    pairs.append({'before':before,'after':after,'delta_pp':100*float(correct1.mean()-correct0.mean()),
                  'corrected':int((~correct0&correct1).sum()),'spoiled':int((correct0&~correct1).sum())})
    if before.startswith('SGD_'):check('shared_post_SGD_checkpoint:'+before,r['quality'][after]['shared_200_update_checkpoint_sha256']==r['quality'][before]['sha256'])
if 'M0_untouched' in arrays:
    original=parent['report']['baseline']['validation'];path=raw/original['prediction_file']
    check('original_predictions_SHA',sha(path)==original['prediction_sha256'])
    with np.load(path,allow_pickle=False) as data:check('exact_M0_replay',all(np.array_equal(data[key],arrays['M0_untouched'][key]) for key in data.files))
timing=r['timing'];aa=timing.get('AA',{});aa_rows=[];admitted={}
expected_aa=['M0_canonical_AA','M0_legacy_AA','hybrid_legacy_AA','hybrid_canonical_AA','R_canonical_AA','R_legacy_AA','E_legacy_AA','E_canonical_AA','M0_final_canonical_AA']
timing_order=[]
for event in ledger:
    if event.get('event')=='latency_pair' and event.get('pair')==0 and event.get('batch')==1:timing_order.append(event['label'])
check('fixed_AA_order',timing_order[:len(aa)]==expected_aa[:len(aa)] and len(aa)<=9)
for label,record in aa.items():
    individual=[]
    for batch,values in record['batches'].items():
        check('AA_pair_order:'+label+':'+batch,values['orders']==['AB','BA']*10 and [p['order'] for p in values['pairs']]==values['orders'])
        for clock in ('cuda_event','synchronized_wall'):
            ratios=[p['B'][clock+'_seconds']/p['A'][clock+'_seconds'] for p in values['pairs']]
            gate=aa_gate(ratios,values['orders']);individual.append(gate['accepted'])
            check('AA_independent_gate:'+label+':'+batch+':'+clock,gate==values[clock+'_admission'])
            aa_rows.append({'label':label,'role':record['role'],'condition':record['condition'],'batch':int(batch),'clock':clock,**gate})
    admitted[label]=all(individual);check('AA_full_profile:'+label,record['accepted']==admitted[label] and set(record['batches'])=={'1','128'})
    probe=record['probe'];check('same_TS_probe:'+label,probe['same_device_logits_accepted'] and probe['atol']==probe['rtol']==1e-5)
    check('fixed_timing_artifact:'+label,record['torchscript']==parent['timing_artifacts'][record['role']])
ab_rows=[]
for role,record in timing.get('AB',{}).items():
    allowed=admitted.get('M0_canonical_AA',False) and admitted.get('M0_final_canonical_AA',False) and admitted.get(role+'_canonical_AA',False)
    check('AB_frozen_gate:'+role,record['comparison_admitted']==allowed)
    if not allowed:
        check('no_unadmitted_AB:'+role,'batches' not in record);continue
    check('AB_pinned_inputs:'+role,record['first_torchscript']==parent['timing_artifacts']['M0'] and record['second_torchscript']==parent['timing_artifacts'][role])
    check('AB_pinned_loading:'+role,record['condition']=='canonical' and record['artifact_id']==parent['timing_artifacts'][role]['artifact_id'] and set(record['batches'])=={'1','128'})
    for side in ('first','second'):
        probe=record[side+'_probe']
        check('AB_reload_probe:'+role+':'+side,probe['same_device_logits_accepted'] and probe['same_probe'] and probe['rows']==2 and probe['atol']==probe['rtol']==1e-5 and math.isfinite(probe['max_abs_difference']))
    for batch,values in record['batches'].items():
        check('AB_pairs:'+role+':'+batch,values['orders']==['AB','BA']*10 and len(values['pairs'])==20)
        for clock in ('cuda_event','synchronized_wall'):
            first=[p['A'][clock+'_seconds'] for p in values['pairs']];second=[p['B'][clock+'_seconds'] for p in values['pairs']]
            ratios=[b/a for a,b in zip(first,second)];median=statistics.median(ratios)
            ab_rows.append({'role':role,'batch':int(batch),'clock':clock,'median_A_ms':1000*statistics.median(first),
                            'median_B_ms':1000*statistics.median(second),'median_B_over_A':median,'observed_A_over_B':1/median,
                            'min_ratio':min(ratios),'max_ratio':max(ratios)})
completed=r['status']=='completed'
if completed:
    check('successful_termination',guard['status']==inner['status']=='finished' and guard['returncode']==inner['returncode']==0)
    check('complete_quality_workload',len(quality)==6 and prediction_rows==30000 and len(step_events)==r['total_SGD_updates']==400)
    check('complete_timing_workload',len(aa)==9 and len(timing.get('AB',{}))==3)
    admitted_ab=['M0_vs_'+role+'_canonical_AB' for role in ('hybrid','R','E') if timing['AB'][role]['comparison_admitted']]
    check('exact_session_order',timing_order==expected_aa+admitted_ab)
    timing_pairs=[v for v in ledger if v.get('event')=='latency_pair']
    check('exact_pair_count',len(timing_pairs)==40*(9+len(admitted_ab)))
    for record in (*aa.values(),*(v for v in timing['AB'].values() if v['comparison_admitted'])):
        label=record['label']
        for batch,values in record['batches'].items():
            observed=[v for v in timing_pairs if v['label']==label and v['batch']==int(batch)]
            check('timing_ledger:'+label+':'+batch,len(observed)==20 and all(v['pair']==i and v['order']==values['pairs'][i]['order'] and v['A']==values['pairs'][i]['A'] and v['B']==values['pairs'][i]['B'] for i,v in enumerate(observed)))
    check('worker_deadline',r['ended_at']<=r['absolute_deadline'] and inner['worker_limit_seconds']==1790)
failed=[item for item in checks if not item['passed']]
summary={'schema':'fedcore-bn-ts-analysis-v1','status':r['status'],'verification_consistent':not failed,
    'checks':len(checks),'failed_checks':failed,'quality':quality,'extra_BN_pairs':pairs,'AA':aa_rows,'AB':ab_rows,
    'SGD_updates':len(step_events),'new_prediction_rows':prediction_rows,'AA_sessions':len(aa),
    'AB_sessions':sum(v.get('comparison_admitted',False) for v in timing.get('AB',{}).values()),
    'prior_seconds':guard['prior_charged_seconds'],'diagnostic_seconds':guard['elapsed_seconds'],
    'total_seconds':guard['total_charged_seconds'],'source_commit':source['source_commit'],
    'wheel_sha256':source['wheel_sha256'],'diagnostic_protocol_sha256':digest(protocol),
    'failure':r.get('failure'),'stop_reason':r.get('stop_reason')}
for name,data in (('analysis.json',summary),('checks.json',checks)):
    (args.output/name).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({key:summary[key] for key in ('status','verification_consistent','checks','failed_checks','SGD_updates','new_prediction_rows','AA_sessions','AB_sessions')}))
raise SystemExit(bool(failed))
