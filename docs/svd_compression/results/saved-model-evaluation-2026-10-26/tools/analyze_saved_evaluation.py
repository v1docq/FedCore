"""Independent NumPy/stdlib audit; no model execution or new test access."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def safe_ratio(pair,clock):
    a,b=pair['A'][clock+'_seconds'],pair['B'][clock+'_seconds']
    return b/a if all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (a,b)) else float('nan')
def aa_gate(ratios,orders):
    if len(ratios)!=20 or orders!=['AB','BA']*10 or any(not math.isfinite(v) or v<=0 for v in ratios):return {'accepted':False,'reason':'invalid_pairs'}
    median=statistics.median(ratios);inside=sum(.85<=v<=1.15 for v in ratios)
    order=statistics.median(ratios[::2])/statistics.median(ratios[1::2])
    return {'accepted':.95<=median<=1.05 and inside>=18 and .9<=order<=1.1,
            'median_ratio':median,'inside_15_percent':inside,'ab_ba_median_ratio':order}

p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path)
p.add_argument('--parent-report',type=Path,required=True);p.add_argument('--parent-freeze',type=Path,required=True)
p.add_argument('--historical-test-npz',type=Path,required=True)
args=p.parse_args();raw=args.raw;args.output.mkdir(parents=True,exist_ok=False)
read=lambda name:json.loads((raw/name).read_text(encoding='utf-8'))
mandatory=('evaluation_protocol.json','source_manifest.json','recovery_admission.json',
           'ledger.jsonl','evaluation-1-supervision.jsonl')
missing_scientific=[name for name in mandatory if not (raw/name).is_file()]
if missing_scientific:
    report=read('report.json');guard=read('evaluation-1-budget-guard.json')
    packet=read('launch-packet.json');source=read('source-manifest.json')
    checks=[{'check':'source_manifest_bytes','passed':sha(raw/'source-manifest.json')==packet['source_manifest_sha256']},
            {'check':'external_budget','passed':guard['prior_charged_seconds']==3827.6313528195024 and guard['elapsed_seconds']<=1800 and math.isclose(guard['total_charged_seconds'],guard['elapsed_seconds']+guard['prior_charged_seconds'],abs_tol=1e-9) and guard['total_charged_seconds']<=14400},
            {'check':'partial_status_disclosed','passed':report['status']!='completed'}]
    summary={'schema':'fedcore-saved-evaluation-analysis-v1','status':report['status'],
        'audit_scope':'control_only','missing_scientific_files':missing_scientific,
        'verification_consistent':all(v['passed'] for v in checks),'checks':len(checks),
        'failed_checks':[v for v in checks if not v['passed']],
        'source_commit':source['source_commit'],'evaluation_seconds':guard['elapsed_seconds'],
        'total_seconds':guard['total_charged_seconds'],'failure':report.get('failure'),
        'quality':[],'AA':[],'AB':[]}
    for name,value in (('analysis.json',summary),('checks.json',checks)):
        (args.output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary));raise SystemExit(not summary['verification_consistent'])
r=read('report.json');completed=r['status']=='completed';protocol=read('evaluation_protocol.json')
source=read('source_manifest.json')['supplied'];parent=read('recovery_admission.json')
inventory=read('evaluation-file-inventory.json')['files'];packet=read('launch-packet.json');receipt=read('launch-receipt.json')
guard=read('evaluation-1-budget-guard.json');inner=read('evaluation-1-supervision.json')
checks=[]
def check(name,value):checks.append({'check':name,'passed':bool(value)})
check('approved_window',packet['one_evaluation_authorized'] and packet['automatic_retry'] is False and packet['repeat_slot_seconds']==1800)
check('external_accounting',guard['prior_charged_seconds']==3827.6313528195024 and guard['elapsed_seconds']<=1800 and math.isclose(guard['total_charged_seconds'],guard['elapsed_seconds']+guard['prior_charged_seconds'],abs_tol=1e-9) and guard['total_charged_seconds']<=14400)
check('frozen_profile',digest(protocol)==r['evaluation_sha256']==source['evaluation_protocol_sha256']==packet['evaluation_protocol_sha256']=='2170fd6a57c24dd75b9a91aa07e9be769e925f052128972659c5200d215198bf')
check('frozen_code_wheel',source['source_commit']==packet['source_commit']=='0693a4bec2cf0cc26397d705dfc2810c7d95f6e7' and source['wheel_sha256']==packet['wheel_sha256']=='9a643d9cbd640a91c74276ccbd50bda16f3a13f466d5944a6bf7178e36f1335c')
check('source_manifest_bytes',sha(raw/'source-manifest.json')==packet['source_manifest_sha256'])
check('launch_guard_identity',sha(raw/'launch-packet.json')==receipt['packet_sha256'] and sha(raw/'evaluation_budget_guard.py')==packet['budget_guard_sha256'])
check('parent_identity',sha(args.parent_report)==parent['report_sha256']==protocol['parent_report_sha256']=='dda102a72595317c7f32345806f4c99b95d717a9361ea8592eb28d703590925c' and json.loads(args.parent_report.read_text())==parent['report'])
check('parent_inventory',sha(raw/'recovery-file-inventory.json')==protocol['parent_inventory_sha256']=='e8ec61fa2b8e59350f294c1934196c406668cc7c4b8b07e3bce1a01b88b44e6c')
check('no_training_search_BN',r['training_updates']==r['total_SGD_updates']==0 and r['baseline_training_performed'] is r['search_performed'] is r['additional_BN_calibration'] is False)
check('previous_test_use_disclosed',r['official_test_previous_use_disclosed'] is True and r['quality_scope']==protocol['quality_scope'])
for name,row in inventory.items():
    path=raw/name
    check('export_presence:'+name,path.is_file() or path.suffix=='.pt')
    if path.is_file():check('export_SHA:'+name,path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
normalize=lambda s:tuple(sorted(line.strip() for line in s.splitlines() if line.strip() and not line.lstrip().startswith('#') and not re.match(r'^fedcore(?:==| @ )',line.strip(),re.IGNORECASE)))
if (raw/'environment_freeze.txt').exists():check('scientific_dependencies',normalize((raw/'environment_freeze.txt').read_text())==normalize(args.parent_freeze.read_text()))
else:check('partial_before_dependencies',not completed)
if 'environment' in r:check('scientific_environment',r['environment']==parent['report']['environment'])
else:check('partial_before_environment',not completed)
roles=read('data_roles.json') if (raw/'data_roles.json').exists() else None
if roles:check('same_data_roles',digest(roles)==r['data']['roles_sha256']==parent['report']['data']['roles_sha256'])
else:check('partial_before_data_roles',not completed)
if 'data' in r:check('same_CIFAR_source_files',r['data']['source_files']==parent['report']['data']['source_files'])
else:check('partial_before_data_files',not completed)
check('historical_test_labels_source_SHA',sha(args.historical_test_npz)=='7462d0effaf1b27c0970733840e76afa48164f5cbdfc715ab72ed1b606bee3ea')
with np.load(args.historical_test_npz,allow_pickle=False) as data:historical_ids=data['row_ids'].copy();historical_labels=data['labels'].copy()
check('historical_test_population',historical_ids.dtype==historical_labels.dtype==np.dtype('int64') and np.array_equal(historical_ids,np.arange(10000)))
observations=[json.loads(line) for line in (raw/'evaluation-1-supervision.jsonl').read_text().splitlines()]
live=[v for v in observations if v['returncode'] is None];missing=sum(v['gpu_process_count'] is None for v in live)
check('live_telemetry_complete',missing==0)
check('no_foreign_GPU_observed',inner['preflight_gpu_processes']==[] and all(v['gpu_process_count']<=1 for v in live if v['gpu_process_count'] is not None))
ledger=[];ledger_failure=None
for number,line in enumerate((raw/'ledger.jsonl').read_text().splitlines(),1):
    try:
        event=json.loads(line)
        if type(event) is not dict:raise ValueError('Non-object event')
    except ValueError as error:ledger_failure=f'line {number}: {error}';break
    ledger.append(event)
check('ledger_parse',ledger_failure is None)
check('no_training_events',not any(v.get('event') in ('diagnostic_SGD_update','recovery_update','baseline_epoch') for v in ledger))
families=[v[0] for v in protocol['artifacts']];pins={v[0]:v[1:] for v in protocol['artifacts']}
limits={family+suffix:rows for family in families for suffix,rows in (('_eager_validation',5000),('_TS_validation',5000),('_TS_official_test',10000))}
batch_events=[v for v in ledger if v.get('event')=='saved_evaluation_batch']
last={};batch_valid=True;label_order=[]
for event in batch_events:
    label=event['label'];previous=last.get(label,{}).get('rows',0)
    if label not in last:label_order.append(label)
    valid=label in limits and type(event.get('rows')) is int and type(event.get('batch_rows')) is int and event['batch_rows']==min(128,limits.get(label,0)-previous) and event['rows']==previous+event['batch_rows'] and 0<event['rows']<=limits.get(label,0)
    batch_valid=bool(batch_valid and valid);last[label]=event
check('exact_batch_prefixes',batch_valid)
expected_evaluation_order=[family+suffix for family in families for suffix in ('_eager_validation','_TS_validation')]+[family+'_TS_official_test' for family in families]
check('fixed_evaluation_order',label_order==expected_evaluation_order[:len(label_order)])
arrays={};metrics={}
for label,event in last.items():
    path=raw/event['prediction_file'];check('last_prefix_SHA:'+label,sha(path)==event['prediction_sha256'])
    with np.load(path,allow_pickle=False) as data:values={key:data[key] for key in data.files}
    n=event['rows'];arrays[label]=values
    check('NPZ_schema:'+label,set(values)=={'row_ids','labels','predictions','cross_entropy'} and all(v.shape==(n,) and v.dtype==np.dtype('float64' if key=='cross_entropy' else 'int64') for key,v in values.items()))
    test=label.endswith('_TS_official_test');expected_ids=historical_ids[:n] if test else np.asarray(roles['search_validation'][:n],dtype=np.int64)
    check('row_order:'+label,np.array_equal(values['row_ids'],expected_ids))
    if test:check('official_labels:'+label,np.array_equal(values['labels'],historical_labels[:n]))
    check('finite_values:'+label,np.isfinite(values['cross_entropy']).all() and np.all(values['cross_entropy']>=0) and np.all((values['predictions']>=0)&(values['predictions']<10)))
    metrics[label]={'rows':n,'correct':int((values['predictions']==values['labels']).sum()),'top1':float((values['predictions']==values['labels']).mean()),'cross_entropy':float(values['cross_entropy'].sum()/n)}
for field,suffix in (('total_eager_validation_rows','_eager_validation'),('total_TS_validation_rows','_TS_validation'),('total_official_test_rows','_TS_official_test')):
    check('count:'+field,r[field]==sum(v['rows'] for key,v in last.items() if key.endswith(suffix)))
check('registered_workload_rows',r['observed_workload']['evaluation_rows']=={key:v['rows'] for key,v in last.items()} and r['total_evaluated_rows']==sum(v['rows'] for v in last.values())<=100000)
def same_metrics(name,record):
    value=metrics[name]
    check('metric:'+name,all(value[k]==record[k] for k in ('rows','correct','top1')) and math.isclose(value['cross_entropy'],record['cross_entropy'],abs_tol=1e-12))
if r.get('sources'):
    with np.load(raw/r['sources']['M0']['validation_reference']['path'],allow_pickle=False) as reference:master_validation_labels=reference['labels'].copy()
    for label,values in arrays.items():
        if not label.endswith('_TS_official_test'):check('validation_labels:'+label,np.array_equal(values['labels'],master_validation_labels[:len(values['labels'])]))
quality=[]
for family,item in r['quality'].items():
    identity,checkpoint_sha,cost=pins[family];source_item=item['source'];stored=inventory[source_item['path']]
    check('source_identity:'+family,source_item==r['sources'][family] and source_item['artifact_id']==identity and source_item['sha256']==checkpoint_sha==stored['sha256'] and source_item['tensor_bytes']==cost==stored['torch_zip_storage_bytes'])
    check('source_state:'+family,source_item['state']=='SGD_lr_0.001' and source_item['lr']==.001 and source_item['SGD_updates']==200 and source_item['additional_BN_calibration'] is False)
    eager_label=family+'_eager_validation';TS_label=family+'_TS_validation';test_label=family+'_TS_official_test'
    if 'eager_validation' in item:
        same_metrics(eager_label,item['eager_validation'])
        reference=source_item['validation_reference'];check('inherited_prediction_SHA:'+family,sha(raw/reference['path'])==reference['sha256']==parent['sources'][family]['validation']['prediction_sha256'])
        with np.load(raw/reference['path'],allow_pickle=False) as old:
            check('exact_eager_replay:'+family,all(old[key].dtype==arrays[eager_label][key].dtype and np.array_equal(old[key],arrays[eager_label][key]) for key in old.files))
    if 'TS_validation' in item:
        same_metrics(TS_label,item['TS_validation']);left,right=arrays[eager_label],arrays[TS_label]
        check('exact_TS_classes:'+family,all(np.array_equal(left[k],right[k]) for k in ('row_ids','labels','predictions')))
        check('TS_CE_tolerance:'+family,np.allclose(right['cross_entropy'],left['cross_entropy'],atol=1e-5,rtol=1e-5) and item['TS_equivalence']['CE_atol']==item['TS_equivalence']['CE_rtol']==1e-5)
    if family in r['official_test']:
        record=r['official_test'][family];same_metrics(test_label,record);ts=item['torchscript'];ts_stored=inventory[ts['path']]
        check('tested_TS_identity:'+family,ts['sha256']==record['torchscript_sha256']==ts_stored['sha256'] and ts['artifact_id']==identity and ts['source_checkpoint_sha256']==record['source_checkpoint_sha256']==checkpoint_sha)
        check('TS_physical_bytes:'+family,ts_stored['torch_zip_storage_bytes']==cost)
        description=r['descriptive_quality'][family];reference=r['official_test']['M0']
        check('descriptive_matched_control:'+family,math.isclose(description['top1_drop_from_matched_M0'],reference['top1']-record['top1'],abs_tol=1e-12) and math.isclose(description['CE_difference_from_matched_M0'],record['cross_entropy']-reference['cross_entropy'],abs_tol=1e-12) and description['selection_performed'] is False)
        quality.append({'family':family,'test_top1':metrics[test_label]['top1'],'test_CE':metrics[test_label]['cross_entropy'],'validation_top1':metrics[TS_label]['top1'],'tensor_bytes':cost,'fraction_M0':cost/44734408,'checkpoint_sha256':checkpoint_sha,'TS_sha256':ts['sha256'],'TS_file_bytes':ts_stored['bytes'],'TS_tensor_bytes':ts_stored['torch_zip_storage_bytes']})
if r.get('frozen_selection'):
    seal=r['frozen_selection'];check('seal_SHA',sha(raw/seal['path'])==seal['sha256']);sealed=read(seal['path'])
    check('sealed_scope',sealed['evaluation_sha256']==digest(protocol) and sealed['official_test_previous_use_disclosed'] and set(sealed['artifacts'])==set(families) and r['test_opened_after_frozen_selection'])
    for family in families:check('sealed_artifacts:'+family,sealed['artifacts'][family]['source']==r['sources'][family] and sealed['artifacts'][family]['torchscript']==r['quality'][family]['torchscript'])
timing=r['timing'];aa=timing.get('AA',{});aa_rows=[];admitted={}
timing_events=[v for v in ledger if v.get('event')=='latency_pair']
check('positive_finite_timing_values',all(type(v[side][clock+'_seconds']) in (int,float) and math.isfinite(v[side][clock+'_seconds']) and v[side][clock+'_seconds']>0 for v in timing_events for side in ('A','B') for clock in ('cuda_event','synchronized_wall')))
for label,record in aa.items():
    gates=[]
    check('AA_TS_identity:'+label,record['torchscript']==r['quality'][record['family']]['torchscript'] and record['load_policy']=='independent_direct_CUDA')
    for batch,values in record['batches'].items():
        orders=[v['order'] for v in values['pairs']]
        check('AA_order:'+label+':'+batch,orders==values['orders']==['AB','BA']*10)
        for clock in ('cuda_event','synchronized_wall'):
            ratios=[safe_ratio(v,clock) for v in values['pairs']]
            gate=aa_gate(ratios,orders);gates.append(gate['accepted'])
            check('AA_raw_gate:'+label+':'+batch+':'+clock,gate==values[clock+'_admission'])
            aa_rows.append({'label':label,'family':record['family'],'batch':int(batch),'clock':clock,**gate})
    admitted[label]=set(record['batches'])=={'1','128'} and all(gates)
    check('AA_full_gate:'+label,record['raw_AA_admitted']==admitted[label])
ab_rows=[]
for family,record in timing.get('AB',{}).items():
    pre=admitted.get('M0_initial_AA',False) and admitted.get('M0_pre_AB_AA',False) and admitted.get(family+'_AA',False)
    full=(set(record.get('batches',{}))=={'1','128'} and all(len(v.get('pairs',()))==20 and v.get('orders')==[p.get('order') for p in v.get('pairs',())]==['AB','BA']*10 and all(type(p.get(side,{}).get(clock+'_seconds')) in (int,float) and math.isfinite(p[side][clock+'_seconds']) and p[side][clock+'_seconds']>0 for p in v.get('pairs',()) for side in ('A','B') for clock in ('cuda_event','synchronized_wall')) for v in record.get('batches',{}).values()))
    final=bool(pre and admitted.get('M0_post_AB_AA',False) and completed and full)
    check('AB_pre_gate:'+family,record['admitted_before_AB']==pre)
    check('AB_final_gate:'+family,record['comparison_admitted']==final)
    if 'batches' not in record:continue
    check('AB_TS_identity:'+family,record['first_torchscript']==r['quality']['M0']['torchscript'] and record['second_torchscript']==r['quality'][family]['torchscript'] and record['load_policy']=='independent_direct_CUDA')
    for batch,values in record['batches'].items():
        check('AB_pairs:'+family+':'+batch,len(values['pairs'])==20 and values['orders']==[v['order'] for v in values['pairs']]==['AB','BA']*10)
        for clock in ('cuda_event','synchronized_wall'):
            first=[v['A'][clock+'_seconds'] for v in values['pairs']];second=[v['B'][clock+'_seconds'] for v in values['pairs']];ratios=[safe_ratio(v,clock) for v in values['pairs']]
            if any(not math.isfinite(v) for v in ratios):continue
            median=statistics.median(ratios)
            ab_rows.append({'family':family,'batch':int(batch),'clock':clock,'comparison_admitted':final,'median_M0_ms':1000*statistics.median(first),'median_model_ms':1000*statistics.median(second),'median_B_over_A':median,'observed_A_over_B':1/median,'min_ratio':min(ratios),'max_ratio':max(ratios)})
pair_counts=Counter(f"{v['label']}/batch_{v['batch']}" for v in timing_events)
check('timing_counters',dict(pair_counts)==r['observed_workload']['timing_pairs'] and len(timing_events)==r['total_latency_pairs']<=440)
expected_pre=protocol['AA_before_AB'];attempted_AB=[f'M0_vs_{family}_AB' for family in protocol['AB_order'] if timing.get('AB',{}).get(family,{}).get('admitted_before_AB')]
actual_order=[v['label'] for v in timing_events if v['batch']==1 and v['pair']==0]
expected_order=expected_pre+attempted_AB+['M0_post_AB_AA']
check('timing_session_order',actual_order==expected_order[:len(actual_order)])
for record in (*aa.values(),*(v for v in timing.get('AB',{}).values() if 'batches' in v)):
    for batch,values in record['batches'].items():
        events=[v for v in timing_events if v['label']==record['label'] and v['batch']==int(batch)]
        check('timing_ledger:'+record['label']+':'+batch,len(events)==len(values['pairs']) and all(v['pair']==i and v['order']==values['pairs'][i]['order'] and v['A']==values['pairs'][i]['A'] and v['B']==values['pairs'][i]['B'] for i,v in enumerate(events)))
if completed:
    check('successful_completion',guard['status']==inner['status']=='finished' and guard['returncode']==inner['returncode']==0 and r['within_slot_deadline'] and r['parents_read_only_verified'])
    check('full_fixed_workload',set(r['quality'])==set(r['official_test'])==set(families) and r['total_validation_rows']==50000 and r['total_official_test_rows']==50000 and len(aa)==7 and len(timing['AB'])==4 and label_order==expected_evaluation_order)
    check('full_timing_budget',len(timing_events)==280+40*len(attempted_AB) and actual_order==expected_order)
failed=[v for v in checks if not v['passed']]
summary={'schema':'fedcore-saved-evaluation-analysis-v1','status':r['status'],'verification_consistent':not failed,'checks':len(checks),'failed_checks':failed,'quality':quality,'AA':aa_rows,'AB':ab_rows,
    'eager_validation_rows':r['total_eager_validation_rows'],'TS_validation_rows':r['total_TS_validation_rows'],'official_test_rows':r['total_official_test_rows'],'latency_pairs':len(timing_events),'training_updates':0,
    'AA_sessions':len(aa),'AB_sessions_executed':sum('batches' in v for v in timing.get('AB',{}).values()),'AB_comparisons_admitted':sum(v['comparison_admitted'] for v in timing.get('AB',{}).values()),
    'prior_seconds':guard['prior_charged_seconds'],'evaluation_seconds':guard['elapsed_seconds'],'total_seconds':guard['total_charged_seconds'],
    'source_commit':source['source_commit'],'wheel_sha256':source['wheel_sha256'],'evaluation_protocol_sha256':digest(protocol),'ledger_failure':ledger_failure,'failure':r.get('failure'),'stop_reason':r.get('stop_reason'),
    'missing_live_telemetry_samples':missing,'peak_allocated_bytes':r.get('peak_allocated_bytes'),'peak_reserved_bytes':r.get('peak_reserved_bytes'),'peak_worker_rss_bytes':inner.get('peak_rss_bytes')}
for name,value in (('analysis.json',summary),('checks.json',checks)):(args.output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({key:summary[key] for key in ('status','verification_consistent','checks','failed_checks','official_test_rows','latency_pairs','AB_comparisons_admitted')}))
raise SystemExit(bool(failed))
