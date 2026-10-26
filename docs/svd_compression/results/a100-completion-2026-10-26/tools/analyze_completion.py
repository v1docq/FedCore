"""Independent completion audit: NumPy/stdlib only, no model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np

def checksum(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def physical_bytes(genotype):
    choices={g['aliases'][0]:g['choice'] for g in genotype['genes']}
    total=4*(64*3*9)+16*64+8+4*(512*10+10)
    for path,outer,incoming in zip((f'layer{s}.{b}' for s in range(1,5) for b in range(2)),
                                  (64,64,128,128,256,256,512,512),(64,64,64,128,128,256,256,512)):
        choice=choices[path];width=choice.get('keep_channels',outer);rank=choice.get('rank')
        total+=4*width*incoming*9+16*width+8
        total+=4*(outer*width*9 if rank is None else rank*width*9+outer*rank)+16*outer+8
        if incoming!=outer:total+=4*outer*incoming+16*outer+8
    return total

def aa_admission(ratios, orders):
    if len(ratios)!=20 or orders!=['AB','BA']*10 or any(not math.isfinite(x) or x<=0 for x in ratios):
        return {'accepted':False,'reason':'invalid_pairs'}
    median=statistics.median(ratios);inside=sum(.85<=x<=1.15 for x in ratios)
    order_ratio=statistics.median(ratios[::2])/statistics.median(ratios[1::2])
    return {'accepted':.95<=median<=1.05 and inside>=18 and .9<=order_ratio<=1.1,
            'median_ratio':median,'inside_15_percent':inside,'ab_ba_median_ratio':order_ratio}

parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
parser.add_argument('--parent-report',type=Path,required=True)
args=parser.parse_args();raw=args.raw;args.output.mkdir(parents=True,exist_ok=False)
read=lambda name:json.loads((raw/name).read_text(encoding='utf-8'))
checks=[]
def check(name,value,details=None): checks.append({'check':name,'passed':bool(value),'details':details})
r=read('report.json');inventory=read('completion-file-inventory.json')['files']
guard=read('completion-1-budget-guard.json');inner=read('completion-1-supervision.json')
packet=read('launch-packet.json');receipt=read('launch-receipt.json');protocol=read('protocol.json')
source=read('source_manifest.json')['supplied'];parent=read('parent_admission.json');old=parent['report']
check('completed_terminal',r['status']=='completed' and guard['status']==inner['status']=='finished' and guard['returncode']==inner['returncode']==0)
check('authorized_single_completion',packet['automatic_retry'] is False and packet['one_completion_authorized'] is True and packet['repeat_slot_seconds']==1800)
check('strict_own_slot',guard['elapsed_seconds']<=1800 and r['ended_at']<=r['absolute_deadline'] and inner['worker_limit_seconds']==1790)
check('cumulative_budget',guard['prior_charged_seconds']==3339.3260714150965 and math.isclose(guard['total_charged_seconds'],guard['prior_charged_seconds']+guard['elapsed_seconds'],abs_tol=1e-9) and guard['total_charged_seconds']<=14400)
check('unchanged_scientific_protocol',digest(protocol)==r['protocol_sha256']==source['protocol_sha256']==packet['protocol_sha256']=='81e30bb6853080e2b6fedf966fc19f44cf717a8192988df5c3db552a3dd12010')
check('new_source_wheel',source['source_commit']==packet['source_commit']=='db4c0669c71a647b1e7df622cdc2885f17bd576a' and source['wheel_sha256']==packet['wheel_sha256']=='c26d3a0683d31b4569e0906b7c3825df3422781b6b2dbcd9f5ef345ed107e5d9')
check('guard_and_packet_hashes',checksum(raw/'launch-packet.json')==receipt['packet_sha256'] and checksum(raw/'completion_budget_guard.py')==packet['budget_guard_sha256'])
check('source_manifest_hash',checksum(raw/'source-manifest.json')==packet['source_manifest_sha256'])
check('parent_pinned',parent['report_sha256']==r['parent_report_sha256']=='10ea02afb504252e925bedb9556712ba3b2d9964659e39c93691c3fc5a4af05d' and checksum(raw/'remote-file-inventory.json')==r['parent_inventory_sha256']=='5ec5d0dd4fe1d403ffa929e1b0904fde881562368040b2ec876a5e396d408a15')
check('parent_report_content',checksum(args.parent_report)==r['parent_report_sha256'] and old==json.loads(args.parent_report.read_text(encoding='utf-8')))
normalize_freeze=lambda text:tuple(sorted(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#') and not re.match(r'^fedcore(?:==| @ )',line.strip(),re.IGNORECASE)))
check('independent_dependency_versions',normalize_freeze((raw/'environment_freeze.txt').read_text())==normalize_freeze((args.parent_report.parent/'environment_freeze.txt').read_text()))
check('parent_snapshot_complete',parent['verified_files']==read('remote-file-inventory.json')['files'] and read('parent-snapshot.json')['full_parent_verified'] and read('parent-snapshot.json')['original_files_modified'] is False)
check('no_training_or_new_search',r['baseline_training_performed'] is r['search_performed'] is False and r['inherited_short_attempts']==old['all_short_attempts'])
check('free_gpu_admission',inner['preflight_gpu_processes']==[] and max(row['gpu_process_count'] or 0 for row in (json.loads(line) for line in (raw/'completion-1-supervision.jsonl').read_text().splitlines()))<=1)
for name,item in inventory.items():
    path=raw/name
    if path.is_file():check('copied_file:'+name,path.stat().st_size==item['bytes'] and checksum(path)==item['sha256'])
check('runtime_environment',r['environment']==old['environment'] and read('parent-snapshot.json')['dependencies_match'])
check('installed_package_manifest',read('source_manifest.json')['verified_package_files_sha256']==source['package_files_sha256'])
roles=read('data_roles.json');check('data_roles_hash',digest(roles)==r['data']['roles_sha256']==old['data']['roles_sha256'])
check('role_sizes',tuple(len(roles[name]) for name in ('train','calibration','search_validation'))==(44000,1000,5000))
parts=[set(roles[name]) for name in ('train','calibration','search_validation')]
check('disjoint_training_roles',not(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2]) and set.union(*parts)==set(range(50000)))

prediction_rows=0;prediction_files=0
def recount(metrics,expected_ids,label):
    global prediction_rows,prediction_files
    path=raw/metrics['prediction_file']
    check('prediction_sha:'+label,checksum(path)==metrics['prediction_sha256'])
    with np.load(path,allow_pickle=False) as data:
        arrays={key:data[key] for key in data.files}
    ids=arrays['row_ids'];targets=arrays['labels'];predictions=arrays['predictions'];losses=arrays['cross_entropy']
    check('prediction_ids:'+label,np.array_equal(ids,np.asarray(expected_ids)) and len(ids)==len(set(map(int,ids))))
    check('prediction_shapes:'+label,all(a.shape==(len(ids),) for a in (targets,predictions,losses)) and np.isfinite(losses).all() and np.all((0<=targets)&(targets<10)) and np.all((0<=predictions)&(predictions<10)))
    correct=int((predictions==targets).sum());ce=float(losses.sum()/len(ids));top1=correct/len(ids)
    check('prediction_metrics:'+label,metrics['rows']==len(ids) and metrics['correct']==correct and metrics['top1']==top1 and math.isclose(metrics['cross_entropy'],ce,abs_tol=1e-12))
    prediction_rows+=len(ids);prediction_files+=1
    return {'rows':len(ids),'correct':correct,'top1':top1,'cross_entropy':ce},arrays

_,baseline_predictions=recount(r['baseline']['validation'],roles['search_validation'],'original_M0_validation')
_,current_predictions=recount(r['baseline_recheck'],roles['search_validation'],'M0_recheck')
check('exact_M0_recheck',all(np.array_equal(baseline_predictions[k],current_predictions[k]) for k in baseline_predictions))
baseline=r['baseline'];recovered=r['recovered'];artifacts={baseline['artifact_id']:baseline,**{a['artifact_id']:a for a in recovered.values()}}
check('six_distinct_artifacts',len(artifacts)==6 and len(recovered)==5)
check('same_original_M0',baseline['sha256']=='531d02916a6a4ad7341f95a1f7e075c91c7b5350bc7009a09d5b93077c4d5510' and baseline['tensor_bytes']==44734408)
check('original_training_admission',baseline['training']['completed_epochs']==80 and baseline['training']['all_parameter_tensors_changed'] and baseline['training']['all_parameters_received_finite_gradients'] and baseline['validation']['top1']==.9506)
genotypes={row['solution_id']:row['genotype'] for records in old['all_short_attempts'].values() for row in records}
selected=read('recovery_selection.json')
winners={}
for name in ('random','evolution'):
    rows=old['all_short_attempts'][name]
    accepted=[row for row in rows if row['status']=='succeeded' and row['quality_accepted'] is True and row['completed_within_deadline'] is True and type(row['tensor_bytes']) is int and 0<row['tensor_bytes']<=35787526 and math.isfinite(row['cross_entropy'])]
    winners[name]=min(accepted,key=lambda row:(row['cross_entropy'],row['tensor_bytes'],row['solution_id']))['solution_id']
check('original_ordered_selection',winners==r['finalist_solution_ids']==selected['finalists'] and selected['source_report_sha256']==r['parent_report_sha256'])
tape=read('recovery_tape.json');updates=tape['updates'];frozen_tape=tuple((tuple(row['row_ids']),row['augmentation_seed']) for row in updates)
check('shared_200_update_tape',len(updates)==200 and digest(frozen_tape)==tape['sha256'] and all(len(row['row_ids'])==128 and set(row['row_ids'])<=parts[0] for row in updates))
generator=np.random.default_rng(990032);ordered=generator.permutation(np.asarray(roles['train'],dtype=np.int64))
expected=tuple((tuple(map(int,ordered[start:start+128])),int(generator.integers(0,2**31))) for start in range(0,25600,128))
check('independent_recovery_rng',expected==frozen_tape)
for identity,artifact in recovered.items():
    check('recovery_roles:'+identity,artifact['roles']==selected['roles'][identity] and artifact['recovery_updates']==200 and artifact['phase']=='recovered_200_updates' and artifact['recovery_tape_sha256']==tape['sha256'])
    recount(artifact['validation'],roles['search_validation'],'recovery_'+identity)
for key,artifact in artifacts.items():
    record=inventory[artifact['path']]
    check('artifact_identity:'+key,key==digest({'solution_id':artifact['solution_id'],'checkpoint_sha256':artifact['sha256'],'phase':artifact['phase']}))
    check('actual_tensor_storage:'+key,physical_bytes(genotypes[artifact['solution_id']])==artifact['tensor_bytes']==record['torch_zip_storage_bytes'])
    check('checkpoint_integrity:'+key,record['bytes']==artifact['file_bytes'] and record['sha256']==artifact['sha256'] and artifact['gpu_reload_exact'] and artifact['completed_within_deadline'])
seal=read('frozen_selection.json');admission=read('selected_artifacts_admission.json')
check('durable_six_artifact_seal',seal['all_artifacts']==artifacts and len(seal['complete_recovery_sha256'])==6 and set(map(tuple,seal['complete_recovery_sha256']))=={(key,value['sha256']) for key,value in artifacts.items()} and admission['artifact_count']==6)
check('role_seal',dict(seal['roles'])=={'dense_original':baseline['artifact_id'],**{role:a['artifact_id'] for a in recovered.values() for role in a['roles']}})
check('test_opened_after_selection',r['test_opened_after_frozen_selection'] is True and inventory['frozen_selection.json']['mtime_ns']<=min(inventory[v['prediction_file']]['mtime_ns'] for v in r['official_test'].values()))
test={};test_targets=[]
for key,metrics in r['official_test'].items():
    test[key],arrays=recount(metrics,range(10000),'official_'+key);test_targets.append(arrays['labels'])
check('same_independent_test',len(test)==6 and set(test)==set(artifacts) and all(np.array_equal(a,test_targets[0]) for a in test_targets))
dense=next(a for a in recovered.values() if 'dense_recovery' in a['roles'])
threshold=test[dense['artifact_id']]['top1']-.02
check('frozen_acceptance_threshold',math.isclose(threshold,r['final_quality_threshold'],abs_tol=1e-12))
rows=[]
for key,a in artifacts.items():
    values=test[key];accepted=a['tensor_bytes']<=35787526 and values['top1']>=threshold
    check('acceptance:'+key,r['final_acceptance'][key]=={'accepted':accepted,'quality_accepted':values['top1']>=threshold,'storage_accepted':a['tensor_bytes']<=35787526})
    rows.append({'artifact_id':key,'solution_id':a['solution_id'],'roles':a.get('roles',['dense_original']),
                 'tensor_bytes':a['tensor_bytes'],'fraction_M0':a['tensor_bytes']/44734408,'file_bytes':a['file_bytes'],
                 **values,'delta_to_recovered_dense_pp':100*(values['top1']-test[dense['artifact_id']]['top1']),
                 'delta_to_M0_pp':100*(values['top1']-test[baseline['artifact_id']]['top1']),
                 'accepted':accepted,'within_2pp_of_M0':values['top1']>=test[baseline['artifact_id']]['top1']-.02})

aa_rows=[]
def verify_aa(record,label):
    for batch,values in record['batches'].items():
        check('aa_orders:'+label+':'+batch,len(values['pairs'])==20 and values['orders']==['AB','BA']*10)
        admissions=[]
        for clock in ('cuda_event','synchronized_wall'):
            ratios=[pair['B'][clock+'_seconds']/pair['A'][clock+'_seconds'] for pair in values['pairs']]
            actual=aa_admission(ratios,values['orders']);admissions.append(actual['accepted'])
            check('aa_recount:'+label+':'+batch+':'+clock,actual==values[clock+'_admission'] and all(math.isclose(ratio,pair[clock+'_ratio_B_over_A'],abs_tol=1e-12) for ratio,pair in zip(ratios,values['pairs'])))
            aa_rows.append({'label':label,'batch':int(batch),'clock':clock,**actual})
        check('aa_batch_gate:'+label+':'+batch,values['accepted']==all(admissions))
    check('aa_full_gate:'+label,record['accepted']==all(v['accepted'] for v in record['batches'].values()))
verify_aa(r['initial_aa'],'initial_M0')
for key,value in r['latency'].items():verify_aa(value['AA'],key)
reference=r['initial_aa']['accepted'] and r['latency'][baseline['artifact_id']]['AA']['accepted']
for key,value in r['latency'].items():
    admitted=reference and value['AA']['accepted']
    check('speed_gate:'+key,value['comparison_admitted']==admitted and ('AB' in value)==admitted)
    ts=value['torchscript'];check('timing_loaded_file:'+key,ts['sha256']==inventory[ts['path']]['sha256'] and ts['file_bytes']==inventory[ts['path']]['bytes'])
check('no_compressed_speed_claim',not any(v['comparison_admitted'] for key,v in r['latency'].items() if key!=baseline['artifact_id']))
ledger=[json.loads(line) for line in (raw/'ledger.jsonl').read_text().splitlines()]
check('completion_only_stages',[v['stage'] for v in ledger if v.get('event')=='stage_started']==['manifest','recovery','final','close'])
check('1000_SGD_updates',sum(a['recovery_updates'] for a in recovered.values())==1000)
failed=[c for c in checks if not c['passed']]
summary={'schema':'fedcore-completion-analysis-v1','status':r['status'],'verification_consistent':not failed,
         'checks':len(checks),'failed_checks':failed,'rows':rows,'AA':aa_rows,
         'prediction_files':prediction_files,'prediction_rows':prediction_rows,
         'newly_computed_prediction_rows':prediction_rows-5000,
         'source_commit':source['source_commit'],'wheel_sha256':source['wheel_sha256'],
         'completion_seconds':guard['elapsed_seconds'],'prior_seconds':guard['prior_charged_seconds'],
         'total_seconds':guard['total_charged_seconds'],'initial_AA':r['initial_aa']['accepted'],
         'compact_accepted':sum(a['accepted'] for a in rows if a['tensor_bytes']<44734408),
         'compact_within_2pp_of_M0':sum(a['within_2pp_of_M0'] for a in rows if a['tensor_bytes']<44734408),
         'peak_allocated_bytes':r['peak_allocated_bytes'],'peak_reserved_bytes':r['peak_reserved_bytes'],
         'peak_rss_bytes':inner['peak_rss_bytes']}
for name,value in (('analysis.json',summary),('checks.json',checks)):
    (args.output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('status','verification_consistent','checks','failed_checks','prediction_files','prediction_rows','compact_accepted','compact_within_2pp_of_M0')}))
raise SystemExit(bool(failed))
