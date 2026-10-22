"""Independent post-run recount: NumPy/standard library, no FedCore or Torch.

Run only after the supervised GPU process has stopped. Reads a copied evidence
directory and the post-run remote file/ZIP inventory. Never evaluates a model.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_bytes(genotype):
    """Independent CIFAR ResNet18 tensor count, including BN running buffers."""
    planes=(64,64,128,128,256,256,512,512)
    incoming=(64,64,64,128,128,256,256,512)
    paths=tuple(f'layer{stage}.{block}' for stage in range(1,5) for block in range(2))
    choices={gene['aliases'][0]:gene for gene in genotype['genes']}
    total=4*(64*3*9)+16*64+8+4*(512*10+10)
    for path,outer,ins in zip(paths,planes,incoming):
        gene=choices[path]
        width=gene['choice'].get('keep_channels',outer)
        rank=gene['choice'].get('rank')
        total+=4*width*ins*9+16*width+8
        total+=4*(outer*width*9 if rank is None else rank*width*9+outer*rank)
        total+=16*outer+8
        if ins!=outer:
            total+=4*outer*ins+16*outer+8
    return total


def dense_genotype():
    return {'schema_version':1,'profile':'cv_internal_channels_v1','genes':[
        {'aliases':[f'layer{stage}.{block}'],'recipe':'DenseCV','choice':{},'recovery':'none'}
        for stage in range(1,5) for block in range(2)]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('evidence',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    base=args.evidence.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    checks=[]
    def check(name,condition,details=None):
        checks.append({'check':name,'passed':bool(condition),'details':details})
    def read(relative,default=None):
        path=base/relative
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default
    worker_report=read('report.json')
    report=worker_report or {}
    protocol=read('protocol.json',{})
    supervision=read('pilot-1-supervision.json',{})
    receipt=read('launch-receipt-2026-10-22.json',{})
    files=read('remote-file-inventory.json',{})
    filemap=files.get('files',{})
    ledger=[json.loads(line) for line in (base/'ledger.jsonl').read_text().splitlines() if line]
    heartbeats=[json.loads(line) for line in (base/'heartbeats.jsonl').read_text().splitlines() if line]
    source=read('source_manifest.json',{})
    if 'supplied' in source: source=source['supplied']
    if worker_report is None:
        # A SIGKILL prevents the worker's finally block. This derived record is
        # written under analysis/, never backfilled as a fabricated raw report.
        report={'status':'supervisor_interrupted_baseline','stop_reason':supervision.get('reason'),
            'failure':None,'protocol_sha256':supervision.get('protocol_sha256'),
            'wall_seconds':supervision.get('elapsed_seconds'),
            'peak_allocated_bytes':max((row.get('peak_allocated_bytes',0) for row in heartbeats),default=0),
            'peak_reserved_bytes':max((row.get('peak_reserved_bytes',0) for row in heartbeats),default=0)}
        check('interruption_is_external',supervision.get('reason')=='external_gpu_contention'
              and supervision.get('returncode')==-9 and supervision.get('gpu_process_count',0)>1)
        check('no_later_stages',[row['stage'] for row in ledger if row.get('event')=='stage_started']==['manifest','baseline'])
    check('protocol_digest',digest(protocol)==report.get('protocol_sha256')==receipt.get('protocol_sha256'))
    check('source_commit',source.get('source_commit')==receipt.get('source_commit')=='3bc0825a9ef1bcd0ac8eb0181b6d5314b57ebb8d')
    check('wheel_identity',source.get('wheel_sha256')==receipt.get('wheel_sha256'))
    check('supervision_terminal',supervision.get('status') in ('finished','stopped'))
    check('free_gpu_admission',supervision.get('preflight_gpu_processes')==[])
    check('slot_limit',report.get('wall_seconds',math.inf)<=14400 and supervision.get('elapsed_seconds',math.inf)<=14400)
    check('saved_pytorch_counters_within_limit',report.get('peak_allocated_bytes',math.inf)<=36*1024**3 and report.get('peak_reserved_bytes',math.inf)<=36*1024**3)
    check('rss_limit',supervision.get('peak_rss_bytes',math.inf)<=16*1024**3)
    check('no_remaining_owned_processes',not files.get('remaining_owned_pids',[]))
    roles=read('data_roles.json',{})
    if roles:
        sets=[set(roles[name]) for name in ('train','calibration','search_validation')]
        check('data_partition_sizes',[len(values) for values in sets]==[44000,1000,5000])
        check('training_partition_complete',set.union(*sets)==set(range(50000)) and sum(map(len,sets))==50000)
        if report.get('data',{}).get('roles_sha256'):
            check('data_roles_digest',digest(roles)==report['data']['roles_sha256'])
        check('split_seed',roles.get('seed')==protocol['split_seed'])
    epochs=[row for row in ledger if row.get('event')=='baseline_epoch']
    baseline=read('baseline.json')
    if baseline:
        check('baseline_80_full_epochs',[row['epoch'] for row in epochs]==list(range(1,81)) and all(row['rows']==44000 for row in epochs))
        check('baseline_update_count',baseline['training']['updates']==80*math.ceil(44000/128))
        check('baseline_training_admission',baseline['validation']['top1']>=.9 and baseline['training']['all_parameters_received_finite_gradients'] and baseline['training']['all_parameter_tensors_changed'])
        check('baseline_physical_bytes',baseline['tensor_bytes']==physical_bytes(dense_genotype()))
    if worker_report is None:
        expected_partials=[f'partial/baseline_epoch_{epoch:03d}.pt' for epoch in range(20,len(epochs)+1,20)]
        check('partial_weight_snapshots',bool(expected_partials) and all(name in filemap for name in expected_partials))
        for name in expected_partials:
            check('partial_storage_payload:'+name,filemap.get(name,{}).get('torch_zip_storage_bytes')==physical_bytes(dense_genotype()))
        check('completed_epoch_integrity',[row['epoch'] for row in epochs]==list(range(1,len(epochs)+1))
              and all(row['rows']==44000 and row['updates']==row['epoch']*math.ceil(44000/128) for row in epochs))
    for relative,record in filemap.items():
        path=base/relative
        if path.is_file():
            check('copied_file:'+relative,path.stat().st_size==record['bytes'] and checksum(path)==record['sha256'])
    prediction_counts={'files':0,'rows':0}
    def predictions(record,namespace):
        relative=record.get('prediction_file')
        if not relative: return
        path=(base/relative).resolve()
        check('prediction_path:'+relative,path.is_relative_to(base) and path.is_file())
        if not path.is_file(): return
        check('prediction_hash:'+relative,checksum(path)==record['prediction_sha256'])
        with np.load(path,allow_pickle=False) as data:
            rowids,labels,predicted,loss=[data[key] for key in ('row_ids','labels','predictions','cross_entropy')]
        expected=roles.get('search_validation',[]) if namespace=='validation' else list(range(10000))
        n=len(rowids)
        correct=int(np.count_nonzero(labels==predicted))
        ce=math.fsum(float(value) for value in loss)/n
        check('prediction_rows:'+relative,rowids.tolist()==expected and n==record['rows'])
        check('prediction_values:'+relative,np.isfinite(loss).all() and np.all((labels>=0)&(labels<10)) and np.all((predicted>=0)&(predicted<10)))
        check('prediction_recount:'+relative,correct==record['correct'] and correct/n==record['top1'] and math.isclose(ce,record['cross_entropy'],rel_tol=1e-11,abs_tol=1e-11))
        prediction_counts['files']+=1
        prediction_counts['rows']+=n
    if baseline: predictions(baseline['validation'],'validation')
    attempts=report.get('all_short_attempts',{})
    genotypes={digest(dense_genotype()):dense_genotype()}
    attempt_rows=[]
    selected={}
    byte_ceiling=math.floor(.8*physical_bytes(dense_genotype()))
    for run,records in attempts.items():
        for record in records:
            key=record['solution_id']
            genotypes[key]=record['genotype']
            check('canonical_id:'+run+':'+key,digest(record['genotype'])==key)
            check('timely_completion:'+run+':'+key,not record['completed_within_deadline'] or record['ended_at']<=min(record['deadline'],report['absolute_deadline']))
            late_journal=any(event.get('event')=='short_deadline_excluded_after_journal' and event.get('run')==run
                             and event.get('solution_id')==key for event in ledger)
            check('journal_overrun_exclusion:'+run+':'+key,not late_journal or record['completed_within_deadline'] is False)
            if 'tensor_bytes' in record:
                check('physical_bytes:'+run+':'+key,physical_bytes(record['genotype'])==record['tensor_bytes'])
            predictions(record,'validation')
            attempt_rows.append({'run':run,'attempt_id':record['attempt_id'],'solution_id':key,'status':record['status'],
                'tensor_bytes':record.get('tensor_bytes'),'cost_ratio':record.get('cost_ratio'),
                'cross_entropy':record.get('cross_entropy'),'top1':record.get('top1'),
                'wall_seconds':record['wall_seconds'],'timely':record['completed_within_deadline'],
                'under_ceiling':record.get('tensor_bytes',math.inf)<=byte_ceiling})
        if run in ('random','evolution'):
            check('charged_evaluations:'+run,len(records)<=12 and len({row['solution_id'] for row in records})==len(records))
            candidates=[row for row in records if row['status']=='succeeded' and row['completed_within_deadline'] is True
                        and row.get('quality_accepted') is True and row.get('tensor_bytes',math.inf)<=byte_ceiling
                        and math.isfinite(row.get('cross_entropy',math.inf))]
            candidates.sort(key=lambda row:(row['cross_entropy'],row['tensor_bytes'],row['solution_id']))
            selected[run]=candidates[0]['solution_id'] if candidates else None
    if 'random' in attempts and 'evolution' in attempts:
        check('independent_prefix4',[row['genotype'] for row in attempts['random'][:4]]==[row['genotype'] for row in attempts['evolution'][:4]] and min(len(attempts['random']),len(attempts['evolution']))>=4)
    if report.get('finalist_solution_ids') is not None:
        check('finalist_recount',selected==report['finalist_solution_ids'])
        check('ceiling_recount',byte_ceiling==report['byte_ceiling'])
    for run in ('random','evolution'):
        result=read('search/'+run+'.json')
        if result:
            check('proposal_budget:'+run,result['created_proposals']<=48)
            check('evaluation_charge:'+run,result['charged_evaluations']==len(attempts.get(run,[])))
            if run=='evolution':
                check('actual_evolution_engine',result['engine']=='FedcoreEvoOptimizer' and result['strict_deadline'] is True)
                by_id={row['solution_id']:row for row in attempts.get(run,[])}
                for event in result['trace']:
                    if event['event']=='evaluated' and event['canonical_solution_id'] in by_id:
                        row=by_id[event['canonical_solution_id']]
                        check('common_deadline:'+row['solution_id'],not row['completed_within_deadline'] or event['completed_within_deadline'] is True)
    recovered=report.get('recovered',{})
    tape=read('recovery_tape.json')
    if tape:
        rawtape=[(entry['row_ids'],entry['augmentation_seed']) for entry in tape['updates']]
        check('tape_digest',digest(rawtape)==tape['sha256'] and len(rawtape)==200 and all(len(ids)==128 for ids,_seed in rawtape))
        check('tape_train_only',all(set(ids)<=set(roles['train']) for ids,_seed in rawtape))
    artifacts={}
    if baseline: artifacts[baseline['artifact_id']]=baseline
    for solution,artifact in recovered.items():
        artifacts[artifact['artifact_id']]=artifact
        check('recovery_complete:'+solution,artifact['recovery_updates']==200 and tape and artifact['recovery_tape_sha256']==tape['sha256'])
        check('recovery_physical:'+solution,solution in genotypes and physical_bytes(genotypes[solution])==artifact['tensor_bytes'])
        predictions(artifact['validation'],'validation')
    if recovered: check('recovery_cap',len(recovered)<=6)
    seal=read('frozen_selection.json')
    if seal:
        check('sealed_artifact_set',dict(seal['complete_recovery_sha256'])=={key:row['sha256'] for key,row in artifacts.items()})
        check('test_opened_after_seal',report.get('test_opened_after_frozen_selection') is True)
    for key,artifact in artifacts.items():
        check('artifact_identity:'+key,key==digest({'solution_id':artifact['solution_id'],'checkpoint_sha256':artifact['sha256'],'phase':artifact['phase']}))
        proof=filemap.get(artifact['path'],{})
        check('checkpoint_bytes:'+key,proof.get('sha256')==artifact['sha256'] and proof.get('bytes')==artifact['file_bytes'])
        check('checkpoint_physical_storage:'+key,proof.get('torch_zip_storage_bytes')==artifact['tensor_bytes'])
        check('checkpoint_gpu_reload:'+key,artifact.get('gpu_reload_exact') is True)
    official=report.get('official_test',{})
    final_rows=[]
    dense_recovery=next((row for row in recovered.values() if 'dense_recovery' in row['roles']),None)
    threshold=None if dense_recovery is None or not official else official[dense_recovery['artifact_id']]['top1']-.02
    for key,metrics in official.items():
        predictions(metrics,'test')
        artifact=artifacts[key]
        acceptance={'storage_accepted':artifact['tensor_bytes']<=byte_ceiling,'quality_accepted':metrics['top1']>=threshold,
                    'accepted':artifact['tensor_bytes']<=byte_ceiling and metrics['top1']>=threshold}
        check('final_acceptance:'+key,acceptance==report['final_acceptance'][key])
        final_rows.append({'artifact_id':key,'solution_id':artifact['solution_id'],'roles':artifact.get('roles',['dense_original']),
            'tensor_bytes':artifact['tensor_bytes'],'ratio':artifact['tensor_bytes']/physical_bytes(dense_genotype()),
            'file_bytes':artifact['file_bytes'],'top1':metrics['top1'],'cross_entropy':metrics['cross_entropy'],
            'delta_pp_vs_dense_recovery':100*(metrics['top1']-(threshold+.02)),**acceptance})
    def latency_admission(record,label):
        passed=[]
        for batch,data in record.get('batches',{}).items():
            pairs=data['pairs']
            orders=[pair['order'] for pair in pairs]
            check('timing_pair_count:'+label+':'+batch,len(pairs)==20 and orders==['AB','BA']*10)
            for clock in ('cuda_event','synchronized_wall'):
                field=clock+'_seconds'
                ratios=[pair['B'][field]/pair['A'][field] for pair in pairs]
                check('timing_raw_ratio:'+label+':'+batch+':'+clock,
                    all(pair['A'][field]>0 and pair['B'][field]>0 and math.isfinite(ratio)
                        and math.isclose(ratio,pair[clock+'_ratio_B_over_A'],rel_tol=1e-12) for pair,ratio in zip(pairs,ratios)))
                median=statistics.median(ratios)
                inside=sum(.85<=value<=1.15 for value in ratios)
                order_ratio=statistics.median(ratios[::2])/statistics.median(ratios[1::2])
                accepted=.95<=median<=1.05 and inside>=18 and .90<=order_ratio<=1.10
                stored=data[clock+'_admission']
                check('aa_recount:'+label+':'+batch+':'+clock,accepted==stored['accepted']
                    and inside==stored['inside_15_percent'] and math.isclose(median,stored['median_ratio'],rel_tol=1e-12)
                    and math.isclose(order_ratio,stored['ab_ba_median_ratio'],rel_tol=1e-12))
                passed.append(accepted)
        result=bool(passed) and all(passed)
        check('aa_overall:'+label,result==record.get('accepted',False))
        return result
    initial=report.get('initial_aa',{})
    initial_pass=latency_admission(initial,'initial') if initial.get('batches') else False
    latency_rows=[]
    latency=report.get('latency',{})
    baseline_id=None if baseline is None else baseline['artifact_id']
    reference_pass=bool(latency.get(baseline_id,{}).get('AA',{}).get('accepted',False))
    for key,result in latency.items():
        if key=='incomplete': continue
        own_pass=latency_admission(result['AA'],'final:'+key)
        admitted=initial_pass and reference_pass and own_pass
        check('ab_admission:'+key,result['comparison_admitted']==admitted and ('AB' in result)==admitted)
        for batch,observation in result.get('AB',{}).get('batches',{}).items():
            pairs=observation['pairs']
            check('ab_pairs:'+key+':'+batch,len(pairs)==20 and [pair['order'] for pair in pairs]==['AB','BA']*10)
            for clock in ('cuda_event','synchronized_wall'):
                field=clock+'_seconds'
                ratios=[pair['B'][field]/pair['A'][field] for pair in pairs]
                check('ab_ratios:'+key+':'+batch+':'+clock,all(value>0 and math.isfinite(value)
                    and math.isclose(value,pair[clock+'_ratio_B_over_A'],rel_tol=1e-12) for pair,value in zip(pairs,ratios)))
                ratio=statistics.median(ratios)
                latency_rows.append({'artifact_id':key,'batch':int(batch),'clock':clock,'pairs':len(pairs),
                    'reference_ms':1000*statistics.median(pair['A'][field] for pair in pairs),
                    'candidate_ms':1000*statistics.median(pair['B'][field] for pair in pairs),
                    'median_paired_B_over_A':ratio,'observed_speedup':1/ratio,
                    'time_reduction_percent':100*(1-ratio),'min_pair_ratio':min(ratios),'max_pair_ratio':max(ratios)})
    summary={'schema':'fedcore-a100-independent-recount-v1','status':report.get('status'),
        'verification_consistent':all(item['passed'] for item in checks),'checks':len(checks),
        'worker_report_present':worker_report is not None,
        'peak_memory_scope':'worker final CUDA counters' if worker_report else 'largest saved cumulative heartbeat counters; not a final lifetime peak',
        'completed_epoch_updates':sum(row['updates']-(epochs[index-1]['updates'] if index else 0) for index,row in enumerate(epochs)),
        'last_observed_updates':max((row.get('updates',0) for row in heartbeats),default=0),
        'failed_checks':[item for item in checks if not item['passed']],
        'elapsed_seconds':report.get('wall_seconds'),'supervised_seconds':supervision.get('elapsed_seconds'),
        'epochs':epochs,'peak_allocated_bytes':report.get('peak_allocated_bytes'),
        'peak_reserved_bytes':report.get('peak_reserved_bytes'),'peak_rss_bytes':supervision.get('peak_rss_bytes'),
        'source_tensor_bytes':physical_bytes(dense_genotype()),'byte_ceiling':byte_ceiling,
        'attempt_counts':{run:dict(Counter(row['status'] for row in records)) for run,records in attempts.items()},
        'selected_finalists':selected,'prediction_recounts':prediction_counts,
        'final_rows':final_rows,'initial_aa':report.get('initial_aa'),'latency':report.get('latency',{}),
        'latency_rows':latency_rows,
        'stop_reason':report.get('stop_reason'),'failure':report.get('failure'),
        'limitations':(['one trained model seed and one search seed'] if worker_report else ['one configured model seed; baseline incomplete; search not started'])+['CE on search-validation is a search objective, not independent test',
                       'no statistical superiority or CUDA sparse-kernel claim']}
    outputs=[('analysis.json',summary),('checks.json',checks),('attempts.json',attempt_rows)]
    if worker_report is None:outputs.append(('derived-interruption-outcome.json',report))
    for name,value in outputs:
        (args.output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    if attempt_rows:
        with (args.output/'attempts.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=attempt_rows[0].keys());writer.writeheader();writer.writerows(attempt_rows)
    print(json.dumps({key:summary[key] for key in ('status','verification_consistent','checks','failed_checks','prediction_recounts','attempt_counts')}))
    return 0 if summary['verification_consistent'] else 1


if __name__=='__main__':
    raise SystemExit(main())
