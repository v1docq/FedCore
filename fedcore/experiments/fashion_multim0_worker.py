"""Multi-M0 worker, exclusively beneath the shared root guardian."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from . import fashion_worker as w
from .fashion_protocol import default_manifest, source_code_hashes, CONTROLS
from .fashion_comparison_plan import ComparisonPlan
from .quantization_pilot_protocol import digest, file_hash
from .quantization_pilot_guard import Limits, atomic_json


def model_plan(root, seed):
    payload = json.loads((root/'manifest.json').read_text())
    if payload['model']['seed'] != seed or payload['search']['evaluations'] != 24:
        raise ValueError('Unexpected frozen multi-M0 configuration')
    return SimpleNamespace(payload=payload, fingerprint=digest(payload))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','train','control','search','freeze','test'])
    p.add_argument('--run-root',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--job');p.add_argument('--space');p.add_argument('--limits',required=True)
    args=p.parse_args();top=Path(args.run_root);w.require_guard(top)
    limits=Limits.read(args.limits)
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    cfg=json.loads((top/'SERIES.json').read_text());design=ComparisonPlan()
    if args.seed not in design.model_seeds:raise ValueError('Undeclared M0')
    root=top/f'm0-{args.seed}';root.mkdir(exist_ok=True)
    if args.stage=='init':
        if (root/'manifest.json').exists():raise FileExistsError('Frozen M0 exists')
        hashes=cfg['raw_sha256'];payload=default_manifest(cfg['commit'],hashes)
        payload['model']['seed']=args.seed
        payload['training']['minimum_stop_accuracy']=0.0
        payload['search'].update(evaluations=24,evo_population=8,generations=24,max_proposals=192)
        payload['jobs']=list(CONTROLS)+[f'{j.space}__{j.optimizer}_{j.search_seed}' for j in design.jobs() if j.model_seed==args.seed]
        payload['series']='multim0_exploratory_v1'
        payload['test_policy']='previously opened official test; exploratory, once after global selection lock'
        atomic_json(root/'manifest.json',payload)
        atomic_json(root/'code.lock.json',{'source_commit':cfg['commit'],'files':source_code_hashes()})
        # Canonical files are copied, never linked: guard disk accounting remains explicit.
        import shutil
        raw=root/'data'/'raw';raw.mkdir(parents=True)
        for name in hashes:shutil.copyfile(Path(cfg['raw_root'])/name,raw/name)
        plan=model_plan(root,args.seed);result=w.prepare(plan,root)
    else:
        plan=model_plan(root,args.seed);w.verify_code(root)
        if args.stage=='train':result=w.train(plan,root,limits)
        elif args.stage=='control':result=w.run_job(plan,root,args.job,limits)
        elif args.stage=='search':
            declared={f'{j.space}__{j.optimizer}_{j.search_seed}':j for j in design.jobs() if j.model_seed==args.seed}
            job=declared[args.job]
            if job.space!=args.space:raise ValueError('Space/job mismatch')
            path=root/'jobs'/(args.job+'.json')
            if path.exists() or (root/'selection.lock.json').exists():raise FileExistsError('No repeated searches')
            atomic_json(path,{'status':'started','job':args.job,'manifest_sha256':plan.fingerprint})
            trained,source,data=w.load_prepared(plan,root)
            seconds=min(600,limits.remaining()-305)
            if seconds<=0:raise TimeoutError('No search+pick reserve')
            ledger=w.ResourceLedger(w.StageBudgets(max_seconds=seconds,max_training_steps=0,max_stages=8192,max_rss_bytes=w.HARD_RSS))
            artifact,evidence=w.run_search(plan,root,f'{job.optimizer}_{job.space}_{job.search_seed}',source,data,trained['budget_bytes'],ledger,seconds,limits,comparison=design,space_name=job.space)
            result={'job':args.job,'manifest_sha256':plan.fingerprint,'trained_sha256':file_hash(root/'trained.json'),
                    'status':w.classify_search_artifact(evidence.get('search',{}),artifact),'artifact':artifact,**evidence}
            atomic_json(path,w.jsonable(result))
        elif args.stage=='freeze':result=w.freeze_selection(plan,root,limits)
        else:
            locked=json.loads((top/'GLOBAL_SELECTION_LOCK.json').read_text())
            if locked[str(args.seed)]!=file_hash(root/'selection.lock.json'):raise ValueError('Global freeze mismatch')
            result=w.closed_test(plan,root,limits)
    print(json.dumps({'stage':args.stage,'seed':args.seed,'job':args.job,'status':result.get('status','complete')},allow_nan=False),flush=True)

if __name__=='__main__':main()

