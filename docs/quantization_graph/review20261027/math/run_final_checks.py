import json,subprocess,hashlib,os,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parent
repo=Path(__file__).resolve().parents[4]
python=sys.executable
env=dict(os.environ,PYTHONPATH=str(repo))
scripts=["verify_ldl.py","verify_formats_lplr.py","verify_tender.py","verify_public_extremes.py","verify_moment_underflow.py"]
def run(name):
    done=subprocess.run([python,str(root/name)],cwd=repo,env=env,capture_output=True,text=True,encoding="utf-8")
    (root/(name+".stdout.txt")).write_text(done.stdout,encoding="utf-8")
    (root/(name+".stderr.txt")).write_text(done.stderr,encoding="utf-8")
    return {"script":name,"exit_code":done.returncode}
with ThreadPoolExecutor(max_workers=4) as pool:runs=list(pool.map(run,scripts))
files=["fedcore/algorithm/quantization/"+name+".py" for name in ["matrix_transforms","tensor_formats","portable_layers","tender","process","graph_specs"]]
files+=["fedcore/algorithm/low_rank/approximation.py","fedcore/algorithm/low_rank/structured_profiles.py"]
out={"evidence_version":"v2_moment_underflow","baseline_head":"50f1cdf51b9f65ea8a5d2b868b7ed46a35738f4c","head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),"runs":runs,"sha256":{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files}}
out["results"]={name:json.loads((root/name).read_text(encoding="utf-8")) for name in ["verify_ldl_results.json","verify_formats_lplr_results.json","verify_tender_results.json","verify_public_extremes_results.json","verify_moment_underflow_results.json"]}
ldl=out["results"]["verify_ldl_results.json"]
out["all_passed"]=all(r["exit_code"]==0 for r in runs) and ldl["ldl_passed"]==ldl["ldl_total"] and ldl["ldlq_passed"]==len(ldl["ldlq_checks"]) and all(r["passed"]==r["total"] for r in out["results"].values() if "passed" in r)
(root/"final_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"runs":runs,"totals":{"ldl":[out["results"]["verify_ldl_results.json"]["ldl_passed"],out["results"]["verify_ldl_results.json"]["ldl_total"]],"ldlq":[out["results"]["verify_ldl_results.json"]["ldlq_passed"],20],**{name:[result["passed"],result["total"]] for name,result in out["results"].items() if "passed" in result}},"sha256":out["sha256"]},indent=2))
raise SystemExit(0 if out["all_passed"] else 1)
