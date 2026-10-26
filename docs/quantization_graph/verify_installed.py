"""Run outside the source checkout after installing its sdist-built wheel.

python -I /checkout/docs/quantization_graph/verify_installed.py /install-target
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

checkout=Path(__file__).resolve().parents[2]
target=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else checkout/'build'/'quantization_install_target_final'
assert not Path.cwd().resolve().is_relative_to(checkout)
os.environ.pop('PYTHONPATH',None)
os.environ['FEDCORE_MODEL_REGISTRY_PATH']=str(checkout/'build'/'installed_registry')
sys.path.insert(0,str(target))
import fedcore
import torch
assert Path(fedcore.__file__).resolve().is_relative_to(target)
from fedcore.experiments.quantization_graph_reference import run_reference
started=time.perf_counter()
example=run_reference(training=True,search=True)
loaded={name:module for name,module in sys.modules.items() if name=='fedcore' or name.startswith('fedcore.')}
assert all(not getattr(module,'__file__',None) or Path(module.__file__).resolve().is_relative_to(target) for module in loaded.values())
files=list((checkout/'fedcore'/'algorithm'/'quantization').glob('*.py'))+[
    checkout/'fedcore'/'tools'/'registry'/'checkpoint_manager.py',
    checkout/'fedcore'/'algorithm'/'low_rank'/'compression_graph.py',
    checkout/'fedcore'/'algorithm'/'low_rank'/'allocation.py',
    checkout/'fedcore'/'experiments'/'quantization_graph_reference.py',
]
hashes={}
for source in files:
    installed=target/source.relative_to(checkout)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(installed.read_bytes()).hexdigest()==digest,(source,installed)
    hashes[source.relative_to(checkout).as_posix()]=digest
report={'status':'passed','profile':'sdist -> wheel -> isolated install-target -> outside-checkout CPU execution',
        'assertions':{'cwd_outside_checkout':True,'source_pythonpath_absent':True,'fedcore_file_under_install_target':True,
                      'all_loaded_fedcore_modules_under_install_target':True,'installed_sources_match_checkout':True},
        'python':sys.version.split()[0],'torch':torch.__version__,'fedcore_version':fedcore.__version__,
        'loaded_fedcore_module_count':len(loaded),'wall_seconds':time.perf_counter()-started,'example':example,'source_hashes':hashes}
output=checkout/'docs'/'quantization_graph'/'results'/'installed_package.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','outside_checkout':True,'module':str(Path(fedcore.__file__).resolve()),'report':str(output)}))
