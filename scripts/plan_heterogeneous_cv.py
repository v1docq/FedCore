"""Emit the frozen Evo-only manifest. This command never launches computation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fedcore.algorithm.heterogeneous.protocol import ExperimentProtocol, pilot_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    protocol = ExperimentProtocol()
    payload = dict(schema='heterogeneous-evo-only-panel-v1',protocol=asdict(protocol),
        budget=protocol.budget(),pilot=pilot_budget(),
        jobs=[dict(model_seed=m,search_seed=s,arm=a.value) for m,s,a in protocol.jobs()],
        execution='not launched; cluster remains stopped',
        required_hardware_gate='trusted host binding + private whole-tree OS deadline + A100 backend pilot')
    value = json.dumps(payload,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream: stream.write(value)
    else: print(value,end='')


if __name__ == '__main__': main()
