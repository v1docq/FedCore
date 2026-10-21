"""Offline bounded MoE mechanisms; no LLM quality or speedup claim.

python examples/sparse_compression/moe_demo.py --output artifacts/sparse_moe
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fedcore.algorithm.pruning.sparse.moe import (
    ReferenceMoE, remove_experts, intervention_scores, genetic_expert_candidates,
    kt_block_search, compress_experts, prepare_expert_recovery,
)
from fedcore.algorithm.low_rank.factor_recovery import finish_factor_lora_stage
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.experiments.measurement import measure_artifact, tensor_state_bytes
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(101)
    source = ReferenceMoE(16,32,6,2,shared=True).eval()
    calibration = torch.randn(24,8,16)
    heldout = torch.randn(8,8,16)
    # This is synthetic teacher-output MSE, explicitly not next-token loss.
    target = source(calibration).detach()
    candidates, search = genetic_expert_candidates(source,calibration,4,
        best_k=2,population=6,generations=3,max_evaluations=12,seed=29)
    selected, joint = kt_block_search((candidates,),lambda maps:
        float((remove_experts(source,maps[0])[0](calibration)-target).square().mean()))
    removed, removal = remove_experts(source,selected[0])
    compressed, factorization = compress_experts(removed,calibration,(4,)*4)
    stage = prepare_expert_recovery(compressed,factor='left',adapter_rank=2)
    split = TensorSplit(calibration,target,tuple(f'calibration-{i}' for i in range(len(calibration))))
    training = train_model(stage.model,split,'regression',epochs=1,batch_size=4,
                           learning_rate=.003,max_updates=4,optimizer_validator=stage.validate_optimizer)
    recovered, recovery = finish_factor_lora_stage(stage)
    models = {"dense":source,"removed":removed,"factorized":compressed,"recovered":recovered}
    records = {}
    manager = CheckpointManager(str(output),auto_cleanup=False)
    reference = source(heldout).detach()
    for name,model in models.items():
        payload = manager.serialize_to_bytes(model)
        restored = manager.restore(torch.load(io.BytesIO(payload),weights_only=True))
        torch.testing.assert_close(restored(heldout),model(heldout))
        checkpoint = output/f'{name}.pt'
        checkpoint.write_bytes(payload)
        measurement = measure_artifact(restored,heldout,output/f'{name}.ts',repeats=10,warmup=2)
        if measurement['status'] != 'succeeded':
            raise RuntimeError(f'{name}: {measurement}')
        records[name] = {"heldout_teacher_mse":float((model(heldout)-reference).square().mean()),
                         "tensor_state_bytes":tensor_state_bytes(model),
                         "checkpoint_file_bytes":checkpoint.stat().st_size,
                         "stored_experts":model.num_experts,"active_experts_per_token":model.top_k,
                         "shared_experts":int(model.shared is not None),"artifact":measurement}
    report = {"scenario":"offline_synthetic_moe_mechanism_v1","seed":101,
              "quality_metric":"heldout_synthetic_teacher_output_mse_not_language_perplexity",
              "search":search,"kt":joint,"removal":removal,"factorization":factorization,
              "training":training,"recovery":recovery,"records":records,
              "limitations":["small_cpu_reference","no_llm_quality_claim","no_gpu_kernel_claim",
                              "no_expected_latency_reduction_with_same_top_k"]}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{'tensor_state_bytes':v['tensor_state_bytes'],
                         'heldout_teacher_mse':v['heldout_teacher_mse']} for k,v in records.items()},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/sparse_moe')
    run(parser.parse_args().output)
