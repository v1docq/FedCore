import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.architecture.utils.misc import count_params
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.repository.constanst_repository import SLRStrategiesEnum
import torch.nn as nn
import torch
from fedcore.tools.metrics.two_model_compairing_metrics import compare_accuracy
from fedcore.tools.dataload.small_cifar10_dataloader import get_small_cifar10_train_and_val_loaders
from fedcore.tools.template_fedcore_models import create_low_rank_with_prune_on_0_epoch
import itertools
from copy import deepcopy
import pandas as pd
from fedcore.algorithm.low_rank.hooks import OnetimeRankPruner
from fedcore.tools.ruler import PerformanceEvaluator
import numpy as np
import matplotlib.pyplot as plt

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)


from fedcore.tools.dataload.small_cifar10_dataloader import get_small_cifar10_train_and_val_loaders
from fedcore.tools.template_fedcore_models import create_low_rank_with_prune_on_0_epoch

train_dataloader, val_dataloader = get_small_cifar10_train_and_val_loaders(batch_size=8, val_size=1000)

TRAINED_LR, data = create_low_rank_with_prune_on_0_epoch(train_dataloader, val_dataloader, epochs=0) #TODO почему то уже даже при одной эпохе качество модели падает с 0.93 до 0.6
# TRAINED_LR.decomposing_mode = "spatial"
TRAINED_LR.fit(input_data=data)
print()

def experiment_body(trained_lr: LowRankModel, bs=8, thr=0.05, mode=SLRStrategiesEnum._member_names_[0], comp="one_layer", device_str="cpu", compare_with_decomposed=False, n_eval=8, eval_only_model_before=False, eval_only_model_after=False):
        """COMPAIRING INIT NON DECOMPOSED MODEL (model_before) WITH MODEL_AFTER
        WITH PRUNING
        """
        print(f"Running experiment bs={bs} thr={thr} mode={mode} compose={comp}")
        # copy model for experiment
        trained_lr = deepcopy(trained_lr)
        #model before - is clean Conv2d model
        if (compare_with_decomposed):
            #if we want to compare decomposed (conv Vh and conv S*U) & decomposed+pruned+compose_mode 1, 2, 3
            trained_lr.model_before = deepcopy(trained_lr.model_after)

        #add correct compose_mode before composition by OneTimeRankPruner
        for name, module in trained_lr.model_after.named_modules():
            if isinstance(module, IDecomposed): 
                 module.compose_mode = comp

        #add pruner and execute
        rank_pruner = OnetimeRankPruner()
        rank_pruner.link_to_trainer(trained_lr.trainer)
        rank_pruner.non_adaptive_threshold = thr
        rank_pruner.SLR_strategy = mode
        rank_pruner.action(None, None)
        #after that we have compressed (by threshold) composed view. 1 2 or 3 4D tensors inside

        #creating performance evaluators
        val_loader_after = deepcopy(val_dataloader)
        device = torch.device(device_str)
        eval_before = PerformanceEvaluator(trained_lr.model_before, data=val_dataloader, batch_size=bs, device=device, n_batches=n_eval)
        eval_after = PerformanceEvaluator(trained_lr.model_after, data=val_loader_after, batch_size=bs, device=device, n_batches=n_eval)
        
        # run evaluation
        if (eval_only_model_before):
            print("eval model_before")
            eval_before.warm_up_cuda()
            thr = eval_before.throughput_eval(1)
            return trained_lr, {}
        if (eval_only_model_after):
            print("eval model_after")
            eval_after.warm_up_cuda()
            thr = eval_after.throughput_eval(1)
            return trained_lr, {}
        
        print("eval model_before")
        metrics_before = eval_before.eval()
        print("eval model_after")
        metrics_after = eval_after.eval()
        acc_before, acc_after = compare_accuracy(trained_lr.model_before, trained_lr.model_after, val_dataloader)

        # store row
        return trained_lr, {
            "batch_size": bs,
            "threshold": thr,
            "low_rank_mode": mode,
            "compose_mode": comp,
            "device": device_str,
            # before metrics
            "before_latency_mean s": metrics_before.get("latency", [None])[0] if isinstance(metrics_before.get("latency"), (list, tuple)) else None,
            "before_latency_std s": metrics_before.get("latency", [None, None])[1] if isinstance(metrics_before.get("latency"), (list, tuple)) else None,
            "before_throughput_mean s": metrics_before.get("throughput", [None])[0] if isinstance(metrics_before.get("throughput"), (list, tuple)) else None,
            "before_throughput_std s": metrics_before.get("throughput", [None, None])[1] if isinstance(metrics_before.get("throughput"), (list, tuple)) else None,
            "before_model_size_mb": metrics_before.get("model_size", [None])[0] if isinstance(metrics_before.get("model_size"), (list, tuple)) else None,
            "before_params": count_params(trained_lr.model_before),
            "before_accuracy": acc_before,
            # after metrics
            "after_params": count_params(trained_lr.model_after),
            "after_latency_mean s": metrics_after.get("latency", [None])[0] if isinstance(metrics_after.get("latency"), (list, tuple)) else None,
            "after_latency_std s": metrics_after.get("latency", [None, None])[1] if isinstance(metrics_after.get("latency"), (list, tuple)) else None,
            "after_throughput_mean s": metrics_after.get("throughput", [None])[0] if isinstance(metrics_after.get("throughput"), (list, tuple)) else None,
            "after_throughput_std s": metrics_after.get("throughput", [None, None])[1] if isinstance(metrics_after.get("throughput"), (list, tuple)) else None,
            "after_model_size_mb": metrics_after.get("model_size", [None])[0] if isinstance(metrics_after.get("model_size"), (list, tuple)) else None,
            "after_accuracy": acc_after,
        }

def full_experiment(trained_lr: LowRankModel, device_str, compare_with_decomposed=False, n_eval=8):
    batch_sizes = [16] #16, 32]                     # можно расширить до 1..128
    thresholds = [0.1]#[0.0001, 0.1, 0.5, 0.9]#0.1, 0.2, 0.4]
    low_rank_modes = [SLRStrategiesEnum._member_names_[0]]
    compose_modes = ["one_layer"] #["one_layer", "two_layers", "three_layers"]


    rows = []
    all_iterations = len(batch_sizes) * len(thresholds) * len(low_rank_modes) * len(compose_modes)

    # loop over parameter grid
    i = 1
    for bs, thr, mode, comp in itertools.product(batch_sizes, thresholds, low_rank_modes, compose_modes):
        print("STEP:", i, "FROM", all_iterations)
        rows.append(experiment_body(trained_lr, bs, thr, mode, comp, device_str, compare_with_decomposed, n_eval)[1])
        print(f"Done device={device_str} bs={bs} thr={thr} comp={comp}")
        i = i + 1
    return rows


#with_flops=True
import pdb
import os
# Set debug mode
# pdb.set_trace()
print(os.getpid())
with profile(activities=[ProfilerActivity.CPU], 
             #with_stack=True, 
             record_shapes=True,
             experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
             ) as prof:
    with record_function("model_inference"):
        ret_model, rows = experiment_body(TRAINED_LR, bs=16, thr=0.2, mode=SLRStrategiesEnum._member_names_[0], comp="two_layers", device_str="cpu", compare_with_decomposed=False, n_eval=8)
