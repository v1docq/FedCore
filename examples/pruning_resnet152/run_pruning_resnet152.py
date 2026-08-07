import sys
import os
import torch

from pathlib import Path 

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate, ModelArchitectureConfigTemplate,
                                     TrainingTemplate, PruningTemplate)
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.data.dataloader import load_data
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from torchvision import models

##########################################################################
### DEFINE ML PROBLEM (classification, object_detection, regression,   ###
### ts_forecasting), PEFT problem (pruning, quantisation, distillation,###
### low_rank) and appropriate loss function both for model and compute ###
##########################################################################

METRIC_TO_OPTIMISE = ['MulticlassAccuracy__10', 'Latency', 
                       'ModelSize']
LOSS = 'cross_entropy'
PROBLEM = 'classification'

pretrained_resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
pretrained_resnet152.fc = torch.nn.Linear(2048, 10) 
INITIAL_ASSUMPTION = pretrained_resnet152 

train_dataloader_params = {"batch_size": 64,
                           'shuffle': True,
                           'is_train': True,
                           'data_type': 'table',
                           'split_ratio': [0.8, 0.2]}
test_dataloader_params = {"batch_size": 100,
                          'shuffle': True,
                          'is_train': False,
                          'data_type': 'table'}

def load_benchmark_dataset(dataset_name, train_dataloader_params, test_dataloader_params):
    fedcore_train_data = load_data(source=dataset_name, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=dataset_name, loader_params=test_dataloader_params)
    return fedcore_train_data, fedcore_test_data


################################################################################
### CREATE SCENARIO FOR FEDCORE AGENT (TRAIN AND OPTIMISE MODEL FROM SCRATCH ###
### or optimise pretrained model with PEFT strategies                        ###
################################################################################

fedot_config = FedotConfigTemplate(problem='classification',
                                   metric=METRIC_TO_OPTIMISE,
                                   pop_size=1,
                                   timeout=1,
                                   initial_assumption=INITIAL_ASSUMPTION)

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

finetune_config = TrainingTemplate(epochs=3,
                                            log_each=3,
                                            eval_each=3,
                                            )


peft_config = PruningTemplate(importance="magnitude",
                              prune_each=5,
                              epochs=9,
                              save_each=0,
                              eval_each=5,
                              pruning_ratio=0.4,
                              )

learning_config = LearningConfigTemplate(criterion='cross_entropy',
                                         learning_strategy='from_checkpoint',
                                         peft_strategy_params=[peft_config])

api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)

if __name__ == "__main__":
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset('CIFAR10', train_dataloader_params,
                                                                   test_dataloader_params)
    fedcore_compressor.fit(fedcore_train_data)
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    print(model_comparison)
    save_path = (REPO_ROOT / 'results' / 'pruning_resnet152/')
    save_path.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(save_path / 'metrics.csv')
    _ = 1