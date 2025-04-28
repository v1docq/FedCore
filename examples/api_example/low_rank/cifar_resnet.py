from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate, LowRankTemplate)
from fedcore.data.dataloader import load_data
from fedcore.api.main import FedCore

##########################################################################
### DEFINE ML PROBLEM (classification, object_detection, regression,   ###
### ts_forecasting), PEFT problem (pruning, quantisation, distillation,###
### low_rank) and appropriate loss function both for model and compute ###
##########################################################################
METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'pruning'
INITIAL_ASSUMPTION = {'path_to_model': 'pretrain_models/pretrain_model_checkpoint_at_15_epoch.pt',
                      'model_type': 'ResNet18'}
INITIAL_MODEL = 'ResNet18'
PRETRAIN_SCENARIO = 'from_checkpoint'
POP_SIZE = 1
DATASET = 'CIFAR10'
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

fedot_config = FedotConfigTemplate(problem=PROBLEM,
                                   metric=METRIC_TO_OPTIMISE,
                                   pop_size=POP_SIZE,
                                   timeout=1,
                                   initial_assumption=INITIAL_ASSUMPTION)

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

peft_config = LowRankTemplate(
    eval_each=5,
    log_each=1,
    strategy='explained_variance',
    rank_prune_each=8,
    non_adaptive_threshold=0.3,
    custom_criterions={'hoer': 10}
)

learning_config = LearningConfigTemplate(criterion=LOSS,
                                         learning_strategy=PRETRAIN_SCENARIO,
                                         peft_strategy=PEFT_PROBLEM,
                                         peft_strategy_params=peft_config)

api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)

if __name__ == "__main__":
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset(DATASET, train_dataloader_params,
                                                                   test_dataloader_params)
    fedcore_compressor.fit(fedcore_train_data)
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    _ = 1