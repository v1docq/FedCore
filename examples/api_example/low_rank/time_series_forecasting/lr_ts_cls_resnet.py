import os

from fedcore.architecture.utils.paths import PATH_TO_DATA
from fedcore.api.main import FedCore
from fedcore.data.dataloader import load_data

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (
    APIConfigTemplate, AutoMLConfigTemplate,
    LearningConfigTemplate, TrainingTemplate, FedotConfigTemplate,
    LowRankTemplate, ModelArchitectureConfigTemplate)

##########################################################################
### DEFINE ML PROBLEM (classification, object_detection, regression,   ###
### ts_forecasting), PEFT problem (pruning, quantisation, distillation,###
### low_rank) and appropriate loss function both for model and compute ###
##########################################################################
METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'pruning'
################################################################################
### CREATE SCENARIO FOR FEDCORE AGENT (TRAIN AND OPTIMISE MODEL FROM SCRATCH ###
### or optimise pretrained model with PEFT strategies                        ###
################################################################################
INITIAL_MODEL = 'ResNet18'
PATH_TO_PRETRAIN = 'pretrain_models/pretrain_model_checkpoint_at_15_epoch.pth'
PRETRAIN_SCENARIO = 'from_checkpoint'

dataset = 'CinCECGTorso'
def load_example_dataset():
    PATH_TO_TRAIN = os.path.join(PATH_TO_DATA, 'time_series_classification', 'one_dim', dataset,
                                 f'{dataset}_TRAIN.tsv')
    PATH_TO_TEST = os.path.join(PATH_TO_DATA, 'time_series_classification', 'one_dim', dataset,
                                f'{dataset}_TEST.tsv')

    train_dataloader_params = {"batch_size": 8,
                               'shuffle': True,
                               'data_type': 'time_series',
                               'split_ratio': [0.8, 0.2]}
    test_dataloader_params = {"batch_size": 8,
                              'shuffle': True,
                              'data_type': 'time_series'}
    fedcore_train_data = load_data(source=PATH_TO_TRAIN, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=PATH_TO_TEST, loader_params=test_dataloader_params)
    return fedcore_train_data, fedcore_test_data


#######################################################################
### CREATE SUBCONFIGS MODEL LEARNING, PEFT AND POSTFINETUNE PROCESS ###
#######################################################################
model_config = ModelArchitectureConfigTemplate(depth=dict(layers=2,
                                                blocks_per_layer=[2, 2]))

pretrain_config = TrainingTemplate(epochs=200,
                                            log_each=10,
                                            eval_each=15,
                                            save_each=50,
                                            criterion=LOSS,
                                            model_architecture=model_config)

peft_config = LowRankTemplate(
    epochs=15,
    log_each=3,
    eval_each=3,
    save_each=-1,
    strategy='explained_variance',
    non_adaptive_threshold=0.2,
    rank_prune_each=8
)
##################################################
### CREATE API CONFIG TEAMPLATE FROM SUBCONFIGS###
##################################################
# subconfig for Fedot AutoML agent
fedot_config = FedotConfigTemplate(problem=PROBLEM,
                                   metric=METRIC_TO_OPTIMISE,
                                   pop_size=3,
                                   timeout=1,
                                   initial_assumption=INITIAL_MODEL)
# config for AutoML agent
automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)
# subconfig for Model Learning agent
learning_config = LearningConfigTemplate(criterion=LOSS,
                                         learning_strategy=PRETRAIN_SCENARIO,
                                         peft_strategy=PEFT_PROBLEM,
                                         peft_strategy_params=peft_config)
# api config template
api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)

if __name__ == "__main__":
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_train_data, fedcore_test_data = load_example_dataset()
    fedcore_compressor.fit(fedcore_train_data)
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
