import os

from fedcore.architecture.utils.paths import PATH_TO_DATA
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.data.dataloader import load_data

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (
    APIConfigTemplate, DeviceConfigTemplate, AutoMLConfigTemplate,
    LearningConfigTemplate, NeuralModelConfigTemplate, ComputeConfigTemplate, FedotConfigTemplate,
    PruningTemplate, ModelArchitectureConfigTemplate)

METRIC_TO_OPTIMISE = ['rmse', 'latency']
INITIAL_MODEL = 'InceptionNet'
SCENARIO = 'from_scratch'
LOSS = 'rmse'
PATH_TO_TRAIN = os.path.join(PATH_TO_DATA, 'time_series_regression', 'multi_dim', 'AppliancesEnergy',
                             'AppliancesEnergy_TRAIN.ts')
PATH_TO_TEST = os.path.join(PATH_TO_DATA, 'time_series_regression', 'multi_dim', 'AppliancesEnergy',
                            'AppliancesEnergy_TEST.ts')

train_dataloader_params = {"batch_size": 8,
                           'shuffle': True,
                           'data_type': 'time_series',
                           'split_ratio': [0.8, 0.2]}
test_dataloader_params = {"batch_size": 8,
                          'shuffle': True,
                          'data_type': 'time_series'}

initial_assumption, learning_strategy = get_scenario_for_api(SCENARIO, INITIAL_MODEL)

fedot_config = FedotConfigTemplate(problem='regression',
                                   metric=METRIC_TO_OPTIMISE,
                                   pop_size=1,
                                   timeout=5,
                                   initial_assumption=initial_assumption)

model_config = ModelArchitectureConfigTemplate(input_dim=None,
                                               output_dim=None,
                                               depth=6)

pretrain_config = NeuralModelConfigTemplate(epochs=200,
                                            log_each=10,
                                            eval_each=15,
                                            save_each=50,
                                            criterion=LOSS,
                                            model_architecture=model_config,
                                            custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                            'maximise_task': False,
                                                                                            'delta': 0.01}))
finetune_config = NeuralModelConfigTemplate(custom_learning_params={'epochs': 10,
                                                                    "learning_rate": 0.0001},
                                            custom_criterions={'loss': LOSS})
peft_config = PruningTemplate(importance="MagnitudeImportance",
                              pruning_ratio=0.5,
                              finetune_params=finetune_config
                              )

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)
learning_config = LearningConfigTemplate(criterion=LOSS,
                                         learning_strategy=learning_strategy,
                                         learning_strategy_params=pretrain_config,
                                         peft_strategy='pruning',
                                         peft_strategy_params=peft_config)
api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)

if __name__ == "__main__":
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_train_data = load_data(source=PATH_TO_TRAIN, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=PATH_TO_TEST, loader_params=test_dataloader_params)
    fedcore_compressor.fit(fedcore_train_data)
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
