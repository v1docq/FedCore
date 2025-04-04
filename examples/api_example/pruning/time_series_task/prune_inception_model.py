import os

from fedcore.architecture.utils.paths import PATH_TO_DATA
from fedcore.tools.example_utils import get_scenario_for_api, get_custom_dataloader
from fedcore.api.main import FedCore
from fedcore.api.utils.checkers_collection import ApiConfigCheck
from fedcore.data.dataloader import load_data
from fedcore.repository.config_repository import DEFAULT_REG_API_CONFIG

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

USER_CONFIG = {'problem': 'regression',
               'metric': METRIC_TO_OPTIMISE,
               'initial_assumption': initial_assumption,
               'pop_size': 1,
               'timeout': 5,
               'learning_strategy': learning_strategy,
               'learning_strategy_params': dict(epochs=200,
                                                learning_rate=0.0001,
                                                loss=LOSS,
                                                custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                                'maximise_task': False,
                                                                                                'delta': 0.01})
                                                ),
               'peft_strategy': 'pruning',
               'peft_strategy_params': dict(pruning_iterations=1,
                                            importance="MagnitudeImportance",
                                            pruner_name='meta_pruner',
                                            importance_norm=2,
                                            pruning_ratio=0.5,
                                            finetune_params={'epochs': 10,
                                                             "learning_rate": 0.0001,
                                                             'loss': LOSS}
                                            )
               }

if __name__ == "__main__":
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_REG_API_CONFIG, **USER_CONFIG)
    fedcore_train_data = load_data(source=PATH_TO_TRAIN, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=PATH_TO_TEST, loader_params=test_dataloader_params)
    fedcore_compressor = FedCore(api_config)
    fedcore_compressor.fit(fedcore_train_data)
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
