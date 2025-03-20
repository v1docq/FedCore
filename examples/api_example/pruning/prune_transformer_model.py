from fedcore.tools.example_utils import get_scenario_for_api, get_custom_dataloader
from fedcore.api.main import FedCore
from fedcore.api.utils.checkers_collection import ApiConfigCheck
from fedcore.data.dataloader import load_data
from fedcore.repository.config_repository import DEFAULT_TSF_API_CONFIG

dataloader_params = {'path_to_dataset': './custom_dataset/forecasting',
                     "seq_len": {"train": 512, "val": 512, "test": 512},
                     "pred_len": {"train": 60,
                                  "val": 360,
                                  "test": 360},
                     "batch_size": {"train": 16,
                                    "val": 8,
                                    "test": 10},
                     "scaling_factor": {"train": 1.0,
                                        "val": 1.0,
                                        "test": 1.0},
                     "feature_columns": ["LIQ_RATE", "injection_0", "injection_1", "injection_2"],
                     "target_columns": ["LIQ_RATE"],
                     "exog_columns": ["injection_0", "injection_1", "injection_2"]}

METRIC_TO_OPTIMISE = ['smape', 'latency']
initial_assumption = {'path_to_model': './pretrain_model/transformer_500_epoch_pretrain.pt',
                      'model_type': 'TST'}
scenario = 'from_checkpoint'
initial_assumption, learning_strategy = get_scenario_for_api(scenario, initial_assumption)
USER_CONFIG = {'problem': 'ts_forecasting',
               'task_params': {'forecast_length': 60},
               'metric': METRIC_TO_OPTIMISE,
               'initial_assumption': initial_assumption,
               'pop_size': 1,
               'timeout': 5,
               'learning_strategy': learning_strategy,
               'learning_strategy_params': dict(epochs=1,
                                                learning_rate=0.0001,
                                                loss='smape',
                                                train_horizon=60,
                                                test_horizon=360,
                                                model_params={"input_dim": 4,
                                                              'seq_len': 512,
                                                              'model_dim': 512,
                                                              "number_heads": 16,
                                                              "n_layers": 3},
                                                custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                                'maximise_task': False,
                                                                                                'delta': 0.01})
                                                ),
               'peft_strategy': 'pruning',
               'peft_strategy_params': dict(pruning_iterations=1,
                                            importance='MagnitudeImportance',
                                            pruner_name='magnitude_pruner',
                                            importance_norm=1,
                                            pruning_ratio=0.7,
                                            finetune_params={'epochs': 1,
                                                             "learning_rate": 0.0001,
                                                             'loss': 'smape'}
                                            )
               }

if __name__ == "__main__":
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_TSF_API_CONFIG, **USER_CONFIG)
    train_data, test_data = load_data(source=get_custom_dataloader, loader_params=dataloader_params)
    fedcore_compressor = FedCore(**api_config)
    fedcore_compressor.fit(train_data)
    model_comparison = fedcore_compressor.get_report(test_data)
