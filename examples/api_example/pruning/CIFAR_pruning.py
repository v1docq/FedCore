from fedcore.api.main import FedCore
from fedcore.api.utils.checkers_collection import ApiConfigCheck
from fedcore.data.dataloader import load_data
from fedcore.api.utils.evaluation import evaluate_original_model, evaluate_optimised_model
from fedcore.repository.config_repository import DEFAULT_CLF_API_CONFIG

DATASET = 'CIFAR10'
DATASET_PARAMS = {'train_bs': 64,
                  'val_bs': 100,
                  'train_shuffle': True,
                  'val_shuffle': False}
METRIC_TO_OPTIMISE = ['accuracy', 'latency', 'throughput']

USER_CONFIG = {'task': 'classification',
               'metric': METRIC_TO_OPTIMISE,
               'initial_assumption': 'ResNet18',
               'timeout': 200,
               'learning_strategy': 'from_scratch',
               'learning_strategy_params': dict(epochs=5,
                                                learning_rate=0.0001,
                                                loss='crossentropy',
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
                                                             'custom_loss': None}
                                            )
               }

if __name__ == "__main__":
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG, **USER_CONFIG)
    input_data = load_data(DATASET)
    fedcore_compressor = FedCore(**api_config)
    fedcore_compressor.fit(input_data)
    pruning_result = evaluate_optimised_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)
    onnx_model = fedcore_compressor.export()
