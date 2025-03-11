from fedcore.tools.example_utils import get_scenario_for_api
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
initial_assumption = {'path_to_model': './pretrain_model/resnet_1_epoch_pretrain.pt',
                      'model_type': 'ResNet18'}
initial_assumption, learning_strategy = get_scenario_for_api('checkpoint',initial_assumption)
USER_CONFIG = {'problem': 'classification',
               'metric': METRIC_TO_OPTIMISE,
               'initial_assumption': initial_assumption,
               'pop_size': 1,
               'timeout': 5,
               'learning_strategy': learning_strategy,
               'learning_strategy_params': dict(epochs=1,
                                                learning_rate=0.0001,
                                                loss='crossentropy',
                                                # custom_loss = ['norm_loss', 'weight_loss'] needs to rework BaseNN class
                                                # custom_loss_params = {'norm_loss':{...},
                                                # 'weight_loss':{...}}
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
                                                             'loss': 'crossentropy'}
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
