from fedcore.api.main import FedCore
from fedcore.api.utils.evaluation import evaluate_optimised_model, evaluate_original_model

pruning_params = dict(epochs=15,
                      pruning_iterations=3,
                      learning_rate=0.001,
                      importance='MagnitudeImportance',
                      pruner_name='magnitude_pruner',
                      importance_norm=1,
                      pruning_ratio=0.75,
                      finetune_params={'epochs': 5,
                                       'custom_loss': None}
                      )
low_rank_params = dict(epochs=5,
                       learning_rate=0.001,
                       hoyer_loss=0.2,
                       energy_thresholds=[0.9],
                       orthogonal_loss=5,
                       decomposing_mode='channel',
                       spectrum_pruning_strategy='energy',
                       finetune_params={'epochs': 10,
                                        'custom_loss': None}
                       )

COMPUTE_CONFIG = {'backend': 'cpu',
                  'distributed': None,
                  'output_folder': './cifar_composite_example',
                  'use_cache': None,
                  'automl_folder': {'optimisation_history': './opt_hist',
                                    'composition_results': './comp_res'}}

AUTOML_CONFIG = {'task': 'classification',
                 'initial_assumption': 'resnet18',
                 'use_automl': False,
                 'available_operations': ['pruning_model', 'low_rank_model'],
                 'optimisation_strategy': {'mutation_agent': 'bandit',
                                           'mutation_strategy': 'growth_mutation_strategy',
                                           'optimisation_agent': 'Fedcore'}}

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': {'lr': 0.001,
                                                'optimizer': 'Adam',
                                                'epochs': 100},
                   'peft_strategy': {'strategy_type': ['pruning', 'low_rank'],
                                     'strategy_params': {'pruning': pruning_params,
                                                         'low_rank': low_rank_params}},
                   'optimisation_loss': {'quality_loss': 'CrossEntropy',
                                         'computational_loss': 'Throughput'}}

DEVICE_CONFIG = {'device_type': 'cpu',
                 'inference': 'onnx'}

experiment_setup = {'device_config': DEVICE_CONFIG,
                    'automl_config': AUTOML_CONFIG,
                    'learning_config': LEARNING_CONFIG,
                    'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    dataset = 'CIFAR10'
    # dataset = 'fedcore/data/datasets/low_rank/dataset'
    torchvision_dataset = False if dataset == 'CIFAR10' else True
    fedcore_compressor = FedCore(**experiment_setup)

    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'ResNet50'})

    fedcore_compressor.fit(input_data)
    pruning_result = evaluate_optimised_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)

    convertation_supplementary_data = {'model_to_export': pruning_result['optimised_model']}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
