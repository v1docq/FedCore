from fedcore.api.main import FedCore
from fedcore.api.utils.evaluation import evaluate_optimised_model, evaluate_original_model

experiment_setup = {'compression_task': 'composite_compression',
                    'cv_task': 'classification',
                    'model_params': dict(pruning_model=dict(epochs=15,
                                                            pruning_iterations=3,
                                                            learning_rate=0.001,
                                                            importance='MagnitudeImportance',
                                                            pruner_name='magnitude_pruner',
                                                            importance_norm=1,
                                                            pruning_ratio=0.75,
                                                            finetune_params={'epochs': 5,
                                                                             'custom_loss': None}
                                                            ),
                                         low_rank_model=dict(epochs=5,
                                                             learning_rate=0.001,
                                                             hoyer_loss=0.2,
                                                             energy_thresholds=[0.9],
                                                             orthogonal_loss=5,
                                                             decomposing_mode='channel',
                                                             spectrum_pruning_strategy='energy',
                                                             finetune_params={'epochs': 10,
                                                                              'custom_loss': None}
                                                             )
                                         ),
                    'initial_assumption': ['pruning_model', 'low_rank_model']}

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