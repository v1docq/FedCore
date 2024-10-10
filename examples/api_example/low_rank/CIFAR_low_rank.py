from fedcore.api.main import FedCore
from fedcore.api.utils.evaluation import evaluate_optimised_model, evaluate_original_model

experiment_setup = {'compression_task': 'low_rank',
                    'cv_task': 'classification',
                    'model_params': dict(epochs=30,
                                         learning_rate=0.001,
                                         hoyer_loss=0.2,
                                         energy_thresholds=[0.5],
                                         orthogonal_loss=5,
                                         decomposing_mode='channel',
                                         spectrum_pruning_strategy='energy',
                                         finetune_params={'epochs': 5,
                                                          'custom_loss': None}
                                         )
                    }

if __name__ == "__main__":
    dataset = 'CIFAR10'
    # dataset = 'fedcore/data/datasets/low_rank/dataset'
    torchvision_dataset = False if dataset == 'CIFAR10' else True
    fedcore_compressor = FedCore(**experiment_setup)

    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'ResNet18'})

    fedcore_compressor.fit(input_data)
    low_rank_result = evaluate_optimised_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)


    convertation_supplementary_data = {'model_to_export': low_rank_result['optimised_model']}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
