from fedcore.api.main import FedCore
from fedcore.api.utils.evaluation import evaluate_original_model, evaluate_optimised_model

experiment_setup = {'compression_task': 'pruning',
                    'cv_task': 'classification',
                    'model_params': dict(epochs=10,
                                         pruning_iterations=1,
                                         learning_rate=0.001,
                                         importance='MagnitudeImportance',
                                         pruner_name='magnitude_pruner',
                                         importance_norm=1,
                                         pruning_ratio=0.5,
                                         finetune_params={'epochs': 3,
                                                          'custom_loss': None}
                                         )
                    }



if __name__ == "__main__":
    dataset = 'CIFAR10'
    # dataset = 'fedcore/examples/data/faster_rcnn'
    torchvision_dataset = False if dataset == 'CIFAR10' else True
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'ResNet18'})

    fedcore_compressor.fit(input_data)
    pruning_result = evaluate_optimised_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)

    convertation_supplementary_data = {'model_to_export': pruning_result['optimised_model']}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
