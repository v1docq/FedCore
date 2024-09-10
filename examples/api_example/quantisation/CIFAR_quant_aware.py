from fedcore.api.utils.evaluation import evaluate_original_model
from fedcore.api.main import FedCore
from fedcore.api.utils.evaluation import evaluate_optimised_model
from fedcore.repository.constanst_repository import ONNX_INT8_CONFIG

experiment_setup = {'compression_task': 'quantisation_aware',
                    'cv_task': 'classification',
                    'framework_config': ONNX_INT8_CONFIG,
                    'model_params': dict(epochs=10,
                                         learning_rate=0.001,
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
    quantised_result = evaluate_optimised_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)

    convertation_supplementary_data = {'model_to_export': quantised_result['optimised_model']}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
