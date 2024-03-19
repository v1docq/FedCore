from fedcore.api.main import FedCore
from fedcore.repository.constanst_repository import ONNX_INT8_CONFIG

experiment_setup = {'problem': 'quantisation',
                    'cv_task': 'classification',
                    'framework_config': ONNX_INT8_CONFIG
                    }

if __name__ == "__main__":
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path=None,
                                              supplementary_data={'dataset_name': 'CIFAR10',
                                                                  'model_name': 'ResNet18'})
    fedcore_compressor.fit(input_data)
    quant_model = fedcore_compressor.predict(input_data).predict
    convertation_supplementary_data = {'model_to_export': quant_model}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)

