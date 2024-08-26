from fedcore.api.main import FedCore

experiment_setup = {'compression_task': 'low_rank',
                    'cv_task': 'classification'
                    }

if __name__ == "__main__":
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path='fedcore/examples/data/low_rank/dataset',
                                              supplementary_data={'torchvision_dataset': True,
                                                                  'torch_model': 'ResNet18'})
    fedcore_compressor.fit(input_data)
    low_rank_model = fedcore_compressor.predict(input_data).predict
    convertation_supplementary_data = {'model_to_export': low_rank_model}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)

