from fedcore.api.main import FedCore

experiment_setup = {'problem': 'pruning',
                    'cv_task': 'segmentation'}

if __name__ == "__main__":
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path='fedcore/examples/data/unet')
    fedcore_compressor.fit(input_data)
    convertation_supplementary_data = {'model_to_export': fedcore_compressor.solver}
    pruned_model_prediction = fedcore_compressor.predict(input_data, output_mode='compress')
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)

