from fedcore.api.main import FedCore

experiment_setup = {'compression_task': 'distilation',
                    'cv_task': 'segmentation'}

if __name__ == "__main__":
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path='fedcore/examples/data/mobilenet')
    fedcore_compressor.fit(input_data)
    distil_model = fedcore_compressor.predict(input_data).predict
    _ = 1
