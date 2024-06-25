from fedcore.api.main import FedCore

experiment_setup = {'problem': 'pruning',
                    'cv_task': 'object_detection'}

if __name__ == "__main__":
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path='fedcore/examples/data/faster_rcnn')
    fedcore_compressor.fit(input_data)
    pruned_model = fedcore_compressor.predict(input_data).predict

