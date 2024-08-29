from fedcore.api.main import FedCore

experiment_setup = {'compression_task': 'low_rank',
                    'cv_task': 'classification',
                    'model_params': dict(epochs=3,
                                         learning_rate=0.001,
                                         hoyer_loss=0.2,
                                         energy_thresholds=[0.9],
                                         orthogonal_loss=5,
                                         decomposing_mode='channel',
                                         spectrum_pruning_strategy='median'
                                         )
                    }

if __name__ == "__main__":
    dataset = 'CIFAR10'
    #dataset = 'fedcore/data/datasets/low_rank/dataset'
    torchvision_dataset = False if dataset == 'CIFAR10' else True
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'ResNet18'})
    fedcore_compressor.fit(input_data)

    low_rank_prediction = fedcore_compressor.predict(input_data, output_mode='compress')
    low_rank_labels = low_rank_prediction.predict.predict
    low_rank_model = fedcore_compressor.solver.root_node.fitted_operation.optimized_model
    low_rank_metrics = fedcore_compressor.get_metrics(labels=low_rank_labels,
                                                      target=fedcore_compressor.target)

    original_prediction = fedcore_compressor.predict(input_data, output_mode='default')
    original_labels = low_rank_prediction.predict.predict
    original_model = fedcore_compressor.solver.root_node.fitted_operation.optimized_model
    original_metrics = fedcore_compressor.get_metrics(labels=original_labels,
                                                      target=fedcore_compressor.target)

    convertation_supplementary_data = {'model_to_export': low_rank_model}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
