from fedcore.api.main import FedCore

experiment_setup = {'compression_task': 'pruning',
                    'cv_task': 'object_detection',
                    'model_params': dict(epochs=1,
                                         learning_rate=0.001,
                                         hoyer_loss=0.2,
                                         energy_thresholds=[0.9],
                                         orthogonal_loss=5,
                                         decomposing_mode='channel',
                                         spectrum_pruning_strategy='median'
                                         )
                    }


def evaluate_pruned_model(fedcore_compressor, input_data):
    low_rank_prediction = fedcore_compressor.predict(input_data, output_mode='compress')
    low_rank_output = low_rank_prediction.predict
    low_rank_model = fedcore_compressor.optimized_model
    low_rank_quality_metrics = fedcore_compressor.evaluate_metric(predicton=low_rank_output,
                                                                  target=fedcore_compressor.target)
    low_rank_inference_metrics = fedcore_compressor.evaluate_metric(predicton=low_rank_output,
                                                                    target=fedcore_compressor.target,
                                                                    metric_type='optimize_computational')
    return dict(low_rank_model=low_rank_model,
                quality_metrics=low_rank_quality_metrics,
                inference_metrics=low_rank_inference_metrics)


def evaluate_original_model(fedcore_compressor, input_data):
    original_prediction = fedcore_compressor.predict(input_data, output_mode='default')
    original_output = original_prediction.predict
    original_model = fedcore_compressor.original_model
    original_quality_metrics = fedcore_compressor.evaluate_metric(predicton=original_output,
                                                                  target=fedcore_compressor.target)
    original_inference_metrics = fedcore_compressor.evaluate_metric(predicton=original_output,
                                                                    target=fedcore_compressor.target,
                                                                    metric_type='computational')
    return dict(original_model_model=original_model,
                quality_metrics=original_quality_metrics,
                inference_metrics=original_inference_metrics)


if __name__ == "__main__":
    dataset = 'CIFAR10'
    # dataset = 'fedcore/examples/data/faster_rcnn'
    torchvision_dataset = False if dataset == 'CIFAR10' else True
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'ResNet18'})

    fedcore_compressor.fit(input_data)
    low_rank_result = evaluate_pruned_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)

    convertation_supplementary_data = {'model_to_export': low_rank_result['low_rank_model']}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
