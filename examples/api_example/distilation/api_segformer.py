from fedcore.api.main import FedCore
from fedcore.api.utils.evaluation import evaluate_optimised_model, evaluate_original_model




experiment_setup = {'compression_task': 'distilation',
                    'cv_task': 'semantic_segmentation',
                    'model_params': dict(epochs=1,
                                         learning_rate=0.001,
                                         hoyer_loss=0.2,
                                         energy_thresholds=[0.9],
                                         orthogonal_loss=5,
                                         decomposing_mode='channel',
                                         spectrum_pruning_strategy='median'
                                         )
                    }

if __name__ == "__main__":
    #dataset = 'CIFAR10'
    dataset = 'fedcore/examples/data/segformer'
    torchvision_dataset = False if dataset == 'CIFAR10' else True
    fedcore_compressor = FedCore(**experiment_setup)
    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'segformer'})
    fedcore_compressor.fit(input_data)
    distil_result = evaluate_optimised_model(fedcore_compressor, input_data)
    original_result = evaluate_original_model(fedcore_compressor, input_data)

    convertation_supplementary_data = {'model_to_export': distil_result['optimised_model']}
    onnx_model = fedcore_compressor.convert_model(supplementary_data=convertation_supplementary_data)
