from fedcore.api.main import FedCore


def get_pruning_config():
    experiment_setup = {'compression_task': 'pruning',
                        'cv_task': 'classification',
                        'metric': ['accuracy', 'latency', 'throughput'],
                        'timeout': 1,
                        'model_params': dict(epochs=1,
                                             pruning_iterations=1,
                                             learning_rate=0.001,
                                             importance='TaylorImportance',
                                             pruner_name='meta_pruner',
                                             importance_norm=1,
                                             pruning_ratio=0.7,
                                             finetune_params={'epochs': 1,
                                                              'custom_loss': None}
                                             )
                        }
    return experiment_setup


def get_dataset():
    dataset = 'CIFAR10'
    torchvision_dataset = False
    fedcore_compressor = FedCore(**get_pruning_config())
    input_data = fedcore_compressor.load_data(path=dataset,
                                              supplementary_data={'torchvision_dataset': torchvision_dataset,
                                                                  'torch_model': 'ResNet18'})
    return fedcore_compressor, input_data


def taylor_pruning():
    fedcore_api, input_data = get_dataset()
    fedcore_api.fit(input_data=input_data, manually_done=True)

if __name__ == '__main__':
    taylor_pruning()
    _ = 1