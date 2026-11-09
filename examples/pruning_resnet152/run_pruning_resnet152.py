import sys
import os
import torch

from pathlib import Path 

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fedcore.api.presets import CompressionPreset, get_preset
from fedcore.data.dataloader import load_data
from fedcore.api.main import FedCore
from torchvision import models

pretrained_resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
pretrained_resnet152.fc = torch.nn.Linear(2048, 10) 
INITIAL_ASSUMPTION = pretrained_resnet152 

train_dataloader_params = {"batch_size": 64,
                           'shuffle': True,
                           'is_train': True,
                           'data_type': 'table',
                           'split_ratio': [0.8, 0.2]}
test_dataloader_params = {"batch_size": 100,
                          'shuffle': True,
                          'is_train': False,
                          'data_type': 'table'}

def load_benchmark_dataset(dataset_name, train_dataloader_params, test_dataloader_params):
    fedcore_train_data = load_data(source=dataset_name, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=dataset_name, loader_params=test_dataloader_params)
    return fedcore_train_data, fedcore_test_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Check training, pruning and reporting on a small CIFAR-10 subset.')
    args = parser.parse_args()
    if args.smoke:
        train_dataloader_params.update(subset=0.002, batch_size=16, num_workers=0)
        test_dataloader_params.update(subset=0.0064, batch_size=16, num_workers=0)

    api_config = get_preset(CompressionPreset.PRUNING,
                            initial_assumption=INITIAL_ASSUMPTION,
                            smoke=args.smoke)
    fedcore_compressor = FedCore(api_config)
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset('CIFAR10', train_dataloader_params,
                                                                   test_dataloader_params)
    fedcore_compressor.fit(fedcore_train_data)
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    print(model_comparison)
    save_path = REPO_ROOT / 'results' / ('pruning_resnet152_smoke' if args.smoke else 'pruning_resnet152')
    save_path.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(save_path / 'metrics.csv')
    _ = 1
