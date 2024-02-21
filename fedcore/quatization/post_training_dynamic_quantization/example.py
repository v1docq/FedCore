import numpy as np
import torch.nn
import torchvision.datasets
from neural_compressor.compression.pruner.utils import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from fedcore.architecture.utils.paths import data_path
from fedcore.architecture.visualisation.visualization import plot_train_test_loss_metric

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model = resnet18()
    torch.save(model.state_dict(), './base_model')

    val_dataset = torchvision.datasets.CIFAR10(data_path('CIFAR10'), train=False, download=True,
                                                transform=transform)
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.1, 0.9])

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1
    )


    # Quantization code
    from neural_compressor import quantization
    from neural_compressor.config import PostTrainingQuantConfig

    conf = (
        PostTrainingQuantConfig()
    )  # default approach is "auto", you can set "dynamic":PostTrainingQuantConfig(approach="dynamic")
    q_model = quantization.fit(
        model=model,
        conf=conf,
        calib_dataloader=val_dataloader
    )
    q_model.save("./output")
