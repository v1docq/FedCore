import numpy as np
import torch.nn
import torchvision.datasets


from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms


from fedcore.architecture.utils.paths import data_path

from fedcore.neural_compressor import PostTrainingQuantConfig
from fedcore.neural_compressor import quantization

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

    conf = (
        PostTrainingQuantConfig()
    )  # default approach is "auto", you can set "dynamic":PostTrainingQuantConfig(approach="dynamic")
    q_model = quantization.fit(
        model=model,
        conf=conf,
        calib_dataloader=val_dataloader
    )
    q_model.save("./output")
