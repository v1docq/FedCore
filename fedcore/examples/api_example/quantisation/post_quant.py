import numpy as np
import torch.nn
import torchvision.datasets
from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.utils.paths import data_path
from fedcore.data.data import CompressionInputData
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.repository.constanst_repository import FEDOT_TASK
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.repository.model_repository import RESNET_MODELS

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model = resnet18(pretrained=True).to(default_device())
    model.fc = nn.Linear(512, 10).to(default_device())

    train_dataset = torchvision.datasets.CIFAR10(data_path('CIFAR10'), train=True, download=True,
                                                 transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    val_dataset = torchvision.datasets.CIFAR10(data_path('CIFAR10'), train=False, download=True,
                                               transform=transform)
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.1, 0.9])

    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)

    repo = FedcoreModels().setup_repository()
    nn_model = RESNET_MODELS['ResNet18'](pretrained=True).eval()

    compression_pipeline = PipelineBuilder().add_node('post_training_quant').build()

    input_data = CompressionInputData(features=np.zeros((2, 2)),
                                      idx=None,
                                      calib_dataloader=val_dataloader,
                                      task=FEDOT_TASK['classification'],
                                      data_type=None,
                                      target=nn_model
                                      )
    input_data.supplementary_data.is_auto_preprocessed = True
    compression_pipeline.fit(input_data)
    quant_model = compression_pipeline.predict(input_data).predict
    quant_model.save('./output')


