import numpy as np
import torch.nn
import torchvision.datasets
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.utils.paths import data_path
from fedcore.data.data import CompressionInputData
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.repository.constanst_repository import FEDOT_TASK
from fedcore.repository.initializer_industrial_models import FedcoreModels

from fedcore.tools.ruler import PerformanceEvaluator

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=1)

    model = resnet18(pretrained=True).to(default_device())
    model.fc = nn.Linear(512, 10).to(default_device())

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    # Train the model
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(default_device()))
            loss = criterion(outputs, labels.to(default_device()))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    evaluator = PerformanceEvaluator(model, test_dataset, batch_size=64)
    performance = evaluator.eval()
    print('Before quantization')
    print(performance)

    model = model.cpu()
    repo = FedcoreModels().setup_repository()
    compression_pipeline = PipelineBuilder().add_node('post_training_quant').build()

    input_data = CompressionInputData(features=np.zeros((2, 2)),
                                      idx=None,
                                      calib_dataloader=val_dataloader,
                                      task=FEDOT_TASK['classification'],
                                      data_type=None,
                                      target=model
                                      )
    input_data.supplementary_data.is_auto_preprocessed = True
    compression_pipeline.fit(input_data)
    quant_model = compression_pipeline.predict(input_data).predict

    int8_onnx_config = Torch2ONNXConfig(
        dtype="int8",
        opset_version=16,
        quant_format="QDQ",  # or "QLinear"
        example_inputs=torch.unsqueeze(train_dataset[0][0], dim=0),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': [0], 'output': [0]}
    )

    quant_model.export("int8-model.onnx", int8_onnx_config)
    onnx_model = ONNXInferenceModel("int8-model.onnx")
    evaluator = PerformanceEvaluator(onnx_model,  test_dataset, batch_size=64)
    performance = evaluator.eval()
    print('after quantization')
    print(performance)
