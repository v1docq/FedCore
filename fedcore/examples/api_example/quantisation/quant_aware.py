import numpy as np
import torch.nn
import torchvision.datasets
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.utils.paths import data_path
from fedcore.architecture.visualisation.visualization import plot_train_test_loss_metric
from fedcore.data.data import CompressionInputData
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.neural_compressor import QuantizationAwareTrainingConfig
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.neural_compressor.training import prepare_compression
from fedcore.repository.constanst_repository import FEDOT_TASK
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.repository.model_repository import RESNET_MODELS
from fedcore.tools.ruler import PerformanceEvaluator

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    test_dataset = torchvision.datasets.CIFAR10(data_path('CIFAR10'), train=False, download=True,
                                                transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    repo = FedcoreModels().setup_repository()
    compression_pipeline = PipelineBuilder().add_node('training_aware_quant', params={'epochs': 8}).build()

    input_data = CompressionInputData(
                                      features=np.zeros((2, 2)),
                                      idx=None,
                                      train_dataloader=train_dataloader,
                                      task=FEDOT_TASK['classification'],
                                      data_type=None,
                                      target=model
                                      )
    input_data.supplementary_data.is_auto_preprocessed = True
    compression_pipeline.fit(input_data)
    quant_model = compression_pipeline.predict(input_data).predict
    quant_model.save('./output')
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
    evaluator = PerformanceEvaluator(onnx_model, test_dataset, batch_size=64)
    performance = evaluator.eval()
    print('after quantization')
    print(performance)

