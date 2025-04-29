import pytest
import torch
from torch.utils.data import DataLoader

from fedcore.architecture.dataset.task_specified.dummy_clf import DummyDatasetCLF
from fedcore.models.backbone.convolutional.resnet import CLF_MODELS, ResNet, ResNetModel


@pytest.mark.parametrize('model_name, in_channels',
                         [(model, channels) for model in CLF_MODELS.keys() for channels in [1, 3]])
def test_resnet(model_name, in_channels):
    layers = int(model_name.split('ResNet')[1])
    model = ResNet(
                   input_dim=in_channels,
                   output_dim=10,
                   depth={
                       'layers': layers,
                       'blocks_per_layer': [2,2]
                       },
                   custom_params={
                       'sizes_per_layer': [64,128],
                       'strides_per_layer': [1,2]
                       }
                       )


    dataset = DummyDatasetCLF(
        num_samples=500,
        channels=in_channels
    )
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for batch in loader:
        inputs, targets = batch
        output_forward = model.forward(inputs)
        assert isinstance(output_forward, torch.Tensor)
        assert output_forward.shape[0] == inputs.shape[0]
        assert output_forward.shape[1] == 10
        break
    assert isinstance(model.model, torch.nn.Module)
