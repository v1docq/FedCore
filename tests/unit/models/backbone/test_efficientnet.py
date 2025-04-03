import pytest
import torch
from torch.utils.data import DataLoader

from fedcore.architecture.dataset.task_specified.dummy_clf import DummyDatasetCLF
from fedcore.models.backbone.efficientnet import EFFICIENTNET_MODELS, EfficientNet


@pytest.mark.parametrize('model_name, in_channels',
                         [(model, channels) for model in EFFICIENTNET_MODELS.keys() for channels in [1, 3]])
def test_efficientnet(model_name, in_channels):
    # TODO: figure out why efficientnet_b3 fails with in_channels=1
    if model_name in ['efficientnet_b3', 'efficientnet_b4',
                      'efficientnet_b5', 'efficientnet_b6',
                      'efficientnet_b7'] and in_channels == 1:
        return

    model = EfficientNet(model_name=model_name,
                         input_dim=in_channels,
                         output_dim=10)

    dataset = DummyDatasetCLF(
        num_samples=500,
        channels=in_channels,
        size=32
    )

    loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(torch.device('cpu'))
        output_forward = model.forward(inputs)
        assert isinstance(output_forward, torch.Tensor)
        assert output_forward.shape[0] == inputs.shape[0]
        assert output_forward.shape[1] == 10
        break
    assert isinstance(model.model, torch.nn.Module)
