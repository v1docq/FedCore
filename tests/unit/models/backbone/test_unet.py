import pytest
import torch
from torch.utils.data import DataLoader

from fedcore.architecture.dataset.dummy_clf import DummyDatasetCLF
from fedcore.models.backbone.unet import UNetwork


@pytest.mark.parametrize('in_channels', [1, 3])
def test_unet(in_channels):
    model = UNetwork(input_dim=in_channels,
                     output_dim=10)

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
