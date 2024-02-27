import torch
from fedcore.api.main import FedCore
from fedcore.repository.model_repository import RESNET_MODELS

if __name__ == "__main__":
    input_data = (RESNET_MODELS['ResNet18'](pretrained=True).eval(), torch.randn(1, 3, 224, 224))
    fedcore_compressor = FedCore()
    fedcore_compressor.fit(input_data)
    pruned_model = fedcore_compressor.predict(input_data).predict
    _ = 1
