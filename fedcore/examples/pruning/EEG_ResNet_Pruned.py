import torch
from fedot.core.data.data import InputData
from torchvision.models import resnet18

from fedcore.algorithm.pruning.pruners import BasePruner

if __name__ == "__main__":
    model = resnet18(pretrained=True).eval()
    example_inputs = torch.randn(1, 3, 224, 224)
    pruner_model = BasePruner()
    input_data = InputData(features=example_inputs,
                           idx=None,
                           task=None,
                           data_type=None,
                           supplementary_data={'model': model,
                                               'channels_to_prune': [2, 6, 9]}
                           )
    pruner_model.fit(input_data)
    pruned_model = pruner_model.predict(input_data)
