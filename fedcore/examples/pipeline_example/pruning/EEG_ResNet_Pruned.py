import torch
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedcore.repository.constanst_repository import FEDOT_TASK
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.repository.model_repository import RESNET_MODELS_ONE_CHANNEL, RESNET_MODELS

if __name__ == "__main__":
    repo = FedcoreModels().setup_repository()
    nn_model = RESNET_MODELS['ResNet18'](pretrained=True).eval()
    example_inputs = torch.randn(1, 3, 224, 224)
    pruner_model = PipelineBuilder().add_node('pruner_model', params={'channels_to_prune': [2, 6, 9],
                                                                      'epochs': 50})


    input_data = InputData(features=example_inputs,
                           idx=None,
                           task=FEDOT_TASK['classification'],
                           data_type=None,
                           target=nn_model
                           )
    input_data.supplementary_data.is_auto_preprocessed = True
    pruner_model.fit(input_data)
    pruned_model = pruner_model.predict(input_data).predict
    _ = 1
