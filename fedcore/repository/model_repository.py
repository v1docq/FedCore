from enum import Enum


from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.algorithm.quantization.quant_aware_training import QuantAwareModel
from fedcore.algorithm.quantization.quant_post_training import QuantPostModel
from fedcore.models.backbone.resnet import *


class AtomizedModel(Enum):
    PRUNER_MODELS = {'pruning_model': BasePruner}

    QUANTISATION_MODELS = {'post_training_quant': QuantPostModel,
                           'training_aware_quant': QuantAwareModel}

    CLF_MODELS = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
        'ResNet152': resnet152,
    }

    PRUNED_MODELS = {
        "ResNet18": pruned_resnet18,
        "ResNet34": pruned_resnet34,
        "ResNet50": pruned_resnet50,
        "ResNet101": pruned_resnet101,
        "ResNet152": pruned_resnet152,
    }


PRUNER_MODELS = AtomizedModel.PRUNER_MODELS.value
QUANTISATION_MODELS = AtomizedModel.QUANTISATION_MODELS.value
RESNET_MODELS = AtomizedModel.CLF_MODELS.value


def default_fedcore_availiable_operation(problem: str = 'pruning'):
    operation_dict = {'pruning': PRUNER_MODELS.keys(),
                      'quantisation': QUANTISATION_MODELS.keys()}

    return operation_dict[problem]
