from enum import Enum

from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.algorithm.quantization.quant_aware_training import QuantAwareModel
from fedcore.algorithm.quantization.quant_post_training import QuantPostModel
from fedcore.models.backbone.mobilenet import MobileNetV3Small, MobileNetV3Large
from fedcore.models.backbone.resnet import *
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, \
    efficientnet_b3, \
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201


class AtomizedModel(Enum):
    PRUNER_MODELS = {'pruning_model': BasePruner}

    QUANTISATION_MODELS = {'post_training_quant': QuantPostModel,
                           'training_aware_quant': QuantAwareModel}

    MOBILENET_MODELS = {
        'mobilenetv3small': MobileNetV3Small,
        'mobilenetv3large': MobileNetV3Large,
    }

    EFFICIENTNET_MODELS = {
        'efficientnet_b0': efficientnet_b0,
        'efficientnet_b1': efficientnet_b1,
        'efficientnet_b2': efficientnet_b2,
        'efficientnet_b3': efficientnet_b3,
        'efficientnet_b4': efficientnet_b4,
        'efficientnet_b5': efficientnet_b5,
        'efficientnet_b6': efficientnet_b6,
        'efficientnet_b7': efficientnet_b7,
    }
    RESNET_MODELS = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
        'ResNet152': resnet152,
    }

    DENSENET_MODELS = {
        'densenet121': densenet121,
        'densenet169': densenet169,
        'densenet201': densenet201,
        'densenet161': densenet161,
    }

    PRUNED_RESNET_MODELS = {
        "ResNet18": pruned_resnet18,
        "ResNet34": pruned_resnet34,
        "ResNet50": pruned_resnet50,
        "ResNet101": pruned_resnet101,
        "ResNet152": pruned_resnet152,
    }


PRUNER_MODELS = AtomizedModel.PRUNER_MODELS.value
QUANTISATION_MODELS = AtomizedModel.QUANTISATION_MODELS.value
RESNET_MODELS = AtomizedModel.RESNET_MODELS.value
DENSENET_MODELS = AtomizedModel.DENSENET_MODELS.value
EFFICIENTNET_MODELS = AtomizedModel.EFFICIENTNET_MODELS.value
MOBILENET_MODELS = AtomizedModel.MOBILENET_MODELS.value
BACKBONE_MODELS = {**MOBILENET_MODELS, **EFFICIENTNET_MODELS, **DENSENET_MODELS, **RESNET_MODELS}


def default_fedcore_availiable_operation(problem: str = 'pruning'):
    operation_dict = {'pruning': PRUNER_MODELS.keys(),
                      'quantisation': QUANTISATION_MODELS.keys()}

    return operation_dict[problem]
