from enum import Enum

from fedcore.algorithm.distillation.distilator import BaseDistilator
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn

from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.algorithm.quantization.quant_aware_training import QuantAwareModel
from fedcore.algorithm.quantization.quant_post_training import QuantPostModel

# from fedcore.models.backbone.chronos import chronos_small
from fedcore.models.backbone.mobilenet import MobileNetV3Small, MobileNetV3Large
from fedcore.models.backbone.resnet import *
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)
from torchvision.models.densenet import (
    densenet121,
    densenet161,
    densenet169,
    densenet201,
)

from fedcore.models.backbone.segformer import segformer_pretrain


class AtomizedModel(Enum):
    TRAINING_MODELS = {"training_model": BaseNeuralModel}

    PRUNER_MODELS = {"pruning_model": BasePruner}

    LOW_RANK_MODELS = {"low_rank_model": LowRankModel}

    QUANTISATION_MODELS = {
        "post_training_quant": QuantPostModel,
        "training_aware_quant": QuantAwareModel,
    }

    DISTILATION_MODELS = {"distilation_model": BaseDistilator}

    MOBILENET_MODELS = {
        "mobilenetv3small": MobileNetV3Small,
        "mobilenetv3large": MobileNetV3Large,
    }

    EFFICIENTNET_MODELS = {
        "efficientnet_b0": efficientnet_b0,
        "efficientnet_b1": efficientnet_b1,
        "efficientnet_b2": efficientnet_b2,
        "efficientnet_b3": efficientnet_b3,
        "efficientnet_b4": efficientnet_b4,
        "efficientnet_b5": efficientnet_b5,
        "efficientnet_b6": efficientnet_b6,
        "efficientnet_b7": efficientnet_b7,
    }
    RESNET_MODELS = {
        "ResNet18": resnet18,
        "ResNet34": resnet34,
        "ResNet50": resnet50,
        "ResNet101": resnet101,
        "ResNet152": resnet152,
    }

    SEGFORMER_MODELS = {"segformer": segformer_pretrain}

    DENSENET_MODELS = {
        "densenet121": densenet121,
        "densenet169": densenet169,
        "densenet201": densenet201,
        "densenet161": densenet161,
    }

    # CHRONOS_MODELS = {'chronos-t5-small': chronos_small}

    PRUNED_RESNET_MODELS = {
        "ResNet18": pruned_resnet18,
        "ResNet34": pruned_resnet34,
        "ResNet50": pruned_resnet50,
        "ResNet101": pruned_resnet101,
        "ResNet152": pruned_resnet152,
    }

    DETECTION_MODELS = {"detection_model": fasterrcnn_mobilenet_v3_large_fpn}


PRUNER_MODELS = AtomizedModel.PRUNER_MODELS.value
QUANTISATION_MODELS = AtomizedModel.QUANTISATION_MODELS.value
DISTILATION_MODELS = AtomizedModel.DISTILATION_MODELS.value
LOW_RANK_MODELS = AtomizedModel.LOW_RANK_MODELS.value
TRAINING_MODELS = AtomizedModel.TRAINING_MODELS.value

RESNET_MODELS = AtomizedModel.RESNET_MODELS.value
DENSENET_MODELS = AtomizedModel.DENSENET_MODELS.value
EFFICIENTNET_MODELS = AtomizedModel.EFFICIENTNET_MODELS.value
MOBILENET_MODELS = AtomizedModel.MOBILENET_MODELS.value
# CHRONOS_MODELS = AtomizedModel.CHRONOS_MODELS.value
SEGFORMER_MODELS = AtomizedModel.SEGFORMER_MODELS.value

BACKBONE_MODELS = {
    **MOBILENET_MODELS,
    **EFFICIENTNET_MODELS,
    **DENSENET_MODELS,
    **RESNET_MODELS,
    #    **CHRONOS_MODELS,
    **SEGFORMER_MODELS,
}
DETECTION_MODELS = AtomizedModel.DETECTION_MODELS.value


def default_fedcore_availiable_operation(problem: str = "pruning"):
    all_operations = [
        "training_aware_quant",
        "post_training_quant",
        "low_rank_model",
        "pruning_model",
    ]
    operation_dict = {
        "pruning": PRUNER_MODELS.keys(),
        "composite_compression": all_operations,
        "quantisation_aware": "training_aware_quant",
        "post_quantisation": "post_training_quant",
        "distilation": DISTILATION_MODELS.keys(),
        "low_rank": LOW_RANK_MODELS.keys(),
        "detection": DETECTION_MODELS.keys(),
        "training": TRAINING_MODELS.keys(),
    }

    return operation_dict[problem]
