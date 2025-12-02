from enum import Enum

from fedcore.algorithm.distillation.distilator import BaseDistilator
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn

from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.algorithm.quantization.quantizers import BaseQuantizer

# from fedcore.models.backbone.chronos import chronos_small
from fedcore.models.backbone.convolutional.mobilenet import MobileNetV3Small, MobileNetV3Large
from fedcore.models.backbone.convolutional.resnet import *
from fedcore.models.backbone.custom.custom import CustomModel
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

from fedcore.models.backbone.pretrain_model.segformer import segformer_pretrain
from fedcore.models.backbone.transformers.tst import TSTModel
from fedcore.models.backbone.convolutional.inception import InceptionTimeModel

class AtomizedModel(Enum):
    TRAINING_MODELS = {"training_model": BaseNeuralModel}

    PRUNER_MODELS = {"pruning_model": BasePruner}

    LOW_RANK_MODELS = {"low_rank_model": LowRankModel}
    
    LORA_TRAINING_MODELS = {"lora_training_model": BaseLoRA}

    QUANTIZATION_MODELS = {"quantization_model": BaseQuantizer}

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
        "ResNet": ResNetModel,
        "ResNet18": resnet18,
        "ResNet34": resnet34,
        "ResNet50": resnet50,
        "ResNet101": resnet101,
        "ResNet152": resnet152,
    }
    INCEPTIONET_MODELS = {
        "InceptionNet": InceptionTimeModel
    }

    SEGFORMER_MODELS = {"segformer": segformer_pretrain}

    DENSENET_MODELS = {
        "densenet121": densenet121,
        "densenet169": densenet169,
        "densenet201": densenet201,
        "densenet161": densenet161,
    }

    # CHRONOS_MODELS = {'chronos-t5-small': chronos_small}

    TRANSFORMER_MODELS = {'TST': TSTModel}

    PRUNED_RESNET_MODELS = {
        "ResNet18": pruned_resnet18,
        "ResNet34": pruned_resnet34,
        "ResNet50": pruned_resnet50,
        "ResNet101": pruned_resnet101,
        "ResNet152": pruned_resnet152,
    }

    DETECTION_MODELS = {"detection_model": fasterrcnn_mobilenet_v3_large_fpn}

    CUSTOM_MODEL = {"custom": CustomModel}


PRUNER_MODELS = AtomizedModel.PRUNER_MODELS.value
QUANTIZATION_MODELS = AtomizedModel.QUANTIZATION_MODELS.value
DISTILATION_MODELS = AtomizedModel.DISTILATION_MODELS.value
LOW_RANK_MODELS = AtomizedModel.LOW_RANK_MODELS.value
LORA_TRAINING_MODELS = AtomizedModel.LORA_TRAINING_MODELS.value
TRAINING_MODELS = AtomizedModel.TRAINING_MODELS.value

RESNET_MODELS = AtomizedModel.RESNET_MODELS.value
DENSENET_MODELS = AtomizedModel.DENSENET_MODELS.value
EFFICIENTNET_MODELS = AtomizedModel.EFFICIENTNET_MODELS.value
INCEPTIONET_MODELS = AtomizedModel.INCEPTIONET_MODELS.value
MOBILENET_MODELS = AtomizedModel.MOBILENET_MODELS.value
# CHRONOS_MODELS = AtomizedModel.CHRONOS_MODELS.value
SEGFORMER_MODELS = AtomizedModel.SEGFORMER_MODELS.value
TRANSFORMER_MODELS = AtomizedModel.TRANSFORMER_MODELS.value
CUSTOM_MODEL = AtomizedModel.CUSTOM_MODEL.value

BACKBONE_MODELS = {
    **MOBILENET_MODELS,
    **INCEPTIONET_MODELS,
    **EFFICIENTNET_MODELS,
    **DENSENET_MODELS,
    **RESNET_MODELS,
    #    **CHRONOS_MODELS,
    **SEGFORMER_MODELS,
    **TRANSFORMER_MODELS,
    **CUSTOM_MODEL
}
DETECTION_MODELS = AtomizedModel.DETECTION_MODELS.value


def default_fedcore_availiable_operation(
    problem: str = "pruning", 
    exclude_operations: list = None,
    exclude_lora: bool = False,
    include_lora: bool = True
):
    """
    Get available operations for a given problem type.
    
    Args:
        problem: Type of problem ('pruning', 'quantization', 'lora_training', etc.)
        exclude_operations: List of operation names to exclude (optional)
        exclude_lora: If True, exclude LoRA training operations (default: False)
        include_lora: If False, exclude LoRA training operations (default: True)
    
    Returns:
        List of available operation names
    
    Examples:
        # Exclude LoRA from evolutionary optimization
        ops = default_fedcore_availiable_operation('composite_compression', exclude_lora=True)
        
        # Or equivalently
        ops = default_fedcore_availiable_operation('composite_compression', include_lora=False)
    """
    all_operations = [
        "quantization_model",
        "low_rank_model",
        "pruning_model",
    ]
    operation_dict = {
        "pruning": PRUNER_MODELS.keys(),
        "composite_compression": all_operations,
        "quantization": QUANTIZATION_MODELS.keys(),
        "distilation": DISTILATION_MODELS.keys(),
        "low_rank": LOW_RANK_MODELS.keys(),
        "lora_training": LORA_TRAINING_MODELS.keys(),
        "detection": DETECTION_MODELS.keys(),
        "training": TRAINING_MODELS.keys(),
    }

    operations = list(operation_dict[problem])
    
    # Handle LoRA exclusion
    if exclude_lora or not include_lora:
        operations = [op for op in operations if op not in LORA_TRAINING_MODELS.keys()]
    
    # Handle general exclusions
    if exclude_operations:
        operations = [op for op in operations if op not in exclude_operations]
    
    return operations
