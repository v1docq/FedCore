from enum import Enum
from fedcore.repository.lazy_registry import LazyFactory, LazyRegistry


class AtomizedModel(Enum):
    TRAINING_MODELS = LazyRegistry({"training_model": LazyFactory("fedcore.models.network_impl.base_nn_model", "BaseNeuralModel")})

    PRUNER_MODELS = LazyRegistry({"pruning_model": LazyFactory("fedcore.algorithm.pruning.pruners", "BasePruner")})

    LOW_RANK_MODELS = LazyRegistry({"low_rank_model": LazyFactory("fedcore.algorithm.low_rank.low_rank_opt", "LowRankModel")})

    LORA_MODELS = LazyRegistry({"lora_model": LazyFactory("fedcore.algorithm.low_rank.lora_operation", "BaseLoRA")})

    QUANTIZATION_MODELS = LazyRegistry({"quantization_model": LazyFactory("fedcore.algorithm.quantization.quantizers", "BaseQuantizer")})

    DISTILATION_MODELS = LazyRegistry({"distilation_model": LazyFactory("fedcore.algorithm.distillation.distilator", "BaseDistilator")})

    MOBILENET_MODELS = LazyRegistry({
        "mobilenetv3small": LazyFactory("fedcore.models.backbone.convolutional.mobilenet", "MobileNetV3Small"),
        "mobilenetv3large": LazyFactory("fedcore.models.backbone.convolutional.mobilenet", "MobileNetV3Large"),
    })

    EFFICIENTNET_MODELS = LazyRegistry({
        "efficientnet_b0": LazyFactory("torchvision.models.efficientnet", "efficientnet_b0"),
        "efficientnet_b1": LazyFactory("torchvision.models.efficientnet", "efficientnet_b1"),
        "efficientnet_b2": LazyFactory("torchvision.models.efficientnet", "efficientnet_b2"),
        "efficientnet_b3": LazyFactory("torchvision.models.efficientnet", "efficientnet_b3"),
        "efficientnet_b4": LazyFactory("torchvision.models.efficientnet", "efficientnet_b4"),
        "efficientnet_b5": LazyFactory("torchvision.models.efficientnet", "efficientnet_b5"),
        "efficientnet_b6": LazyFactory("torchvision.models.efficientnet", "efficientnet_b6"),
        "efficientnet_b7": LazyFactory("torchvision.models.efficientnet", "efficientnet_b7"),
    })
    RESNET_MODELS = LazyRegistry({
        "ResNet": LazyFactory("fedcore.models.backbone.convolutional.resnet", "ResNetModel"),
        "ResNet18": LazyFactory("fedcore.models.backbone.convolutional.resnet", "resnet18"),
        "ResNet34": LazyFactory("fedcore.models.backbone.convolutional.resnet", "resnet34"),
        "ResNet50": LazyFactory("fedcore.models.backbone.convolutional.resnet", "resnet50"),
        "ResNet101": LazyFactory("fedcore.models.backbone.convolutional.resnet", "resnet101"),
        "ResNet152": LazyFactory("fedcore.models.backbone.convolutional.resnet", "resnet152"),
    })
    INCEPTIONET_MODELS = LazyRegistry({
        "InceptionNet": LazyFactory("fedcore.models.backbone.convolutional.inception", "InceptionTimeModel")
    })

    SEGFORMER_MODELS = LazyRegistry({"segformer": LazyFactory("fedcore.models.backbone.pretrain_model.segformer", "segformer_pretrain")})

    DENSENET_MODELS = LazyRegistry({
        "densenet121": LazyFactory("torchvision.models.densenet", "densenet121"),
        "densenet169": LazyFactory("torchvision.models.densenet", "densenet169"),
        "densenet201": LazyFactory("torchvision.models.densenet", "densenet201"),
        "densenet161": LazyFactory("torchvision.models.densenet", "densenet161"),
    })

    # CHRONOS_MODELS = {'chronos-t5-small': chronos_small}

    TRANSFORMER_MODELS = LazyRegistry({'TST': LazyFactory("fedcore.models.backbone.transformers.tst", "TSTModel")})

    PRUNED_RESNET_MODELS = LazyRegistry({
        "ResNet18": LazyFactory("fedcore.models.backbone.convolutional.resnet", "pruned_resnet18"),
        "ResNet34": LazyFactory("fedcore.models.backbone.convolutional.resnet", "pruned_resnet34"),
        "ResNet50": LazyFactory("fedcore.models.backbone.convolutional.resnet", "pruned_resnet50"),
        "ResNet101": LazyFactory("fedcore.models.backbone.convolutional.resnet", "pruned_resnet101"),
        "ResNet152": LazyFactory("fedcore.models.backbone.convolutional.resnet", "pruned_resnet152"),
    })

    DETECTION_MODELS = LazyRegistry({"detection_model": LazyFactory("torchvision.models.detection.faster_rcnn", "fasterrcnn_mobilenet_v3_large_fpn")})

    CUSTOM_MODEL = LazyRegistry({"custom": LazyFactory("fedcore.models.backbone.custom.custom", "CustomModel")})


PRUNER_MODELS = AtomizedModel.PRUNER_MODELS.value
QUANTIZATION_MODELS = AtomizedModel.QUANTIZATION_MODELS.value
DISTILATION_MODELS = AtomizedModel.DISTILATION_MODELS.value
LOW_RANK_MODELS = AtomizedModel.LOW_RANK_MODELS.value
LORA_MODELS = AtomizedModel.LORA_MODELS.value
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

BACKBONE_MODELS = LazyRegistry.combine(
    MOBILENET_MODELS, INCEPTIONET_MODELS, EFFICIENTNET_MODELS,
    DENSENET_MODELS, RESNET_MODELS, SEGFORMER_MODELS,
    TRANSFORMER_MODELS, CUSTOM_MODEL,
)

DETECTION_MODELS = AtomizedModel.DETECTION_MODELS.value


def default_fedcore_availiable_operation(problem: str = "pruning"):
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
        "lora": LORA_MODELS.keys(),
        "detection": DETECTION_MODELS.keys(),
        "training": TRAINING_MODELS.keys(),
    }

    return list(operation_dict[problem])
