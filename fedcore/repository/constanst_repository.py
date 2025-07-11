from enum import Enum, auto
from functools import partial

import torch
import torch_pruning as tp
import torchvision
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.verification_rules import (
    has_primary_nodes,
)
from fedot.core.repository.metrics_repository import QualityMetricsEnum
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
    TsForecastingParams,
)
from golem.core.dag.verification_rules import (
    has_no_cycle,
    has_no_isolated_nodes,
    has_one_root,
)
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.tuning.optuna_tuner import OptunaTuner
from torch import nn

from fedcore.architecture.dataset.task_specified.object_detection_datasets import YOLODataset
from fedcore.architecture.dataset.datasets_from_source import CustomDatasetForImages
from fedcore.architecture.dataset.task_specified.segmentation_dataset import (
    SegmentationDataset,
    SemanticSegmentationDataset,
)
from fedcore.metrics.api_metric import (
    calculate_classification_metric,
    calculate_computational_metric,
    calculate_forecasting_metric,
    calculate_regression_metric,
)
from fedcore.models.network_impl.decomposed_layers import DecomposableLayers  ### don't del

from fedcore.models.network_modules.layers.attention_layers import MultiHeadAttention
from fedcore.models.network_modules.losses import (
    CenterLoss,
    CenterPlusLoss,
    ExpWeightedLoss,
    FocalLoss,
    HuberLoss,
    LogCoshLoss,
    MaskedLossWrapper,
    RMSELoss,
    SMAPELoss,
    TweedieLoss,
)

from fedcore.repository.setups import QAT_1, PTQ_1
from fedcore.models.network_impl.hooks import LoggingHooks, ModelLearningHooks  # don't del
from fedcore.algorithm.low_rank.hooks import LRHooks  # don't del

from fedcore.losses.low_rank_loss import HoyerLoss, OrthogonalLoss
from fedcore.architecture.utils.misc import EnumNoValue

from fedcore.models.network_impl.hooks import (
    Optimizers, Schedulers, ModelLearningHooks, LoggingHooks,
)  # don't del
from fedcore.algorithm.low_rank.rank_pruning import SLRStrategiesEnum  # don't del

default_param_values_dict = dict(
    problem=None,
    task_params=None,
    timeout=None,
    n_jobs=-1,
    logging_level=50,
    seed=42,
    parallelization_mode="populational",
    show_progress=True,
    max_depth=6,
    max_arity=3,
    pop_size=20,
    num_of_generations=None,
    keep_n_best=1,
    available_operations=None,
    metric=None,
    cv_folds=2,
    genetic_scheme=None,
    early_stopping_iterations=None,
    early_stopping_timeout=10,
    optimizer=None,
    collect_intermediate_metric=False,
    max_pipeline_fit_time=None,
    initial_assumption=None,
    preset=None,
    use_pipelines_cache=True,
    use_preprocessing_cache=True,
    use_input_preprocessing=True,
    use_auto_preprocessing=False,
    use_meta_rules=False,
    cache_dir=None,
    keep_history=True,
    history_dir=None,
    with_tuning=True,
)


class FedotTaskEnum(Enum):
    classification = Task(TaskTypesEnum.classification)
    regression = Task(TaskTypesEnum.regression)
    ts_forecasting = Task(
        TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=1)
    )


class FedCoreTaskEnum(Enum):  # FEDCORE_TASK
    pruning = auto()
    quantization = auto()
    distilation = auto()
    low_rank = auto()
    evo_composed = auto()


class CVTasks(Enum):
    classification = auto()
    segmentation = auto()
    object_detection = auto()


class FedCoreCVDataset(Enum):  # FEDCORE_CV_DATASET
    classification = CustomDatasetForImages
    segmentation = SegmentationDataset
    semantic_segmentation = SemanticSegmentationDataset
    object_detection = CustomDatasetForImages
    object_detection_YOLO = YOLODataset


class FedotTunerStrategy(Enum):  # FEDOT_TUNER_STRATEGY = {"optuna": OptunaTuner}
    optuna = OptunaTuner


class FedotEvoMultiStrategy(Enum):
    spea2 = SelectionTypesEnum.spea2
    tournament = SelectionTypesEnum.tournament


class FedotMutationStrategy(Enum):
    params_mutation_strategy = [0.8, 0.2]
    growth_mutation_strategy = [0.3, 0.7]
    initial_population_diversity_strategy = [1.0, 0.0]


class FedotGeneticMultiStrategy(Enum):
    steady_state = GeneticSchemeTypesEnum.steady_state
    generational = GeneticSchemeTypesEnum.generational
    parameter_free = GeneticSchemeTypesEnum.parameter_free


class FedotOperationConstant(Enum):
    FEDOT_GET_METRICS = {
        "regression": calculate_regression_metric,
        "ts_forecasting": calculate_forecasting_metric,
        "classification": calculate_classification_metric,
        "computational_fedcore": calculate_computational_metric,
        "computational_original": calculate_computational_metric
    }

    FEDOT_API_PARAMS = default_param_values_dict

    FEDOT_TUNER_STRATEGY = EnumNoValue(FedotTunerStrategy)

    FEDOT_EVO_MULTI_STRATEGY = EnumNoValue(FedotEvoMultiStrategy)

    FEDOT_GENETIC_MULTI_STRATEGY = EnumNoValue(FedotGeneticMultiStrategy)

    AVAILABLE_CLS_OPERATIONS = tuple()

    AVAILABLE_REG_OPERATIONS = tuple()

    FEDCORE_GRAPH_VALIDATION = (
        has_one_root,  # model have root node and it is a GraphNode
        has_no_cycle,  # model dont contain cycle (lead to infinity eval loop)
        has_no_isolated_nodes,  # model dont have isolated operation (impossible to get final predict)
        has_primary_nodes,  # model must contain primary node (root of computational tree)
        # has_final_operation_as_model,
        # has_no_conflicts_with_data_flow,
        # has_correct_data_connections,
        # has_no_conflicts_during_multitask,
        # has_correct_data_sources
    )

    # FEDOT_ASSUMPTIONS = {
    #     "pruning": partial(PipelineBuilder().add_node, operation_type="pruning_model"),
    #     "low_rank": partial(PipelineBuilder().add_node, operation_type="low_rank_model"),
    #     "quantization": partial(PipelineBuilder().add_node, operation_type="quantization_model"),
    #     "distilation": PipelineBuilder().add_node("distilation_model"),
    #     "detection": PipelineBuilder().add_node(
    #         "detection_model", params={"pretrained": True}
    #     ),
    #     "training": PipelineBuilder().add_node("training_model"),
    # }

    FEDOT_ENSEMBLE_ASSUMPTIONS = {}

class PEFTStrategies(Enum):
    pruning = partial(PipelineBuilder().add_node, operation_type="pruning_model")
    low_rank = partial(PipelineBuilder().add_node, operation_type="low_rank_model")
    quantization = partial(PipelineBuilder().add_node, operation_type='quantization_model')
    distilation = partial(PipelineBuilder().add_node, operation_type="distilation_model")
    detection = partial(PipelineBuilder().add_node, operation_type="detection_model", params={"pretrained": True})
    training = partial(PipelineBuilder().add_node, operation_type="training_model")


class ModelCompressionConstant(Enum):
    ENERGY_THR = [0.9, 0.95, 0.99, 0.999]
    DECOMPOSE_MODE = "channel"
    COMPOSE_MODE = "two_layers"
    # HOER_LOSS = 1
    # ORTOGONAL_LOSS = 5
    MODELS_FROM_LENGTH = {
        122: "ResNet18",
        218: "ResNet34",
        320: "ResNet50",
        626: "ResNet101",
        932: "ResNet152",
    }
    PRUNERS = {
        "magnitude_pruner": tp.pruner.MagnitudePruner,
        "group_norm_pruner": tp.pruner.GroupNormPruner,
        "batch_norm_pruner": tp.pruner.BNScalePruner,
        "growing_reg_pruner": tp.pruner.GrowingRegPruner,
        "meta_pruner": tp.pruner.MetaPruner,
        "manual_conv": tp.pruner.MetaPruner,
    }

    PRUNER_REQUIRED_GRADS = {
        "TaylorImportance": tp.importance.TaylorImportance,
        "GroupTaylorImportance": tp.importance.GroupTaylorImportance,
    }

    PRUNER_REQUIRED_REG = {
        "HessianImportance": tp.importance.HessianImportance,
        "BNScaleImportance": tp.importance.BNScaleImportance,
        "GroupMagnitudeImportance": tp.importance.GroupMagnitudeImportance,
        "GroupHessianImportance": tp.importance.GroupHessianImportance,
    }

    PRUNER_WITHOUT_REQUIREMENTS = {
        "MagnitudeImportance": tp.importance.MagnitudeImportance,
        "LAMPImportance": tp.importance.LAMPImportance,
        "RandomImportance": tp.importance.RandomImportance,
    }

    # PRUNING_IMPORTANCE = {
    #     "MagnitudeImportance": tp.importance.MagnitudeImportance,
    #     "TaylorImportance": tp.importance.TaylorImportance,
    #     "HessianImportance": tp.importance.HessianImportance,
    #     "BNScaleImportance": tp.importance.BNScaleImportance,
    #     "LAMPImportance": tp.importance.LAMPImportance,
    #     "RandomImportance": tp.importance.RandomImportance,
    #     "GroupNormImportance": tp.importance.GroupNormImportance,
    #     "GroupTaylorImportance": tp.importance.GroupTaylorImportance,
    #     "GroupHessianImportance": tp.importance.GroupHessianImportance,
    # }
    GROUP_PRUNING_IMPORTANCE = {
        "GroupNormImportance": tp.importance.GroupMagnitudeImportance,
        "GroupTaylorImportance": tp.importance.GroupTaylorImportance,
        "GroupHessianImportance": tp.importance.GroupHessianImportance,
    }

    PRUNING_FUNC = {
        "conv_out": tp.prune_conv_out_channels,
        "conv_in": tp.prune_conv_in_channels,
        "batchnorm_out": tp.prune_batchnorm_out_channels,
        "batchnorm_in": tp.prune_batchnorm_in_channels,
        "linear_out": tp.prune_linear_out_channels,
        "linear_in": tp.prune_linear_in_channels,
        "embedding_out": tp.prune_embedding_out_channels,
        "embedding_in": tp.prune_embedding_in_channels,
        "parameter_out": tp.prune_parameter_in_channels,
        "parameter_in": tp.prune_parameter_out_channels,
        "mha_out": tp.prune_multihead_attention_out_channels,
        "mha_in": tp.prune_multihead_attention_in_channels,
    }
    # MANUAL_PRUNING_STRATEGY = {
    #     "manual_conv": ["conv_out", "conv_in"],
    #     "manual_linear": ["linear_out", "linear_in"],
    #     "manual_attention": ["mha_out", "mha_in"],
    #     "manual_parameter": ["conv_out", "conv_in"],
    #     "manual_embedding": ["embedding_out", "embedding_in"],
    # }

    PRUNING_NORMS = [0, 1, 2]
    PRUNING_REDUCTION = ["sum", "mean", "max", 'prod', 'first']
    PRUNING_NORMALIZE = ["sum", "mean", "max", 'gaussian']

    PRUNING_LAYER_TYPE = {torch.nn.Linear: 'TorchLinear',
                          torch.nn.MultiheadAttention: 'TorchMHA',
                          MultiHeadAttention: 'FedCoreMHA'
                          }

    PROHIBIT_TO_DECOMPOSE = {torch.nn.modules.linear.NonDynamicallyQuantizableLinear}

    QUANT_MODEL_TYPES = {
        # "pytorch": PyTorchModel,
        # "pytorch_ipex": IPEXModel,
        # "pytorch_fx": PyTorchFXModel,
        # "onnxruntime": ONNXModel,
        # "onnxrt_qlinearops": ONNXModel,
        # "onnxrt_qdq": ONNXModel,
        # "onnxrt_integerops": ONNXModel,
    }


class PrunerImportances(Enum):
    magnitude = tp.importance.MagnitudeImportance
    activation_entropy = 'custom_depth'
    taylor = tp.importance.TaylorImportance
    hessian = tp.importance.HessianImportance
    bn_scale = tp.importance.BNScaleImportance
    lamp = tp.importance.LAMPImportance
    random = tp.importance.RandomImportance
    group_magnitude = tp.importance.GroupMagnitudeImportance
    group_taylor = tp.importance.GroupTaylorImportance
    group_hessian = tp.importance.GroupHessianImportance


class TorchLossesConstant(Enum):
    cross_entropy = nn.CrossEntropyLoss
    binary_cross_entropy = nn.BCEWithLogitsLoss
    mse = nn.MSELoss
    kl = nn.KLDivLoss  #
    rmse = RMSELoss
    smape = SMAPELoss
    tweedy = TweedieLoss
    focal = FocalLoss
    center_plus = CenterPlusLoss
    center = CenterLoss
    masked = MaskedLossWrapper
    log_cosh = LogCoshLoss
    huber = HuberLoss
    exp_weighted = ExpWeightedLoss


class DistilationMetricsEnum(QualityMetricsEnum):
    intermediate_layers_attention = "intermediate_attention"
    intermediate_layers_feature = "intermediate_feature"
    last_layer = "last_layer"
    RMSE = RMSELoss
    SMAPE = SMAPELoss
    TWEEDIE_LOSS = TweedieLoss
    FOCAL_LOSS = FocalLoss
    CENTER_PLUS_LOSS = CenterPlusLoss
    CENTER_LOSS = CenterLoss
    MASK_LOSS = MaskedLossWrapper
    LOG_COSH_LOSS = LogCoshLoss
    HUBER_LOSS = HuberLoss
    EXPONENTIAL_WEIGHTED_LOSS = ExpWeightedLoss


class InferenceMetricsEnum(QualityMetricsEnum):
    latency = "latency"
    throughput = "throughput"


class CVMetricsEnum(QualityMetricsEnum):
    cv_clf_metric = "cv_clf_metric"


class ONNX_CONFIG(Enum):
    INT8_CONFIG = {
        "dtype": "int8",
        "opset_version": 16,
        "quant_format": "QDQ",  # or "QLinear"
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": [0], "output": [0]},
    }
    INT5_CONFIG = {
        "dtype": "int5",
        "opset_version": 16,
        "quant_format": "QDQ",  # or "QLinear"
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": [0], "output": [0]},
    }
    INT4_CONFIG = {
        "dtype": "int4",
        "opset_version": 16,
        "quant_format": "QDQ",  # or "QLinear"
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": [0], "output": [0]},
    }


class FedcoreInitialAssumptions(Enum):
    qat_1 = QAT_1
    ptq_1 = PTQ_1


class StructureCriterions(Enum):
    hoer = HoyerLoss
    orthogonal = OrthogonalLoss


class TorchvisionBenchmark(Enum):
    classification = ["CIFAR10", "CIFAR100", 'FasnionMNIST']
    object_detection = ['VOCDetection', 'COCO']
    segmentation = ['VOCSegmentation']


class HistoryVisualisationParams(Enum):
    frame_duration = 3000
    pruning_params = ['importance', 'importance_norm', 'pruning_ratio']


AVAILABLE_REG_OPERATIONS = FedotOperationConstant.AVAILABLE_REG_OPERATIONS.value
AVAILABLE_CLS_OPERATIONS = FedotOperationConstant.AVAILABLE_CLS_OPERATIONS.value
# EXCLUDED_OPERATION_MUTATION = FedotOperationConstant.EXCLUDED_OPERATION_MUTATION.value
# FEDOT_TASK = FedotOperationConstant.FEDOT_TASK.value
FEDOT_TASK = EnumNoValue(FedotTaskEnum)
# FEDOT_ASSUMPTIONS = FedotOperationConstant.FEDOT_ASSUMPTIONS.value  ###
FEDOT_ASSUMPTIONS = EnumNoValue(PEFTStrategies)
FEDOT_API_PARAMS = FedotOperationConstant.FEDOT_API_PARAMS.value
FEDOT_ENSEMBLE_ASSUMPTIONS = FedotOperationConstant.FEDOT_ENSEMBLE_ASSUMPTIONS.value
FEDOT_TUNER_STRATEGY = FedotOperationConstant.FEDOT_TUNER_STRATEGY.value
FEDOT_EVO_MULTI_STRATEGY = FedotOperationConstant.FEDOT_EVO_MULTI_STRATEGY.value
FEDOT_GENETIC_MULTI_STRATEGY = FedotOperationConstant.FEDOT_GENETIC_MULTI_STRATEGY.value
FEDOT_GET_METRICS = FedotOperationConstant.FEDOT_GET_METRICS.value
# FEDCORE_TASK = FedotOperationConstant.FEDCORE_TASK.value
FEDCORE_TASK = EnumNoValue(FedCoreTaskEnum)
# CV_TASK = FedotOperationConstant.CV_TASK.value
CV_TASK = EnumNoValue(CVTasks)
# FEDCORE_CV_DATASET = FedotOperationConstant.FEDCORE_CV_DATASET.value
FEDCORE_CV_DATASET = EnumNoValue(FedCoreCVDataset)
# FEDCORE_MUTATION_STRATEGY = FedotOperationConstant.FEDOT_MUTATION_STRATEGY.value
FEDCORE_MUTATION_STRATEGY = EnumNoValue(FedotMutationStrategy)
FEDCORE_GRAPH_VALIDATION = FedotOperationConstant.FEDCORE_GRAPH_VALIDATION.value

# ENERGY_THR = ModelCompressionConstant.ENERGY_THR.value
DECOMPOSE_MODE = ModelCompressionConstant.DECOMPOSE_MODE.value
COMPOSE_MODE = ModelCompressionConstant.COMPOSE_MODE.value
DECOMPOSABLE_LAYERS: dict = DecomposableLayers
DIM_SUM_LIM: int = 1024
DIM_LIM: int = 2048
PROHIBIT_TO_DECOMPOSE = ModelCompressionConstant.PROHIBIT_TO_DECOMPOSE.value
# HOER_LOSS = ModelCompressionConstant.HOER_LOSS.value
# ORTOGONAL_LOSS = ModelCompressionConstant.ORTOGONAL_LOSS.value
MODELS_FROM_LENGTH = ModelCompressionConstant.MODELS_FROM_LENGTH.value
PRUNERS = ModelCompressionConstant.PRUNERS.value
PRUNING_IMPORTANCE = EnumNoValue(PrunerImportances)
PRUNING_NORMS = ModelCompressionConstant.PRUNING_NORMS.value
PRUNING_REDUCTION = ModelCompressionConstant.PRUNING_REDUCTION.value
PRUNING_NORMALIZE = ModelCompressionConstant.PRUNING_NORMALIZE.value
# PRUNING_LAYERS_IMPL = ModelCompressionConstant.PRUNING_LAYERS_IMPL.value
GROUP_PRUNING_IMPORTANCE = ModelCompressionConstant.GROUP_PRUNING_IMPORTANCE.value
PRUNER_REQUIRED_REG = ModelCompressionConstant.PRUNER_REQUIRED_REG.value
PRUNER_REQUIRED_GRADS = ModelCompressionConstant.PRUNER_REQUIRED_GRADS.value
PRUNER_WITHOUT_REQUIREMENTS = ModelCompressionConstant.PRUNER_WITHOUT_REQUIREMENTS.value
# MANUAL_PRUNING_STRATEGY = ModelCompressionConstant.MANUAL_PRUNING_STRATEGY.value
PRUNING_FUNC = ModelCompressionConstant.PRUNING_FUNC.value
PRUNING_LAYER_TYPE = ModelCompressionConstant.PRUNING_LAYER_TYPE.value
# QUANT_MODEL_TYPES = ModelCompressionConstant.QUANT_MODEL_TYPES.value
INITIAL_ASSUMPTIONS = {kvp.name: kvp.value for kvp in FedcoreInitialAssumptions}

CROSS_ENTROPY = TorchLossesConstant.cross_entropy.value
MULTI_CLASS_CROSS_ENTROPY = TorchLossesConstant.binary_cross_entropy.value
MSE = TorchLossesConstant.mse.value
KL_LOSS = TorchLossesConstant.kl.value
# CONTRASTIVE_LOSS = ContrastiveLossesEnum.CONTRASTIVE_LOSS.value
# VICREG_LOSS = ContrastiveLossesEnum.VICREG_LOSS.value

RMSE = TorchLossesConstant.rmse.value
SMAPE = TorchLossesConstant.smape.value
# TWEEDIE_LOSS = TorchLossesConstant.TWEEDIE_LOSS.value
# FOCAL_LOSS = TorchLossesConstant.FOCAL_LOSS.value
# CENTER_PLUS_LOSS = TorchLossesConstant.CENTER_PLUS_LOSS.value
# CENTER_LOSS = TorchLossesConstant.CENTER_LOSS.value
# MASK_LOSS = TorchLossesConstant.MASK_LOSS.value
# LOG_COSH_LOSS = TorchLossesConstant.LOG_COSH_LOSS.value
# HUBER_LOSS = TorchLossesConstant.HUBER_LOSS.value
# EXPONENTIAL_WEIGHTED_LOSS = TorchLossesConstant.EXPONENTIAL_WEIGHTED_LOSS.value
ONNX_INT8_CONFIG = ONNX_CONFIG.INT8_CONFIG.value
DEFAULT_TORCH_DATASET = {
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    'FasnionMNIST': torchvision.datasets.FashionMNIST,
    'EuroSAT': torchvision.datasets.EuroSAT,
    "MNIST": torchvision.datasets.MNIST,
    'COCO': torchvision.datasets.CocoDetection,
    'ImageNet': torchvision.datasets.ImageNet,
    'Imagenette': torchvision.datasets.Imagenette,
    'VOCSegmentation': torchvision.datasets.VOCSegmentation,
    'VOCDetection': torchvision.datasets.VOCDetection
}

HISTORY_VIZ_PARAMS = HistoryVisualisationParams
