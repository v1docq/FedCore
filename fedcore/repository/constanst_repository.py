import torch
import torch_pruning as tp
from enum import Enum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
import torchvision
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.metrics_repository import QualityMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from torch import nn

from fedcore.architecture.dataset.object_detection_datasets import COCODataset, YOLODataset
from fedcore.architecture.dataset.prediction_datasets import CustomDatasetForImages
from fedcore.architecture.dataset.segmentation_dataset import SemanticSegmentationDataset, SegmentationDataset
from fedcore.architecture.dataset.segmentation_dataset import SemanticSegmentationDataset
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.sequential import SequentialTuner
from multiprocessing import cpu_count
import math

from fedcore.models.network_modules.losses import CenterLoss, CenterPlusLoss, ExpWeightedLoss, FocalLoss, \
    HuberLoss, LogCoshLoss, MaskedLossWrapper, RMSELoss, SMAPELoss, TweedieLoss

def default_device(device_type: str = 'CPU'):
    """Return or set default device. Modified from fastai.

    Args:
        device_type: 'CUDA' or 'CPU' or None (default: 'CUDA'). If None, use CUDA if available, else CPU.

    Returns:
        torch.device: The default device: CUDA if available, else CPU.

    """
    if device_type == 'CUDA':
        defaults.use_cuda = True
        return torch.device("cuda")
    elif device_type == 'cpu':
        defaults.use_cuda = False
        return torch.device("cpu")

    if device_type is None:
        if torch.cuda.is_available() or _has_mps():
            device_type = True
    if device_type:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        if _has_mps():
            return torch.device("mps")
class FedotOperationConstant(Enum):
    FEDOT_TASK = {'classification': Task(TaskTypesEnum.classification),
                  'regression': Task(TaskTypesEnum.regression),
                  'ts_forecasting': Task(TaskTypesEnum.ts_forecasting,
                                         TsForecastingParams(forecast_length=1))}

    FEDCORE_TASK = ['pruning', 'quantisation', 'distilation', 'low_rank', 'evo_composed']
    CV_TASK = ['classification', 'segmentation', 'object_detection']
    FEDCORE_CV_DATASET = {'classification': CustomDatasetForImages,
                          'segmentation': SegmentationDataset,
                          'semantic_segmentation': SemanticSegmentationDataset,
                          'object_detection': CustomDatasetForImages,
                          'object_detection_YOLO': YOLODataset}

                  'regression': Task(TaskTypesEnum.regression)
    }
    EXCLUDED_OPERATION_MUTATION = {
        'regression': [
            'one_hot_encoding',
            'label_encoding',
            'isolation_forest_class',
            'tst_model',
            'omniscale_model',
            'isolation_forest_reg',
            'inception_model',
            'xcm_model',
            'resnet_model',
            'signal_extractor',
            'recurrence_extractor'
        ],
        'classification': [
            'isolation_forest_reg',
            'tst_model',
            'resnet_model',
            'xcm_model',
            'one_hot_encoding',
            'label_encoding',
            'isolation_forest_class',
            'signal_extractor',
            'knnreg',
            'recurrence_extractor'
        ]
    }
    FEDOT_API_PARAMS = default_param_values_dict = dict(problem=None,
                                                        task_params=None,
                                                        timeout=None,
                                                        n_jobs=-1,
                                                        logging_level=50,
                                                        seed=42,
                                                        parallelization_mode='populational',
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
                                                        with_tuning=True
                                                        )

    FEDOT_TUNER_STRATEGY = {'optuna': OptunaTuner}

    FEDOT_EVO_MULTI_STRATEGY = {'spea2': SelectionTypesEnum.spea2,
                                'tournament': SelectionTypesEnum.tournament}

    FEDOT_GENETIC_MULTI_STRATEGY = {'steady_state': GeneticSchemeTypesEnum.steady_state,
                                    'generational': GeneticSchemeTypesEnum.generational,
                                    'parameter_free': GeneticSchemeTypesEnum.parameter_free}
    AVAILABLE_CLS_OPERATIONS = []

    AVAILABLE_REG_OPERATIONS = []

    FEDOT_ASSUMPTIONS = {
        'pruning': PipelineBuilder().add_node('pruning_model'),
        'low_rank': PipelineBuilder().add_node('low_rank_model'),
        'quantisation': PipelineBuilder().add_node('post_training_quant'),
        'distilation': PipelineBuilder().add_node('distilation_model'),
        'pruning': PipelineBuilder().add_node('pruner_model', params={'channels_to_prune': [2, 6, 9],
                                                                      'epochs': 50}),
        'quantisation': PipelineBuilder().add_node('pruner_model', params={'channels_to_prune': [2, 6, 9],
                                                                           'epochs': 50}),
        'detection': PipelineBuilder().add_node('detection_model', params={'pretrained': True})
    }

    FEDOT_ENSEMBLE_ASSUMPTIONS = {}


class ModelCompressionConstant(Enum):
    ENERGY_THR = [0.9, 0.95, 0.99, 0.999]
    DECOMPOSE_MODE = 'channel'
    FORWARD_MODE = 'one_layer'
    HOER_LOSS = 0.1
    ORTOGONAL_LOSS = 10
    MODELS_FROM_LENGTH = {
        122: 'ResNet18',
        218: 'ResNet34',
        320: 'ResNet50',
        626: 'ResNet101',
        932: 'ResNet152',
    }
    PRUNERS = {'magnitude_pruner': tp.pruner.MagnitudePruner,
               'group_norm_pruner': tp.pruner.GroupNormPruner,
               'batch_norm_pruner': tp.pruner.BNScalePruner,
               'growing_reg_pruner': tp.pruner.GrowingRegPruner,
               'meta_pruner': tp.pruner.MetaPruner}

    PRUNER_REQUIRED_GRADS = {
        "TaylorImportance": tp.importance.TaylorImportance,
        "GroupTaylorImportance": tp.importance.GroupTaylorImportance,
    }

    PRUNER_REQUIRED_REG = {
        "HessianImportance": tp.importance.HessianImportance,
        "BNScaleImportance": tp.importance.BNScaleImportance,
        "GroupNormImportance": tp.importance.GroupNormImportance,
        "GroupHessianImportance": tp.importance.GroupHessianImportance
    }

    PRUNER_WITHOUT_REQUIREMENTS = {
        "MagnitudeImportance": tp.importance.MagnitudeImportance,
        "LAMPImportance": tp.importance.LAMPImportance,
        "RandomImportance": tp.importance.RandomImportance
    }

    PRUNING_IMPORTANCE = {"MagnitudeImportance": tp.importance.MagnitudeImportance,
                          "TaylorImportance": tp.importance.TaylorImportance,
                          "HessianImportance": tp.importance.HessianImportance,
                          "BNScaleImportance": tp.importance.BNScaleImportance,
                          "LAMPImportance": tp.importance.LAMPImportance,
                          "RandomImportance": tp.importance.RandomImportance,
                          "GroupNormImportance": tp.importance.GroupNormImportance,
                          "GroupTaylorImportance": tp.importance.GroupTaylorImportance,
                          "GroupHessianImportance": tp.importance.GroupHessianImportance
                          }
    GROUP_PRUNING_IMPORTANCE = {"GroupNormImportance": tp.importance.GroupNormImportance,
                                "GroupTaylorImportance": tp.importance.GroupTaylorImportance,
                                "GroupHessianImportance": tp.importance.GroupHessianImportance
                                }
    PRUNING_NORMS = [0, 1, 2]
    PRUNING_REDUCTION = ["sum", "mean", "max", 'prod', 'first']
    PRUNING_NORMALIZE = ["sum", "mean", "max", 'gaussian']
    PRUNING_LAYERS_IMPL = (torchvision.ops.misc.Conv2dNormActivation,
                           torch.nn.modules.container.Sequential,
                           torch.nn.modules.conv.Conv2d)


class TorchLossesConstant(Enum):
    CROSS_ENTROPY = nn.CrossEntropyLoss
    MULTI_CLASS_CROSS_ENTROPY = nn.BCEWithLogitsLoss
    MSE = nn.MSELoss
    KL_LOSS = nn.KLDivLoss  #


class DistilationMetricsEnum(QualityMetricsEnum):
    intermediate_layers_attention = 'intermediate_attention'
    intermediate_layers_feature = 'intermediate_feature'
    last_layer = 'last_layer'
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
    latency = 'latency'
    throughput = 'throughput'


class CVMetricsEnum(QualityMetricsEnum):
    cv_clf_metric = 'cv_clf_metric'


class ONNX_CONFIG(Enum):
    INT8_CONFIG = {
        'dtype': "int8",
        'opset_version': 16,
        'quant_format': "QDQ",  # or "QLinear"
        'input_names': ["input"],
        'output_names': ["output"],
        'dynamic_axes': {'input': [0], 'output': [0]}
    }
    INT5_CONFIG = {
        'dtype': "int5",
        'opset_version': 16,
        'quant_format': "QDQ",  # or "QLinear"
        'input_names': ["input"],
        'output_names': ["output"],
        'dynamic_axes': {'input': [0], 'output': [0]}
    }
    INT4_CONFIG = {
        'dtype': "int4",
        'opset_version': 16,
        'quant_format': "QDQ",  # or "QLinear"
        'input_names': ["input"],
        'output_names': ["output"],
        'dynamic_axes': {'input': [0], 'output': [0]}
    }


AVAILABLE_REG_OPERATIONS = FedotOperationConstant.AVAILABLE_REG_OPERATIONS.value
AVAILABLE_CLS_OPERATIONS = FedotOperationConstant.AVAILABLE_CLS_OPERATIONS.value
EXCLUDED_OPERATION_MUTATION = FedotOperationConstant.EXCLUDED_OPERATION_MUTATION.value
FEDOT_TASK = FedotOperationConstant.FEDOT_TASK.value
FEDOT_ASSUMPTIONS = FedotOperationConstant.FEDOT_ASSUMPTIONS.value
FEDOT_API_PARAMS = FedotOperationConstant.FEDOT_API_PARAMS.value
FEDOT_ENSEMBLE_ASSUMPTIONS = FedotOperationConstant.FEDOT_ENSEMBLE_ASSUMPTIONS.value
FEDOT_TUNER_STRATEGY = FedotOperationConstant.FEDOT_TUNER_STRATEGY.value
FEDOT_EVO_MULTI_STRATEGY = FedotOperationConstant.FEDOT_EVO_MULTI_STRATEGY.value
FEDOT_GENETIC_MULTI_STRATEGY = FedotOperationConstant.FEDOT_GENETIC_MULTI_STRATEGY.value
FEDCORE_TASK = FedotOperationConstant.FEDCORE_TASK.value
CV_TASK = FedotOperationConstant.CV_TASK.value
FEDCORE_CV_DATASET = FedotOperationConstant.FEDCORE_CV_DATASET.value

ENERGY_THR = ModelCompressionConstant.ENERGY_THR.value
DECOMPOSE_MODE = ModelCompressionConstant.DECOMPOSE_MODE.value
FORWARD_MODE = ModelCompressionConstant.FORWARD_MODE.value
HOER_LOSS = ModelCompressionConstant.HOER_LOSS.value
ORTOGONAL_LOSS = ModelCompressionConstant.ORTOGONAL_LOSS.value
MODELS_FROM_LENGTH = ModelCompressionConstant.MODELS_FROM_LENGTH.value
PRUNERS = ModelCompressionConstant.PRUNERS.value
PRUNING_IMPORTANCE = ModelCompressionConstant.PRUNING_IMPORTANCE.value
PRUNING_NORMS = ModelCompressionConstant.PRUNING_NORMS.value
PRUNING_REDUCTION = ModelCompressionConstant.PRUNING_REDUCTION.value
PRUNING_NORMALIZE = ModelCompressionConstant.PRUNING_NORMALIZE.value
PRUNING_LAYERS_IMPL = ModelCompressionConstant.PRUNING_LAYERS_IMPL.value
GROUP_PRUNING_IMPORTANCE = ModelCompressionConstant.GROUP_PRUNING_IMPORTANCE.value
PRUNER_REQUIRED_REG = ModelCompressionConstant.PRUNER_REQUIRED_REG.value
PRUNER_REQUIRED_GRADS = ModelCompressionConstant.PRUNER_REQUIRED_GRADS.value
PRUNER_WITHOUT_REQUIREMENTS = ModelCompressionConstant.PRUNER_WITHOUT_REQUIREMENTS.value

CROSS_ENTROPY = TorchLossesConstant.CROSS_ENTROPY.value
MULTI_CLASS_CROSS_ENTROPY = TorchLossesConstant.MULTI_CLASS_CROSS_ENTROPY.value
MSE = TorchLossesConstant.MSE.value
RMSE = TorchLossesConstant.RMSE.value
SMAPE = TorchLossesConstant.SMAPE.value
TWEEDIE_LOSS = TorchLossesConstant.TWEEDIE_LOSS.value
FOCAL_LOSS = TorchLossesConstant.FOCAL_LOSS.value
CENTER_PLUS_LOSS = TorchLossesConstant.CENTER_PLUS_LOSS.value
CENTER_LOSS = TorchLossesConstant.CENTER_LOSS.value
MASK_LOSS = TorchLossesConstant.MASK_LOSS.value
LOG_COSH_LOSS = TorchLossesConstant.LOG_COSH_LOSS.value
HUBER_LOSS = TorchLossesConstant.HUBER_LOSS.value
EXPONENTIAL_WEIGHTED_LOSS = TorchLossesConstant.EXPONENTIAL_WEIGHTED_LOSS.value

MULTI_ARRAY = DataTypeConstant.MULTI_ARRAY.value
MATRIX = DataTypeConstant.MATRIX.value

CPU_NUMBERS = ComputationalConstant.CPU_NUMBERS.value
BATCH_SIZE_FOR_FEDOT_WORKER = ComputationalConstant.BATCH_SIZE_FOR_FEDOT_WORKER.value
FEDOT_WORKER_NUM = ComputationalConstant.FEDOT_WORKER_NUM.value
FEDOT_WORKER_TIMEOUT_PARTITION = ComputationalConstant.FEDOT_WORKER_TIMEOUT_PARTITION.value
PATIENCE_FOR_EARLY_STOP = ComputationalConstant.PATIENCE_FOR_EARLY_STOP.value
KL_LOSS = TorchLossesConstant.KL_LOSS.value

ONNX_INT8_CONFIG = ONNX_CONFIG.INT8_CONFIG.value
