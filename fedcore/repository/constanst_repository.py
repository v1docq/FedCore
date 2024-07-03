import torch_pruning as tp
from enum import Enum
from functools import partial
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from torch import nn
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.sequential import SequentialTuner
from multiprocessing import cpu_count
import math

from fedcore.models.network_modules.losses import CenterLoss, CenterPlusLoss, ExpWeightedLoss, FocalLoss, \
    HuberLoss, LogCoshLoss, MaskedLossWrapper, RMSELoss, SMAPELoss, TweedieLoss


class FedotOperationConstant(Enum):
    FEDOT_TASK = {'classification': Task(TaskTypesEnum.classification),
                  'regression': Task(TaskTypesEnum.regression),
                  'ts_forecasting': Task(TaskTypesEnum.ts_forecasting,
                                         TsForecastingParams(forecast_length=1))}
    EXCLUDED_OPERATION_MUTATION = {
        'regression': ['one_hot_encoding',
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
        'ts_forecasting': [
            'one_hot_encoding',
            'label_encoding',
            'isolation_forest_class'
            'xgbreg',
            'sgdr',
            'treg',
            'knnreg',
            'dtreg'
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
        ]}
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

    FEDOT_TUNING_METRICS = {'classification': ClassificationMetricsEnum.accuracy,
                            'regression': RegressionMetricsEnum.RMSE}
    FEDOT_TUNER_STRATEGY = {'sequential': partial(SequentialTuner, inverse_node_order=True),
                            'simultaneous': SimultaneousTuner,
                            'IOptTuner': IOptTuner,
                            'optuna': OptunaTuner}
    FEDOT_HEAD_ENSEMBLE = {'regression': 'fedot_regr',
                           'classification': 'fedot_cls'}
    FEDOT_ATOMIZE_OPERATION = {'regression': 'fedot_regr',
                               'classification': 'fedot_cls'}
    AVAILABLE_CLS_OPERATIONS = [
        'rf',
        'logit',
        'scaling',
        'normalization',
        'xgboost',
        'dt',
        'mlp',
        'kernel_pca']

    AVAILABLE_REG_OPERATIONS = [
        'scaling',
        'normalization',
        'xgbreg',
        'dtreg',
        'treg',
        'kernel_pca'
    ]

    FEDOT_ASSUMPTIONS = {
        'pruning': PipelineBuilder().add_node('pruner_model', params={'channels_to_prune': [2, 6, 9],
                                                                      'epochs': 50}),
        'quantisation': PipelineBuilder().add_node('pruner_model', params={'channels_to_prune': [2, 6, 9],
                                                                           'epochs': 50}),
    }

    FEDOT_ENSEMBLE_ASSUMPTIONS = {
        'pruning': PipelineBuilder().add_node('logit')
    }


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

    PRUNING_IMPORTANCE = {"MagnitudeImportance": tp.importance.MagnitudeImportance,
                          "TaylorImportance": tp.importance.TaylorImportance,
                          "HessianImportance": tp.importance.HessianImportance,
                          "BNScaleImportance": tp.importance.BNScaleImportance,
                          "LAMPImportance": tp.importance.LAMPImportance,
                          "RandomImportance": tp.importance.RandomImportance,
                          }
    PRUNING_NORMS = [0, 1, 2]
    PRUNING_REDUCTION = ["sum", "mean", "max", 'prod', 'first']
    PRUNING_NORMALIZE = ["sum", "mean", "max", 'gaussian']


class TorchLossesConstant(Enum):
    CROSS_ENTROPY = nn.CrossEntropyLoss
    MULTI_CLASS_CROSS_ENTROPY = nn.BCEWithLogitsLoss
    MSE = nn.MSELoss
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

class DataTypeConstant(Enum):
    MULTI_ARRAY = DataTypesEnum.image
    MATRIX = DataTypesEnum.table

class ComputationalConstant(Enum):
    CPU_NUMBERS = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
    GLOBAL_IMPORTS = {
        'numpy': 'np',
        'cupy': 'np',
        'torch': 'torch',
        'torch.nn': 'nn',
        'torch.nn.functional': 'F'
    }
    BATCH_SIZE_FOR_FEDOT_WORKER = 1000
    FEDOT_WORKER_NUM = 5
    FEDOT_WORKER_TIMEOUT_PARTITION = 4
    PATIENCE_FOR_EARLY_STOP = 15

AVAILABLE_REG_OPERATIONS = FedotOperationConstant.AVAILABLE_REG_OPERATIONS.value
AVAILABLE_CLS_OPERATIONS = FedotOperationConstant.AVAILABLE_CLS_OPERATIONS.value
EXCLUDED_OPERATION_MUTATION = FedotOperationConstant.EXCLUDED_OPERATION_MUTATION.value
FEDOT_HEAD_ENSEMBLE = FedotOperationConstant.FEDOT_HEAD_ENSEMBLE.value
FEDOT_TASK = FedotOperationConstant.FEDOT_TASK.value
FEDOT_ATOMIZE_OPERATION = FedotOperationConstant.FEDOT_ATOMIZE_OPERATION.value
FEDOT_TUNING_METRICS = FedotOperationConstant.FEDOT_TUNING_METRICS.value
FEDOT_ASSUMPTIONS = FedotOperationConstant.FEDOT_ASSUMPTIONS.value
FEDOT_API_PARAMS = FedotOperationConstant.FEDOT_API_PARAMS.value
FEDOT_ENSEMBLE_ASSUMPTIONS = FedotOperationConstant.FEDOT_ENSEMBLE_ASSUMPTIONS.value
FEDOT_TUNER_STRATEGY = FedotOperationConstant.FEDOT_TUNER_STRATEGY.value

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
