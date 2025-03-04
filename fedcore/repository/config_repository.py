from enum import Enum


class ComputeConfigConstant(Enum):
    DEFAULT_COMPUTE_CONFIG = {'backend': 'gpu',
                              'distributed': dict(processes=False,
                                                  n_workers=1,
                                                  threads_per_worker=1,
                                                  memory_limit=0.3
                                                  ),
                              'output_folder': './results',
                              'use_cache': None,
                              'automl_folder': {'optimisation_history': './results/opt_hist',
                                                'composition_results': './results/comp_res'}}


class AutomlLearningConfigConstant(Enum):
    DEFAULT_AUTOML_CONFIG = dict(timeout=10,
                                 pop_size=5,
                                 early_stopping_iterations=10,
                                 early_stopping_timeout=10,
                                 with_tuning=False,
                                 n_jobs=-1)


class NeuralModelLearningConfigConstant(Enum):
    DEFAULT_LEARNING_CONFIG = dict(epochs=15, learning_rate=0.0001)


class PeftLearningConfigConstant(Enum):
    DEFAULT_PRUNING_CONFIG = dict(pruning_iterations=5,
                                  importance='MagnitudeImportance',
                                  pruner_name='magnitude_pruner',
                                  importance_norm=1,
                                  pruning_ratio=0.5,
                                  finetune_params={'epochs': 5,
                                                   'custom_loss': None}
                                  )
    DEFAULT_QUANTISATION_CONFIG = {}
    DEFAULT_DISTILATION_CONFIG = {}
    DEFAULT_LOW_RANK_CONFIG = {}


class AutomlConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'use_automl': True,
                         'optimisation_strategy': {'optimisation_strategy':
                                                       {'mutation_agent': 'random',
                                                        'mutation_strategy': 'growth_mutation_strategy'},
                                                   'optimisation_agent': 'Industrial'}}
    DEFAULT_CLF_AUTOML_CONFIG = {'task': 'classification', **DEFAULT_SUBCONFIG}
    DEFAULT_REG_AUTOML_CONFIG = {'task': 'regression', **DEFAULT_SUBCONFIG}
    DEFAULT_TSF_AUTOML_CONFIG = {'task': 'ts_forecasting', 'task_params': {'forecast_length': 14}, **DEFAULT_SUBCONFIG}


class LearningConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'learning_strategy': 'from_scratch',
                         'learning_strategy_params': NeuralModelLearningConfigConstant.DEFAULT_LEARNING_CONFIG.value,
                         'peft_strategy': 'pruning',
                         'peft_strategy_params': PeftLearningConfigConstant.DEFAULT_PRUNING_CONFIG.value}
    DEFAULT_CLF_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'accuracy'}, **DEFAULT_SUBCONFIG}
    DEFAULT_REG_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'rmse'}, **DEFAULT_SUBCONFIG}
    DEFAULT_TSF_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'rmse'}, **DEFAULT_SUBCONFIG}
    TASK_MAPPING = {
        'classification': {
            'task': 'classification',
            'use_automl': True,
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}},
        'regression': {
            'task': 'regression',
            'use_automl': True,
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}},
        'ts_forecasting': {
            'task': 'ts_forecasting',
            'use_automl': True,
            'task_params': {
                'forecast_length': 14},
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}}}


class EdgeConfigConstant(Enum):
    DEFAULT_ONNX_CONFIG = {'device': 'gpu',
                           'inference': 'onnx'}


DEFAULT_AUTOML_LEARNING_CONFIG = AutomlLearningConfigConstant.DEFAULT_AUTOML_CONFIG.value
DEFAULT_COMPUTE_CONFIG = ComputeConfigConstant.DEFAULT_COMPUTE_CONFIG.value
DEFAULT_CLF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_CLF_AUTOML_CONFIG.value
DEFAULT_REG_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_REG_AUTOML_CONFIG.value
DEFAULT_TSF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_TSF_AUTOML_CONFIG.value

DEFAULT_CLF_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_CLF_LEARNING_CONFIG.value
DEFAULT_REG_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_REG_LEARNING_CONFIG.value
DEFAULT_TSF_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_TSF_LEARNING_CONFIG.value

DEFAULT_EDGE_CONFIG = EdgeConfigConstant.DEFAULT_ONNX_CONFIG.value

DEFAULT_CLF_API_CONFIG = {'device_config': DEFAULT_EDGE_CONFIG,
                          'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                          'learning_config': DEFAULT_CLF_LEARNING_CONFIG,
                          'compute_config': DEFAULT_COMPUTE_CONFIG}

DEFAULT_REG_API_CONFIG = {'device_config': DEFAULT_EDGE_CONFIG,
                          'automl_config': DEFAULT_REG_AUTOML_CONFIG,
                          'learning_config': DEFAULT_REG_LEARNING_CONFIG,
                          'compute_config': DEFAULT_COMPUTE_CONFIG}

DEFAULT_TSF_API_CONFIG = {'device_config': DEFAULT_EDGE_CONFIG,
                          'automl_config': DEFAULT_TSF_AUTOML_CONFIG,
                          'learning_config': DEFAULT_TSF_LEARNING_CONFIG,
                          'compute_config': DEFAULT_COMPUTE_CONFIG}

TASK_MAPPING = LearningConfigConstant.TASK_MAPPING.value
