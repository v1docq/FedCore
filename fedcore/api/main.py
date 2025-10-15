import logging
import os
import warnings
from functools import partial
from typing import Union, Optional
import numpy as np
import pandas as pd
import torch
from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.pipeline import Pipeline
from pymonad.either import Either
from pymonad.maybe import Maybe
from torch import Tensor
from torch.utils.data import DataLoader

from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.abstraction.decorators import DaskServer, exception_handler
from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import (
    FEDOT_ASSUMPTIONS,
    FEDOT_GET_METRICS,
)
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.api.api_configs import ConfigTemplate
from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer

warnings.filterwarnings("ignore")


class FedCore(Fedot):
    """Simplified FedCore API for model compression."""

    def __init__(self, api_config: ConfigTemplate, **kwargs):
        super(Fedot, self).__init__()
        api_config.update(kwargs)
        self.manager = api_config
        self.logger = logging.Logger('Fedcore')
        self.fedcore_model = None

    def __init_fedcore_backend(self, input_data: Optional[InputData] = None):
        self.logger.info('Initialising Fedcore Repository')
        self.repo = FedcoreModels().setup_repository()
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('Initialising solver')
        
        # Базовый конфиг Fedot
        fedot_config = getattr(self.manager.automl_config, 'fedot_config', {}) if hasattr(self.manager, 'automl_config') else {}
        
        self.manager.solver = Fedot(**fedot_config,
                                    use_input_preprocessing=False,
                                    use_auto_preprocessing=False)
        
        # Инициализация пайплайна
        peft_strategy = self.manager.config.get('peft_strategy', 'low_rank')
        peft_strategy_params = getattr(self.manager, 'learning_strategy_params', {}).to_dict() if hasattr(self.manager, 'learning_strategy_params') else {}
        
        if peft_strategy in FEDOT_ASSUMPTIONS:
            initial_assumption = FEDOT_ASSUMPTIONS[peft_strategy](params=peft_strategy_params)
            initial_pipeline = initial_assumption.build()
            self.manager.solver.params.data.update({'initial_assumption': initial_pipeline})
        
        return input_data

    def __init_dask(self, input_data):
        self.logger.info('Initialising Dask Server')
        try:
            dask_server = DaskServer(getattr(self.manager.compute_config, 'distributed', False))
            self.manager.dask_client = dask_server.client
            self.manager.dask_cluster = dask_server.cluster
        except Exception as e:
            self.logger.warning(f'Dask initialization failed: {e}')
            self.manager.dask_client = None
            self.manager.dask_cluster = None
        return input_data

    def _process_input_data(self, input_data):
        # Упрощенная обработка входных данных
        peft_task = self.manager.config.get('peft_strategy', 'low_rank')
        model = "ResNet18"  # значение по умолчанию
        learning_params = getattr(self.manager, 'learning_strategy_params', {})
        
        data_cls = DataCheck(
            peft_task=peft_task,
            model=model,
            learning_params=learning_params
        )
        train_data = Either.insert(input_data).then(data_cls.check_input_data).value
        
        # Упрощенный workaround
        if hasattr(train_data, 'features'):
            if hasattr(train_data.features, 'train_dataloader'):
                train_data.train_dataloader = train_data.features.train_dataloader
            if hasattr(train_data.features, 'val_dataloader'):
                train_data.val_dataloader = train_data.features.val_dataloader
        
        return train_data

    def fit(self, input_data: CompressionInputData, **kwargs):
        """Simplified fit method"""
        try:
            # Упрощенный пайплайн инициализации
            processed_data = self._process_input_data(input_data)
            processed_data = self.__init_fedcore_backend(processed_data)
            processed_data = self.__init_dask(processed_data)
            processed_data = self.__init_solver(processed_data)
            
            # Обучение
            self.fedcore_model = self.manager.solver.fit(processed_data)
            return self.fedcore_model
            
        except Exception as e:
            self.logger.error(f"Fit failed: {e}")
            raise

    def predict(self, predict_data: CompressionInputData, output_mode: str = 'labels', **kwargs):
        """Simplified predict method"""
        try:
            processed_data = self._process_input_data(predict_data)
            
            if self.fedcore_model is None:
                # Fallback to basic prediction
                learning_params = getattr(self.manager, 'learning_strategy_params', {}).to_dict() if hasattr(self.manager, 'learning_strategy_params') else {}
                learning_params['model'] = processed_data.target
                self.fedcore_model = BaseNeuralModel(learning_params)
            
            prediction = self.fedcore_model.predict(processed_data, output_mode)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Predict failed: {e}")
            raise

    def get_report(self, test_data: CompressionInputData):
        """Simplified report generation"""
        try:
            # Базовые метрики для демонстрации
            eval_regime = ['original', 'fedcore']
            
            # Получаем предсказания
            predictions = []
            for mode in eval_regime:
                try:
                    pred = self.predict(test_data, output_mode=mode)
                    predictions.append(pred if isinstance(pred, OutputData) else getattr(pred, 'predict', pred))
                except Exception as e:
                    self.logger.warning(f"Prediction for {mode} failed: {e}")
                    predictions.append(None)
            
            # Базовые метрики качества
            problem = 'classification'
            if hasattr(self.manager, 'automl_config') and hasattr(self.manager.automl_config, 'fedot_config'):
                problem = self.manager.automl_config.fedot_config.get('problem', 'classification')
            
            # Собираем результаты
            results = {}
            for i, (pred, regime) in enumerate(zip(predictions, eval_regime)):
                if pred is not None:
                    if problem in ['regression', 'ts_forecasting']:
                        metrics = ["mse", "mae", "r2"]
                    else:
                        metrics = ["accuracy", "f1"]
                    
                    # Используем API для расчета метрик
                    try:
                        metric_result = FEDOT_GET_METRICS[problem](
                            target=test_data.val_dataloader,
                            predict=pred,
                            metric_names=metrics
                        )
                        results[regime] = metric_result
                    except Exception as e:
                        self.logger.warning(f"Metric calculation for {regime} failed: {e}")
                        results[regime] = {metric: 0.0 for metric in metrics}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """Simplified shutdown"""
        if hasattr(self.manager, 'dask_client') and self.manager.dask_client:
            try:
                self.manager.dask_client.close()
            except:
                pass
        
        if hasattr(self.manager, 'dask_cluster') and self.manager.dask_cluster:
            try:
                self.manager.dask_cluster.close()
            except:
                pass