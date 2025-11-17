import os
import uuid
from typing import Any, Optional, Union
import logging

import numpy as np
import torch
import torch_pruning as tp
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.computational.devices import default_device, extract_device
from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from fedcore.models.network_impl.utils.trainer_factory import create_trainer_from_input_data
from torchinfo import summary
from fedcore.tools.registry.model_registry import ModelRegistry

DEVICE = default_device('cuda')


class BaseCompressionModel:
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("BaseCompressionModel.__init__() called")
        
        # self.epochs = params.get("epochs", 10)
        self.batch_size = params.get("batch_size", 16)
        self.activation = params.get("activation", "ReLU")
        # self.learning_rate = 0.001
        self.model = None
        # self.model_for_inference = None
        # self.optimizer = None
        self.params = params
        
        self._fedcore_id = params.get("fedcore_id")
        if self._fedcore_id is None:
            self._fedcore_id = f"fedcore_{uuid.uuid4().hex[:8]}"
        
        logger.debug(f"fedcore_id: {self._fedcore_id}")
        
        self._model_id_before = None
        self._model_id_after = None
        self._model_before_cached = None
        self._model_after_cached = None
        self._registry = ModelRegistry()
        
        logger.debug(f"BaseCompressionModel initialized with ModelRegistry, auto_cleanup={self._registry.auto_cleanup}")

    # def _save_and_clear_cache(self):
    #     """Save model and clear cache using ModelRegistry.
        
    #     Saves the current model to registry and clears memory cache.
    #     ModelRegistry handles proper cleanup including GPU memory management.
    #     """
    #     import logging
    #     logger = logging.getLogger(__name__)
        
    #     if self.model is None:
    #         logger.debug("No model to save, skipping cache clearing")
    #         return
        
    #     try:
    #         model_id = self._registry.register_model(
    #             fedcore_id=self._fedcore_id,
    #             model=self.model,
    #             stage="cache",
    #             mode=None,
    #             delete_model_after_save=True
    #         )
    #         self.model = None
    #         logger.info(f"Model saved to registry and cleared from memory. model_id={model_id}")

    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
                
    #     except Exception as e:
    #         logger.error(f"Failed to save model to registry: {e}")
    #         if self.model is not None:
    #             self.model.zero_grad()
    #             del self.model
    #             self.model = None
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()
    #             logger.warning("Manual cleanup performed due to registry failure")

    @property
    def model_before(self):
        """Get model_before from cache or registry."""
        if self._model_before_cached is not None:
            return self._model_before_cached
        
        if self._model_id_before is None:
            return None
        
        loaded_model = self._registry.load_model_from_latest_checkpoint(
            self._fedcore_id, self._model_id_before, DEVICE
        )
        
        if loaded_model is not None and isinstance(loaded_model, torch.nn.Module):
            self._model_before_cached = loaded_model
            return self._model_before_cached
        
        return None
    
    @model_before.setter
    def model_before(self, value):
        """Set model_before - stores in cache and optionally in registry."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"model_before setter called: value={'not None' if value else 'None'}, _model_id_before={self._model_id_before}")
        
        self._model_before_cached = value
        if value is not None and self._model_id_before is None:
            logger.info("Registering model_before in ModelRegistry")
            self._model_id_before = self._registry.register_model(
                fedcore_id=self._fedcore_id,
                model=value,
                stage="before",
                mode=None
            )
            logger.info(f"model_before registered with id={self._model_id_before}")
        else:
            logger.debug(f"Skipping registration: value is None={value is None}, already registered={self._model_id_before is not None}")
    
    @property
    def model_after(self):
        """Get model_after from cache or registry."""
        if self._model_after_cached is not None:
            return self._model_after_cached
        
        if self._model_id_after is None:
            return None
        
        loaded_model = self._registry.load_model_from_latest_checkpoint(
            self._fedcore_id, self._model_id_after, DEVICE
        )
        
        if loaded_model is not None and isinstance(loaded_model, torch.nn.Module):
            self._model_after_cached = loaded_model
            return self._model_after_cached
        
        return None
    
    @model_after.setter
    def model_after(self, value):
        """Set model_after - stores in cache and registers changes."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"model_after setter called: value={'not None' if value else 'None'}, _model_id_after={self._model_id_after}")
        
        self._model_after_cached = value
        if value is not None:
            if self._model_id_after is None:
                logger.info("Registering new model_after in ModelRegistry")
                self._model_id_after = self._registry.register_model(
                    fedcore_id=self._fedcore_id,
                    model=value,
                    stage="after",
                    mode=None
                )
                logger.info(f"model_after registered with id={self._model_id_after}")
            else:
                logger.info("Registering changes for model_after")
                self._registry.register_changes(
                    fedcore_id=self._fedcore_id,
                    model_id=self._model_id_after,
                    model=value,
                    stage="after",
                    mode=None
                )
                logger.info("model_after changes registered")
        else:
            logger.debug("Skipping registration: value is None")
    
    def _save_model_checkpoint(self, model, stage: str):
        """Save model checkpoint to registry.
        
        Args:
            model: Model to save
            stage: Stage name (e.g., 'before_compression', 'after_compression')
        """
        model_id = self._registry.register_model(
            fedcore_id=self._fedcore_id,
            model=model,
            stage=stage,
            mode=None
        )
        return model_id
    
    def _init_model(self, input_data, additional_hooks=tuple()):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("BaseCompressionModel._init_model() started")
        
        model = input_data.model
        logger.info(f"Model type from input_data.target: {type(model).__name__}")
        
        # Support passing a filesystem path to a checkpoint/model at the node input
        if isinstance(model, str):
            logger.info(f"Loading model from path: {model}")
            device = default_device()
            loaded = torch.load(model, map_location=device)
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
            else:
                model = loaded
            logger.info(f"Model loaded: type={type(model).__name__}")
        
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected model to be either file path or torch.nn.Module, got {type(model)}")

        logger.info("Calling model_before setter")
        self.model_before = model
        logger.info(f"model_before setter completed, _model_id_before={self._model_id_before}")
        
        # Create trainer using factory
        self.trainer = create_trainer_from_input_data(input_data, self.params)
        self.trainer.register_additional_hooks(additional_hooks)
        self.trainer.model = model

        return model

    def _fit_model(self, ts: CompressionInputData, split_data: bool = False):
        pass

    def _predict_model(self, x_test, output_mode: str = "default"):
        pass

    def _get_example_input(self, input_data: Union[InputData, CompressionInputData]):
        # Handle both CompressionInputData directly and InputData with features as CompressionInputData
        if isinstance(input_data, CompressionInputData):
            compression_data = input_data
        else:
            # input_data is InputData, features should be CompressionInputData
            compression_data = input_data.features
        
        # Prefer val_dataloader, fallback to train_dataloader if val_dataloader is None
        dataloader = compression_data.val_dataloader or compression_data.train_dataloader
        if dataloader is None:
            raise ValueError("Neither val_dataloader nor train_dataloader is available in input_data")
        
        b = next(iter(dataloader))
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return b[0]
        return b

    # def finetune(self, finetune_object: callable, finetune_data):
    #     # TODO del it! 1) finetune may be included into the train loop (just look at LowRank)
    #     # 2) here the logic is base and need to be more flexible (no extra loss, no scheduler, no different types batch handling)
    #     self.optimizer = finetune_object.optimizer(
    #         finetune_object.model.parameters(), lr=finetune_object.learning_rate
    #     )
    #     finetune_object.model.train()
    #     for epoch in range(5):  # loop over the dataset multiple times
    #         running_loss = 0.0
    #         for i, data in enumerate(finetune_data.features.train_dataloader, 0):
    #             # get the inputs; data is a list of [inputs, labels]
    #             inputs, labels = data
    #             # zero the parameter gradients
    #             self.optimizer.zero_grad()

    #             # forward + backward + optimize
    #             outputs = finetune_object.model(inputs.to(default_device()))
    #             loss = finetune_object.criterion(outputs, labels.to(default_device()))
    #             loss.backward()
    #             self.optimizer.step()

    #             # print statistics
    #             running_loss += loss.item()
    #             if i % 200 == 0:  # print every 20000 mini-batches
    #                 print(
    #                     "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 200)
    #                 )
    #                 running_loss = 0.0
    #     finetune_object.model.eval()
    #     return finetune_object

    def fit(self, input_data: CompressionInputData):
        """
        Method for feature generation for all series
        """
        self.num_classes = input_data.num_classes
        self.task_type = input_data.task
        self._fit_model(input_data)
        # self._save_and_clear_cache()
    def predict_for_fit(self, input_data: CompressionInputData, output_mode: str = 'fedcore'):
        return self.predict(input_data, output_mode)
        # return self.model_after if output_mode == 'fedcore' else self.trainer.predict(input_data, output_mode)

    def predict(
            self, input_data: CompressionInputData, output_mode: str = "fedcore"
    ) -> torch.nn.Module:
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)

    def estimate_params(self, example_batch, model_before, model_after):
        # in future we don't want to store both models simult.
        # base_macs, base_nparams = tp.utils.count_ops_and_params(model_before, example_batch)
        
        is_huggingface = (
            hasattr(model_before, 'config') and 
            hasattr(model_before.config, 'model_type') and
            hasattr(model_before, 'base_model')  
        )
        
        if is_huggingface:
            base_nparams = model_before.num_parameters()  
            nparams = model_after.num_parameters()
            
            base_macs, macs = 0, 0  
        else:
            base_info = summary(model=model_before, input_data=example_batch.to(extract_device(model_before)), verbose=0)
            base_macs, base_nparams = base_info.total_mult_adds, base_info.total_params

            info = summary(model=model_after, input_data=example_batch.to(extract_device(model_after)), verbose=0)
            macs, nparams = info.total_mult_adds, info.total_params
        
        return dict(params_before=base_nparams, macs_before=base_macs,
                    params_after=nparams, macs_after=macs)
        
    # don't del its for New Year
    def _estimate_params(self, model, example_batch):
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_batch)
        return base_macs, base_nparams
    
    # don't del its for New Year
    def _diagnose(self, model, example_batch, *previos_results, annotation=''):
        logging.info(annotation)
        base_macs, base_nparams, *_ = previos_results
        macs, nparams = self._estimate_params(model, example_batch.to(extract_device(model)))
        logging.info("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        logging.info("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))