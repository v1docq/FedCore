import os
from typing import Any, Optional, Union

import numpy as np
import torch
import torch_pruning as tp
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.comptutaional.devices import default_device, extract_device
from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from torchinfo import summary
from fedcore.tools.model_registry import ModelRegistry


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
        # self.epochs = params.get("epochs", 10)
        self.batch_size = params.get("batch_size", 16)
        self.activation = params.get("activation", "ReLU")
        # self.learning_rate = 0.001
        self.model = None
        self.model_for_inference = None
        # self.optimizer = None
        self.params = params
        self._fedcore_id = params.get("fedcore_id")

    def _save_and_clear_cache(self):
        try:
            # the pruned model
            state_dict = tp.state_dict(self.model)
            torch.save(state_dict, "pruned.pth")
            new_model = self.model.eval()
            # load the pruned state_dict into the unpruned model.
            loaded_state_dict = torch.load("pruned.pth", map_location="cpu")
            tp.load_state_dict(new_model, state_dict=loaded_state_dict)
        except Exception:
            self.model.zero_grad()  # Remove gradients
            prefix = f"model_{self.__repr__()}_activation_{self.activation}_epochs_{self.epochs}_bs_{self.batch_size}.pt"
            torch.save(self.model.state_dict(), prefix)
            del self.model
            with torch.no_grad():
                torch.cuda.empty_cache()
            self.model = self.model_for_inference.model.to(torch.device("cpu"))
            self.model.load_state_dict(
                torch.load(prefix, map_location=torch.device("cpu"))
            )
            os.remove(prefix)

    def _init_model(self, input_data, additional_hooks=tuple()):
        model = input_data.target
        # Support passing a filesystem path to a checkpoint/model at the node input
        if isinstance(model, str):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded = torch.load(model, map_location=device)
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
            else:
                model = loaded
        
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected model to be either file path or torch.nn.Module, got {type(model)}")

        self.model_before = model
        is_forecaster = input_data.task.task_type.value.__contains__('forecasting')
        if is_forecaster:
            self.trainer = BaseNeuralForecaster(self.params)
        else:
            self.trainer = BaseNeuralModel(self.params)
        self.trainer.register_additional_hooks(additional_hooks)
        self.trainer.model = model

        # Initialize and register the model in a process-wide registry (single instance per FedCore)
        registry = ModelRegistry().get_instance(model=model)
        # If a string path was provided originally, persist that; otherwise serialize current model
        orig_target = input_data.target
        if isinstance(orig_target, str):
            self._model_id = ModelRegistry.register_model(
                fedcore_id=self._fedcore_id,
                model_path=orig_target,
                metrics={}
            )
        else:
            self._model_id = ModelRegistry.register_model(
                fedcore_id=self._fedcore_id,
                model=model,
                metrics={}
            )

        return model

    def _fit_model(self, ts: CompressionInputData, split_data: bool = False):
        pass

    def _predict_model(self, x_test, output_mode: str = "default"):
        pass

    def _get_example_input(self, input_data: InputData):
        b = next(iter(input_data.features.val_dataloader))
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
        self.target = input_data.target
        self.task_type = input_data.task
        self._fit_model(input_data)
        self._save_and_clear_cache()

    def predict_for_fit(self, input_data: CompressionInputData, output_mode: str = 'fedcore'):
        return self.model_after if output_mode == 'fedcore' else self.model_before

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
        print(annotation)
        base_macs, base_nparams, *_ = previos_results
        macs, nparams = self._estimate_params(model, example_batch.to(extract_device(model)))
        print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))
