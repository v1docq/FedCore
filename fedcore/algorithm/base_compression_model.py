from copy import deepcopy
import os
from typing import Sequence

import torch
import torch_pruning as tp
from fedot.core.data.data import InputData

from fedcore.architecture.comptutaional.devices import extract_device
from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from torchinfo import summary

from fedcore.models.network_impl.hooks import BaseHook


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

    def __init__(self, params: dict = {}):
        self.batch_size = params.get("batch_size", 16)
        self.activation = params.get("activation", "ReLU")
        self.model = None
        self.model_for_inference = None
        self.params = params

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

    def _init_model_before_model_after(self, input_data):
        self.model_before = input_data.target
        self.model_after = deepcopy(self.model_before)

    def _init_trainer_with_model_after(self, input_data, additional_hooks: Sequence[BaseHook]):
        is_forecaster = input_data.task.task_type.value.__contains__('forecasting')
        if is_forecaster:
            self.trainer = BaseNeuralForecaster(self.model_after, self.params, additional_hooks)
        else:
            self.trainer = BaseNeuralModel(self.model_after, self.params, additional_hooks)

    def _init_trainer_model_before_model_after(self, input_data, additional_hooks: Sequence[BaseHook]):
        self._init_model_before_model_after(input_data)
        self._init_trainer_with_model_after(input_data, additional_hooks)

    def _fit_model(self, ts: CompressionInputData, split_data: bool = False):
        pass

    def _predict_model(self, x_test, output_mode: str = "default"):
        pass

    def _get_example_input(self, input_data: InputData):
        b = next(iter(input_data.features.val_dataloader))
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return b[0]
        return b

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
        base_info = summary(model=model_before, input_data=example_batch.to(extract_device(model_before)), verbose=0)
        base_macs, base_nparams = base_info.total_mult_adds, base_info.total_params

        info = summary(model=model_after, input_data=example_batch.to(extract_device(model_after)), verbose=0)
        macs, nparams = info.total_mult_adds, info.total_params

        print("Params: %.6f M => %.6f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.6f B => %.6f B" % (base_macs / 1e9, macs / 1e9))
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
        print("MACs: %.2f B => %.2f B" % (base_macs / 1e9, macs / 1e9))
