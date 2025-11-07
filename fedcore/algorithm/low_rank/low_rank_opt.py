from torch import nn
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module_in_place
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import (
    DECOMPOSE_MODE,
    LRHooks
)
from fedcore.algorithm.base_compression_model import BaseCompressionModel


class LowRankModel(BaseCompressionModel):
    """Singular value decomposition for model structure optimization.

    Args:
    """
    DEFAULT_HOOKS: list[type[BaseHook]] = [prop.value for prop in LRHooks]

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE) 
        self.decomposer = params.get('decomposer', 'svd')
        self.compose_mode = params.get("compose_mode", None)

    def _init_trainer_model_before_model_after_and_incapsulate_hooks(self, input_data):
        additional_hooks = BaseNeuralModel.filter_hooks_by_params(self.params, self.DEFAULT_HOOKS)
        additional_hooks = [hook_type() for hook_type in additional_hooks]
        super()._init_trainer_model_before_model_after(input_data, additional_hooks)
        
        decompose_module_in_place(
            self.model_after, self.decomposing_mode, self.decomposer, self.compose_mode
        )

    def fit(self, input_data) -> None:
        """Run model training with optimization.

        Args:
            input_data: An instance of the model class
        """
        super()._prepare_trainer_and_model_to_fit(input_data)
        # base_params = self._estimate_params(self.model_before, example_batch)
        self.model_after = self.trainer.fit(input_data)
        # self.compress(self.model_after)
        # check params
        example_batch = self._get_example_input(input_data)#.to(extract_device(self.model_before))
        self.estimate_params(example_batch, self.model_before, self.model_after)
        self.model_after._structure_changed__ = True
        return self.model_after

    def compress(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, IDecomposed):
                # module.inference_mode = True
                module.compose_weight_for_inference()

    def load_model(self, model, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        load_svd_state_dict(
            model=model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            compose_mode=self.compose_mode,
        )
        model.to(self.device)
