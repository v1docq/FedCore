from torch import nn

from fedcore.algorithm.low_rank.decomposer import DecomposerType, init_decomposer_from_whole_params
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module_in_place
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.models.network_impl.utils.hooks import BaseHook
from fedcore.repository.constant_repository import (
    DECOMPOSE_MODE,
    LRHooks
)
from fedcore.algorithm.base_compression_model import BaseCompressionModel

from fedcore.algorithm.low_rank.reassembly import TransMLA, FlatLLM
from external.transmlacore.modify_config import settings


class LowRankModel(BaseCompressionModel):
    """Compression model that applies low-rank (SVD-based) decomposition.

    The class wraps training of a base neural network and replaces eligible
    layers with low-rank decomposed counterparts. It also registers rank
    pruning hooks defined in :data:`LRHooks` and provides utilities for
    composing decomposed weights and loading decomposed checkpoints.

    Parameters
    ----------
    params : dict, optional
        Configuration dictionary. Common keys include:

        * ``decomposing_mode``: decomposition mode passed to
          :func:`decompose_module_in_place` (defaults to
          :data:`DECOMPOSE_MODE`).
        * ``decomposer``: name of the decomposer to use
          (e.g. ``"svd"``; default is ``"svd"``).
        * ``compose_mode``: compose mode passed to
          :func:`decompose_module_in_place` and
          :func:`load_svd_state_dict` (or ``None`` to use default).
        * Any additional keys supported by :class:`BaseCompressionModel`
          and :class:`BaseNeuralModel` (e.g. optimizer, scheduler, device).
    """
    DEFAULT_HOOKS: list[type[BaseHook]] = [prop.value for prop in LRHooks]

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE) 
        self.decomposer = init_decomposer_from_whole_params(params)
        self.compose_mode = params.get("compose_mode", None)

    def _init_trainer_model_before_model_after_and_incapsulate_hooks(self, input_data):
        """Initialize trainer, models and attach low-rank hooks.

        This method:

        1. Filters available hooks using :meth:`BaseNeuralModel.filter_hooks_by_params`
           and :data:`DEFAULT_HOOKS`.
        2. Instantiates the selected hooks and passes them to
           :meth:`BaseCompressionModel._init_trainer_model_before_model_after`
           to build ``trainer``, ``model_before`` and ``model_after``.
        3. Applies :func:`decompose_module_in_place` to ``self.model_after``
           so that supported layers are replaced with low-rank decomposed
           implementations.

        Parameters
        ----------
        input_data :
            Object used to initialize the underlying trainer/model
            (e.g. experimenter, dataset descriptor or config object).
        """
        additional_hooks = BaseNeuralModel.filter_hooks_by_params(self.params, self.DEFAULT_HOOKS)
        additional_hooks = [hook_type() for hook_type in additional_hooks]
        super()._init_trainer_model_before_model_after(input_data, additional_hooks)
        
        if self._model_id_before: #TODO after merge some context was lost (Model registry PR), check necessity this logic
            self._registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage="before",
                mode=self.__class__.__name__
            )

        decompose_module_in_place(
            self.model_after, self.decomposing_mode, self.decomposer, self.compose_mode
        )

    def fit(self, input_data) -> None:
        """Train the model with low-rank–aware hooks and estimate compression.

        The method prepares the trainer and models, runs training, and then
        evaluates parameter statistics before and after decomposition.

        Parameters
        ----------
        input_data :
            Object required by the trainer to perform fitting
            (e.g. experimenter instance, dataloaders, or config).

        Returns
        -------
        nn.Module
            Trained and decomposed model instance (``self.model_after``) with
            ``_structure_changed__`` flag set to ``True``.
        """
        super()._prepare_trainer_and_model_to_fit(input_data)
        self.model_after = self.trainer.fit(input_data)

        if self._model_id_after: #TODO seems broken after merge, why do we need to set metrics={}?
            self._registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_after,
                metrics={},
                stage="after",
                mode=None,
                trainer=self.trainer
            )

        # check params
        example_batch = self._get_example_input(input_data)  # .to(extract_device(self.model_before))
        self.estimate_params(example_batch, self.model_before, self.model_after)
        self.model_after._structure_changed__ = True
        return self.model_after

    def compress(self, model: nn.Module):
        """Compose weights of all decomposed layers for inference.

        This helper iterates over all modules of the given ``model`` and,
        for each instance of :class:`IDecomposed`, calls
        :meth:`IDecomposed.compose_weight_for_inference` to materialize the
        effective weight matrix.

        Parameters
        ----------
        model : nn.Module
            Model whose decomposed layers should be switched to inference form.
        """
        for module in model.modules():
            if isinstance(module, IDecomposed):
                # module.inference_mode = True
                module.compose_weight_for_inference()

        model_type = getattr(getattr(model, 'config', None), 'model_type', None)

        if model_type in settings:
            from transformers import AutoTokenizer

            model_name = getattr(model.config, "name_or_path", None)
            if model_name is None:
                raise ValueError("Can't find model name in config to load tokenizer")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            trans_mla = TransMLA()
            model = trans_mla.reassemble(model, tokenizer=tokenizer)

        if model_type in ['llama', 'mistral']:
            from transformers import AutoTokenizer

            model_name = getattr(model.config, "name_or_path", None)
            if model_name is None:
                raise ValueError("Can't find model name in config to load tokenizer")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            flat_llm = FlatLLM()
            model = flat_llm.reassemble(model, architecture=model_type, tokenizer=tokenizer)

    def load_model(self, model, state_dict_path: str) -> None:
        """Load a decomposed (SVD-based) checkpoint into a model.

        The method uses :func:`load_svd_state_dict` to restore a state dict
        that already contains low-rank–decomposed parameters, and then moves
        the model to ``self.device``.

        Parameters
        ----------
        model : nn.Module
            Model instance into which the state dict will be loaded.
        state_dict_path : str
            Path to the serialized state dict file containing decomposed
            parameters.
        """
        load_svd_state_dict(
            model=model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            compose_mode=self.compose_mode,
            decomposer=self.decomposer
        )
        model.to(self.device)

    def predict_for_fit(self, input_data, output_mode: str = 'fedcore'):
        """Return model after training."""
        return self.predict(input_data, output_mode)

    def predict(self, input_data, output_mode: str = 'fedcore'):
        """Prediction using compressed model."""
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
