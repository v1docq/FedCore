"""Quantization model for FedCore based on the standard PyTorch approach.

This module defines :class:`BaseQuantizer`, a compression node that applies
PyTorch quantization to a neural network:

* **dynamic** – post-training dynamic quantization via
  :func:`torch.ao.quantization.quantize_dynamic` (weights only);
* **static** – post-training static quantization via FX graph-mode
  ``prepare_fx`` → calibration on the validation loader → ``convert_fx``;
* **qat** – quantization-aware training via ``prepare_qat_fx``, a short
  fine-tuning loop and final ``convert_fx``.

PTQ runs on CPU because the supported quantization backends (``fbgemm``,
``qnnpack``) are CPU-only. The original (floating-point) model is never
mutated: quantization is applied to a deep copy, so the pipeline can still
evaluate the ``original`` regime after ``fit``.

The class exposes the standard FedCore ``fit / predict / predict_for_fit``
interface and therefore can be used directly as a FedCore pipeline node
(``quantization_model``).
"""

import logging
from copy import deepcopy

import torch
from torch import nn
from torch.ao.quantization import (
    quantize_dynamic,
    get_default_qconfig,
    get_default_qat_qconfig,
    default_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantization_mappings import (
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_QAT_MODULE_MAPPINGS,
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    prepare_qat_fx,
    convert_fx,
)

from fedot.core.data.data import InputData

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.models.network_impl.utils.trainer_factory import create_trainer_from_input_data
from fedcore.repository.constant_repository import TorchLossesConstant
from fedcore.models.network_impl.utils.hooks import Optimizers
from fedcore.tools.registry.model_registry import ModelRegistry

__all__ = ['BaseQuantizer']


class BaseQuantizer(BaseCompressionModel):
    """Base class for model quantization in FedCore.

    Wraps a neural model and applies one of the standard PyTorch
    quantization workflows: post-training dynamic/static quantization or
    quantization-aware training (QAT).

    Parameters
    ----------
    params : dict, optional
        Configuration dictionary. Common keys include:

        * ``"quant_type"`` – quantization mode: ``"dynamic"`` (default),
          ``"static"`` or ``"qat"``;
        * ``"backend"`` – quantization backend (``"fbgemm"`` by default,
          ``"qnnpack"`` on ARM);
        * ``"dtype"`` – target quantization dtype for dynamic quantization
          (:data:`torch.qint8` by default, :data:`torch.float16` is also
          supported for dynamic quantization on CPU);
        * ``"allow_emb"`` – whether to quantize embedding layers
          (default ``False``);
        * ``"allow_conv"`` – whether to quantize convolution layers
          (default ``True``);
        * ``"device"`` – device of the original floating-point model;
          quantized models are always produced on CPU;
        * ``"qat_params"`` – dict with ``epochs``, ``optimizer``,
          ``criterion`` and ``lr`` used for QAT fine-tuning;
        * Any additional keys supported by
          :class:`BaseCompressionModel`.
    """

    QUANT_TYPES = ('dynamic', 'static', 'qat')

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Quantization params
        self.quant_type = params.get('quant_type', 'dynamic')
        if self.quant_type not in self.QUANT_TYPES:
            raise ValueError(
                f'Unknown quant_type: {self.quant_type}. '
                f'Supported types: {self.QUANT_TYPES}')
        self.backend = params.get('backend', 'fbgemm')
        self.dtype = params.get('dtype', torch.qint8)
        self.allow_emb = params.get('allow_emb', False)
        self.allow_conv = params.get('allow_conv', True)
        self.inplace = params.get('inplace', False)

        # PyTorch eager/FX quantization backends are CPU-only.
        self._quantize_device = torch.device('cpu')
        torch.backends.quantized.engine = self.backend

        # QAT params
        self.qat_params = dict(params.get('qat_params') or {})
        self.qat_params.setdefault('epochs', 2)
        self.qat_params.setdefault('optimizer', 'adam')
        self.qat_params.setdefault('criterion', 'cross_entropy')
        self.qat_params.setdefault('lr', 0.001)
        self.optimizer = Optimizers[self.qat_params.get('optimizer', 'adam')]
        self.criterion = TorchLossesConstant[self.qat_params.get('criterion', 'cross_entropy')]

        self._quantization_index = 0

        self.logger.info(
            f'[INIT] quant_type: {self.quant_type}, backend: {self.backend}, '
            f'dtype: {self.dtype}, allow_embedding: {self.allow_emb}, '
            f'allow_convolution: {self.allow_conv}')

    def __repr__(self):
        """Return a short string representation with quantization mode."""
        return f'{self.quant_type.upper()} Quantization'

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def _set_allow_mappings(self):
        """Return the set of module types that may be quantized."""
        mapping_dict = {
            'dynamic': DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
            'static': DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
            'qat': DEFAULT_QAT_MODULE_MAPPINGS,
        }[self.quant_type]
        allow = set(mapping_dict.keys())
        if not self.allow_conv:
            allow -= {nn.Conv1d, nn.Conv2d, nn.Conv3d}
        if not self.allow_emb:
            allow -= {nn.Embedding, nn.EmbeddingBag}
        return allow

    def _get_qconfig(self):
        """Return the base :class:`QConfig` for the configured mode."""
        if self.quant_type == 'dynamic':
            if self.dtype == torch.float16:
                return float16_dynamic_qconfig
            return default_dynamic_qconfig
        if self.quant_type == 'static':
            return get_default_qconfig(self.backend)
        return get_default_qat_qconfig(self.backend)

    def _build_qconfig_mapping(self) -> QConfigMapping:
        """Build the FX :class:`QConfigMapping` honouring allow-flags."""
        qconfig_mapping = QConfigMapping().set_global(self._get_qconfig())
        if not self.allow_conv:
            for conv_type in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
                qconfig_mapping.set_object_type(conv_type, None)
        if self.allow_emb:
            qconfig_mapping.set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
            qconfig_mapping.set_object_type(nn.EmbeddingBag, float_qparams_weight_only_qconfig)
        return qconfig_mapping

    # ------------------------------------------------------------------ #
    # Model / data initialisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_compression_data(input_data):
        """Normalize supported input containers to CompressionInputData-like object.

        FedCore pipelines pass :class:`CompressionInputData` directly, while
        ``DataCheck`` wraps it into ``InputData(features=<CompressionInputData>,
        target=<model>)``. Both shapes are accepted. The model is taken from
        ``input_data.model`` with a fallback to ``input_data.target``.
        """
        model = getattr(input_data, 'model', None)
        if model is None and isinstance(getattr(input_data, 'target', None), torch.nn.Module):
            model = input_data.target
        if (not hasattr(input_data, 'train_dataloader')
                and hasattr(input_data, 'features')
                and hasattr(input_data.features, 'train_dataloader')):
            input_data = input_data.features
        if model is not None:
            input_data.model = model
        return input_data

    def _get_example_input(self, input_data: InputData) -> torch.Tensor:
        """Extract a single batch of features as example input (on CPU)."""
        loader = input_data.val_dataloader or input_data.train_dataloader
        batch = next(iter(loader))
        example_input = batch[0] if isinstance(batch, (list, tuple)) else batch
        return example_input.to(self._quantize_device)

    @staticmethod
    def _batch_to_device(batch, device):
        """Move the features of a (features, target) batch to ``device``."""
        if isinstance(batch, (list, tuple)):
            return batch[0].to(device)
        if torch.is_tensor(batch):
            return batch.to(device)
        raise TypeError(f'Unsupported batch type: {type(batch)}')

    def _init_model(self, input_data):
        """Resolve the model from input data and create the trainer."""
        model = getattr(input_data, 'model', None)
        if model is None:
            model = input_data.target
        if isinstance(model, str):
            loaded = torch.load(model, map_location=self.device)
            if isinstance(loaded, dict) and 'model' in loaded:
                model = loaded['model']
            else:
                model = loaded

        self.model_before = model

        self.trainer = create_trainer_from_input_data(input_data, self.qat_params)
        self.trainer.model = self.model_before

        self._model_registry = ModelRegistry()
        self._quantization_index += 1
        if self._model_id_before:
            self._model_registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage='before',
                mode=self.__class__.__name__,
            )

        self.logger.info('[MODEL] Model initialized for quantization.')

    # ------------------------------------------------------------------ #
    # Quantization workflows
    # ------------------------------------------------------------------ #
    def _model_copy_for_quantization(self):
        """Return a detached CPU copy of the model ready for quantization."""
        model = deepcopy(self.model_before)
        return model.to(self._quantize_device).eval()

    def _fit_dynamic(self, input_data: InputData) -> nn.Module:
        """Post-training dynamic quantization (weights only, on CPU)."""
        model = self._model_copy_for_quantization()
        quant_model = quantize_dynamic(
            model,
            qconfig_spec=self._set_allow_mappings(),
            dtype=self.dtype,
            inplace=False,
        )
        self.logger.info('[FIT] Dynamic PTQ performed successfully.')
        return quant_model

    def _fit_static(self, input_data: InputData) -> nn.Module:
        """Post-training static quantization via FX graph mode.

        Pipeline: ``prepare_fx`` → calibration on the validation loader →
        ``convert_fx``.
        """
        model = self._model_copy_for_quantization()
        example_inputs = self._get_example_input(input_data)
        qconfig_mapping = self._build_qconfig_mapping()

        prepared = prepare_fx(model, qconfig_mapping, example_inputs)
        self._calibrate(prepared, input_data)
        quant_model = convert_fx(prepared)
        self.logger.info('[FIT] Static PTQ performed successfully.')
        return quant_model

    def _fit_qat(self, input_data: InputData) -> nn.Module:
        """Quantization-aware training via FX graph mode.

        Pipeline: ``prepare_qat_fx`` → short fine-tuning on the train
        loader → ``convert_fx``.
        """
        model = self._model_copy_for_quantization()
        example_inputs = self._get_example_input(input_data)
        qconfig_mapping = self._build_qconfig_mapping()

        prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs)
        self._qat_train_loop(prepared, input_data)
        quant_model = convert_fx(prepared)
        self.logger.info('[FIT] QAT performed successfully.')
        return quant_model

    def _calibrate(self, prepared_model: nn.Module, input_data: InputData):
        """Run a calibration pass over the validation loader."""
        prepared_model.eval()
        loader = input_data.val_dataloader or input_data.train_dataloader
        with torch.no_grad():
            for batch in loader:
                prepared_model(self._batch_to_device(batch, self._quantize_device))

    def _qat_train_loop(self, prepared_model: nn.Module, input_data: InputData):
        """Fine-tune the QAT-prepared model on the train loader."""
        prepared_model.train()
        optimizer = self.optimizer.value(prepared_model.parameters(),
                                         lr=self.qat_params.get('lr', 0.001))
        criterion = self.criterion.value()
        loader = input_data.train_dataloader
        epochs = self.qat_params.get('epochs', 2)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                inputs = self._batch_to_device(batch, self._quantize_device)
                targets = batch[1].to(self._quantize_device) \
                    if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
                optimizer.zero_grad()
                output = prepared_model(inputs)
                if targets is None:
                    continue
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.logger.info(
                f'[QAT] epoch {epoch + 1}/{epochs}, '
                f'train loss: {running_loss / max(len(loader), 1):.4f}')

    # ------------------------------------------------------------------ #
    # FedCore node interface
    # ------------------------------------------------------------------ #
    def fit(self, input_data: InputData):
        """Quantize the model according to the configured mode.

        The original model is left untouched; the quantized copy is stored
        as ``model_after`` (and registered in the model registry).

        If static quantization or QAT fails (e.g. the model cannot be FX
        traced), the method falls back to dynamic quantization so the
        pipeline node stays usable.

        Parameters
        ----------
        input_data : InputData
            FedCore input data with the model and dataloaders.

        Returns
        -------
        torch.nn.Module
            Quantized model stored in ``self.model_after``.
        """
        input_data = self._resolve_compression_data(input_data)
        self._init_model(input_data)
        self.data_batch_for_calib = self._get_example_input(input_data)

        quant_model = None
        try:
            if self.quant_type == 'dynamic':
                quant_model = self._fit_dynamic(input_data)
            elif self.quant_type == 'static':
                quant_model = self._fit_static(input_data)
            else:
                quant_model = self._fit_qat(input_data)
        except Exception:
            self.logger.exception(
                f'[{self.quant_type.upper()}] Quantization failed, '
                f'falling back to dynamic quantization.')
            quant_model = self._fit_dynamic(input_data)

        quant_model.eval()
        quant_model._is_quantized = True
        self.model_after = quant_model

        if self._model_id_after:
            self._model_registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_after,
                metrics={},
                stage='after',
                mode=self.__class__.__name__,
            )

        return self.model_after

    def _inference_device(self, model: nn.Module) -> torch.device:
        """Pick the inference device: quantized models run on CPU only."""
        if getattr(model, '_is_quantized', False):
            return self._quantize_device
        return self.device

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Return predictions of the quantized (or original) model."""
        return self.predict(input_data, output_mode)

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Run prediction with either the quantized or the original model.

        Parameters
        ----------
        input_data : InputData
            Data for inference.
        output_mode : str, optional
            If ``"fedcore"`` (default), use the quantized model
            ``self.model_after``; otherwise use the original model
            ``self.model_before``.

        Returns
        -------
        Any
            Output of the trainer ``predict`` for the selected model.
        """
        model = self.model_after if output_mode == 'fedcore' else self.model_before
        self.trainer.model = model
        self.trainer.device = self._inference_device(model)
        return self.trainer.predict(input_data, output_mode)
