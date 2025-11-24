"""Training hooks for model quantization.

This module defines several quantization hooks that can be attached to
:class:`BaseNeuralModel`:

* :class:`DynamicQuantizationHook` – post-training dynamic quantization (PTQ)
  using :func:`torch.ao.quantization.quantize_dynamic`.
* :class:`StaticQuantizationHook` – post-training static quantization with
  calibration on a validation loader.
* :class:`QATHook` – quantization-aware training (QAT) hook that prepares
  the model for QAT and later converts it to a quantized version.

All hooks inherit from :class:`AbstractQuantizationHook`, which implements
common scheduling and backend configuration.
"""

from abc import abstractmethod
import torch
from torch import nn

from fedcore.models.network_impl.hooks import BaseHook
from torch.ao.quantization import (quantize_dynamic)


class AbstractQuantizationHook(BaseHook):
    """Base class for quantization hooks.

    This abstract hook configures common quantization settings and provides a
    unified :meth:`action` implementation that:

    * sets the global quantized engine backend;
    * delegates to a subclass-specific :meth:`_child_action`;
    * marks the model as quantized via ``model._is_quantized = True``.

    Subclasses implement different quantization strategies (dynamic, static,
    QAT) by overriding :meth:`_child_action`.

    Parameters
    ----------
    quant_each : int
        Epoch interval at which the quantization hook should be fired.
        Passed to :meth:`BaseHook.is_epoch_arrived_default`. For QAT this
        may also be used as final conversion epoch.
    dtype :
        Target quantization dtype (e.g. ``torch.qint8``) used by
        :func:`torch.ao.quantization.quantize_dynamic` or other quant APIs.
    allowed_quant_module_mapping : set[nn.Module]
        Collection of module types that are allowed to be quantized (used as
        ``qconfig_spec`` for dynamic quantization).
    prepare_qat_after_epoch : int
        Epoch at which QAT preparation (``prepare_qat``) should be applied.
        Used only by :class:`QATHook`.
    quantized_engine_backend : str
        Name of the quantized engine backend
        (e.g. ``"fbgemm"`` or ``"qnnpack"``) assigned to
        ``torch.backends.quantized.engine`` before quantization.
    """
    _QUANT_TYPE = 'AbstractQuantizationHook'

    def __init__(self, 
                 quant_each: int, 
                 dtype, 
                 allowed_quant_module_mapping: set[nn.Module], 
                 prepare_qat_after_epoch: int,
                 quantized_engine_backend: str
        ):
        super().__init__()
        self.dtype = dtype
        self.quant_each = quant_each
        self.allowed_quant_module_mapping = allowed_quant_module_mapping
        self.prepare_qat_after_epoch = prepare_qat_after_epoch
        self.quantized_engine_backend = quantized_engine_backend

    @classmethod
    def check_init(cls, d: dict):
        """Check whether this hook type should be instantiated.

        Parameters
        ----------
        d : dict
            Hook configuration dictionary, expected to contain
            ``"quant_type"`` key.

        Returns
        -------
        bool
            ``True`` if the base :class:`BaseHook` conditions are satisfied
            and ``d["quant_type"]`` matches ``cls._QUANT_TYPE``, otherwise
            ``False``.
        """
        if (not super().check_init(d)):
            return False
        if d["quant_type"] != cls._QUANT_TYPE:
            return False
        return True
    
    @abstractmethod
    def _child_action(self, epoch, kws):
        """Subclass-specific quantization logic.

        This method must be implemented by concrete quantization hooks
        (:class:`DynamicQuantizationHook`, :class:`StaticQuantizationHook`,
        :class:`QATHook`). It is called from :meth:`action` after the
        quantized backend is set.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Arbitrary keyword arguments passed from the trainer (e.g.
            loaders, extra metadata).
        """
        pass
    
    def action(self, epoch, kws):
        """Perform quantization step and mark the model as quantized.

        This method sets ``torch.backends.quantized.engine`` to the configured
        backend, delegates to :meth:`_child_action`, and then flags the model
        with ``_is_quantized = True``.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer, forwarded to
            :meth:`_child_action`.
        """
        torch.backends.quantized.engine = self.quantized_engine_backend
        self._child_action(epoch, kws)
        self.model._is_quantized = True


class DynamicQuantizationHook(AbstractQuantizationHook):
    """Hook for dynamic post-training quantization (PTQ).

    At the scheduled epochs, this hook calls
    :func:`torch.ao.quantization.quantize_dynamic` on the model using the
    provided module mapping and dtype. The model is temporarily put into
    eval mode during quantization and then switched back to train mode.

    Quantization type identifier: ``"dynamic"``.
    """
    _SUMMON_KEY = 'quant_type'
    HOOK_PLACE = 40
    _QUANT_TYPE = 'dynamic'

    def trigger(self, epoch, kws) -> bool:
        """Check whether dynamic quantization should run at this epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Returns
        -------
        bool
            ``True`` if ``epoch`` matches the schedule defined by
            ``self.quant_each``, otherwise ``False``.
        """
        return self.is_epoch_arrived_default(epoch, self.quant_each)

    def _child_action(self, epoch, kws):
        """Apply dynamic quantization to the whole model.

        Parameters
        ----------
        epoch : int
            Current epoch number (unused).
        kws : dict
            Extra arguments from the trainer (unused).

        Notes
        -----
        The model is quantized in-place using ``quantize_dynamic``, with
        ``qconfig_spec`` taken from ``self.allowed_quant_module_mapping``.
        """
        print("[HOOK] Performing Dynamic PTQ hook operations.")
        self.model.eval()
        quantize_dynamic(
                    self.model,
                    qconfig_spec=self.allowed_quant_module_mapping,
                    dtype=self.dtype,
                    inplace=True)
        self.model.train()
        print("[HOOK] Dynamic PTQ setup completed.")


class StaticQuantizationHook(AbstractQuantizationHook):
    """Hook for static post-training quantization with calibration.

    At the scheduled epochs, this hook:

    1. Puts the model into eval mode.
    2. Calls :func:`torch.quantization.prepare` to insert observers.
    3. Runs a calibration pass over ``kws["val_loader"]`` under
       ``torch.no_grad()``.
    4. Calls :func:`torch.quantization.convert` to produce the quantized
       model.
    5. Switches the model back to train mode.

    Quantization type identifier: ``"static"``.
    """
    _SUMMON_KEY = 'quant_type'
    HOOK_PLACE = 40
    _QUANT_TYPE = 'static'

    def trigger(self, epoch, kws):
        """Check whether static quantization should run at this epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Returns
        -------
        bool
            ``True`` if ``epoch`` matches the schedule defined by
            ``self.quant_each``, otherwise ``False``.
        """
        return self.is_epoch_arrived_default(epoch, self.quant_each)

    def _child_action(self, epoch, kws):
        """Run prepare–calibrate–convert static PTQ pipeline.

        Parameters
        ----------
        epoch : int
            Current epoch number (unused).
        kws : dict
            Dictionary expected to contain ``"val_loader"`` with a validation
            dataloader for calibration.
        """
        self.model.eval()
        torch.quantization.prepare(self.model, inplace=True)
        
        with torch.no_grad():
            for data, _ in kws['val_loader']:
                self.model(data.to(self.hookable_trainer.device))
        
        torch.quantization.convert(self.model, inplace=True)
        self.model.train()


class QATHook(AbstractQuantizationHook):
    """Hook for quantization-aware training (QAT).

    This hook has two distinct phases controlled by epochs:

    * At ``epoch == prepare_qat_after_epoch`` it prepares the model for QAT
      by calling :func:`torch.quantization.prepare_qat` in-place.
    * At ``epoch == quant_each`` (or at the last epoch if ``quant_each == -1``),
      it converts the QAT-prepared model into a quantized model via
      :func:`torch.quantization.convert`.

    Between these two phases, the trainer can perform regular training steps
    on the QAT-prepared model.

    Quantization type identifier: ``"qat"``.
    """
    _SUMMON_KEY = 'quant_type'
    HOOK_PLACE = 40
    _QUANT_TYPE = 'qat'
        
    def trigger(self, epoch, kws) -> bool:
        """Determine whether to prepare QAT or convert at the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Returns
        -------
        bool
            ``True`` if either QAT preparation or final conversion should
            occur at this epoch, otherwise ``False``.
        """
        if (epoch == self.prepare_qat_after_epoch):
            return True
        elif (epoch == self.quant_each or (self.quant_each == -1 and epoch == self.hookable_trainer.epochs)):
            return True
        return False

    def _child_action(self, epoch, kws):
        """Run QAT preparation or final conversion depending on the epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        kws : dict
            Extra arguments from the trainer (unused).

        Notes
        -----
        * At ``epoch == prepare_qat_after_epoch``:
          :func:`torch.quantization.prepare_qat` is called in-place.
        * At ``epoch == quant_each`` or if ``quant_each == -1`` and this is
          the last epoch:
          the model is converted to a quantized version via
          :func:`torch.quantization.convert`.
        """
        if (epoch == self.prepare_qat_after_epoch):
            torch.quantization.prepare_qat(self.model, inplace=True)
        elif (epoch == self.quant_each or (self.quant_each == -1 and epoch == self.hookable_trainer.epochs)):
            self.model.eval()
            torch.quantization.convert(self.model, inplace=True)
            self.model.train()
