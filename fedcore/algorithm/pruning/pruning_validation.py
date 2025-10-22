"""
Validation helpers for pruning.

This module provides a small utility class, :class:`PruningValidator`, that
performs sanity checks and fixes after pruning, and prepares metadata used by
pruning backends:

- **Shape fixes after pruning**:
  Some PyTorch layers (e.g., BatchNorm1d, Conv1d, Linear) may become
  inconsistent after channel/neuron removal. ``validate_pruned_layers`` trims
  their parameter tensors to match the new declared sizes.

- **Ignore lists**:
  ``filter_ignored_layers`` selects layers that must not be pruned (e.g., final
  classifier / first input layer) and applies additional model-specific rules.

- **Channel groups**:
  ``validate_channel_groups`` discovers attention modules to supply head counts
  for structured/channel-group pruning.

The class is intentionally conservative: it only performs minimal, safe
adjustments and leaves the actual pruning/fine-tuning logic to higher-level
components.
"""

import torch
from torch import nn
from torchvision.models import VisionTransformer

from fedcore.repository.constanst_repository import PRUNING_FUNC, PRUNING_LAYER_TYPE
from fedcore.models.network_modules.layers.attention_layers import MultiHeadAttention


class PruningValidator:
    """
    Utilities to validate and post-process models around pruning.

    Parameters
    ----------
    model : nn.Module
        The target model (typically a deep-copied working copy).
    input_dim : int
        Input channel/feature dimensionality used to protect first layers.
    output_dim : int
        Output dimensionality (e.g., number of classes) used to protect the head.

    Notes
    -----
    ``specified_models`` collects model identifiers for which additional ignore
    rules are applied in :meth:`_filter_specified_layers`.
    """
    def __init__(self, model, input_dim, output_dim):
        self.model = model
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.specified_models = ["ssd", "raft_larget", "retinanet_resnet50_fpn_v2",
                                 "object_detection", "chronos", "fcos_resnet50_fpn", "keypointrcnn_resnet50_fpn",
                                 "maskrcnn_resnet50_fpn_v2"]

    def validate_pruned_layers(self, pruned_model):
        """
        Trim parameter tensors to match layer metadata after pruning.

        BatchNorm1d:
            - Cuts ``bias``, ``weight``, ``running_mean``, ``running_var`` to
              ``num_features`` if needed.

        Conv1d:
            - Trims the **input** channel dimension of ``weight`` to
              ``in_channels`` (keeps all output channels and kernel size).

        Linear:
            - Trims the **input** feature dimension of ``weight`` to
              ``in_features`` (keeps all output features).

        Parameters
        ----------
        pruned_model : nn.Module
            The model after pruning masks have been applied/removed.

        Returns
        -------
        nn.Module
            The same instance with tensors resized for consistency.
        """
        list_of_layers = list(pruned_model.modules())
        for layer in list_of_layers:
            if isinstance(layer, torch.nn.BatchNorm1d):
                current_layer_shape = layer.bias.shape[0]
                shape_after_pruning = layer.num_features
                if current_layer_shape != shape_after_pruning:
                    layer.bias = torch.nn.Parameter(layer.bias[:shape_after_pruning])
                    layer.running_var = torch.Tensor(layer.running_var[:shape_after_pruning])
                    layer.running_mean = torch.Tensor(layer.running_mean[:shape_after_pruning])
                    layer.weight = torch.nn.Parameter(layer.weight[:shape_after_pruning])
            if isinstance(layer, torch.nn.Conv1d):
                current_layer_shape = layer.weight.shape[1]
                shape_after_pruning = layer.in_channels
                if current_layer_shape != shape_after_pruning:
                    layer.weight = torch.nn.Parameter(layer.weight[:, :shape_after_pruning, :])
            if isinstance(layer, torch.nn.Linear):
                current_layer_shape = layer.weight.shape[1]
                shape_after_pruning = layer.in_features
                if current_layer_shape != shape_after_pruning:
                    layer.weight = torch.nn.Parameter(layer.weight[:, :shape_after_pruning])
        return pruned_model

    def _filter_specified_layers(self, model_name, model, ignored_layers):
        """
        Apply model-specific ignore rules for detection/segmentation/time-series.

        The rules are keyed by substrings in ``model_name`` and append modules
        that should not be pruned (heads, ROI components, FPN, etc.).

        Parameters
        ----------
        model_name : str
            String identifier of the architecture/class.
        model : nn.Module
            Model instance (used to access nested modules).
        ignored_layers : list
            Existing list to be extended with additional modules.

        Returns
        -------
        list
            The updated ``ignored_layers`` list.
        """
        if model_name.__contains__("ssd"):
            ignored_layers.append(model.head)
        if model_name.__contains__("raft_larget"):
            ignored_layers.extend(
                [model.corr_block, model.update_block, model.mask_predictor]
            )
        if model_name.__contains__("object_detection"):
            ignored_layers.extend(
                [
                    model.rpn.head.cls_logits,
                    model.rpn.head.bbox_pred,
                    model.backbone.fpn,
                    model.roi_heads,
                ]
            )
        if model_name.__contains__("chronos"):
            ignored_layers.extend(
                [model.model.model.encoder, model.model.model.decoder]
            )
        if model_name.__contains__("fcos_resnet50_fpn"):
            ignored_layers.extend(
                [
                    model.head.classification_head.cls_logits,
                    model.head.regression_head.bbox_reg,
                    model.head.regression_head.bbox_ctrness,
                ]
            )
        if model_name.__contains__("keypointrcnn_resnet50_fpn"):
            ignored_layers.extend(
                [
                    model.rpn.head.cls_logits,
                    model.backbone.fpn.layer_blocks,
                    model.rpn.head.bbox_pred,
                    model.roi_heads.box_head,
                    model.roi_heads.box_predictor,
                    model.roi_heads.keypoint_predictor,
                ]
            )
        if model_name.__contains__("maskrcnn_resnet50_fpn_v2"):
            ignored_layers.extend(
                [
                    model.rpn.head.cls_logits,
                    model.rpn.head.bbox_pred,
                    model.roi_heads.box_predictor,
                    model.roi_heads.mask_predictor,
                ]
            )
        return ignored_layers

    def filter_layers_for_tst(self, model):
        """
        Build ignore list tailored for Time Series Transformer (TST).

        Protects Linear layers that directly interface with the model's input
        and output spaces (i.e., layers whose ``in_features`` equal ``input_dim``
        or ``out_features`` equal ``output_dim``).

        Parameters
        ----------
        model : nn.Module
            Model to scan.

        Returns
        -------
        list[nn.Module]
            Linear layers that must not be pruned.
        """
        ignored_layers = []
        list_of_layers = list(model.modules())
        for layer in list_of_layers:
            is_linear_layer = isinstance(layer, torch.nn.Linear)
            if is_linear_layer:
                if any([layer.in_features == self.input_dim, layer.out_features == self.output_dim]):
                    ignored_layers.append(layer)
        return ignored_layers

    def filter_ignored_layers(self, model, model_name):
        """
        Construct a list of layers that should be excluded from pruning.

        General rules
        -------------
        - Do not prune the final classification/regression head
          (``Linear`` with ``out_features == output_dim``).
        - Do not prune the very first ``Conv1d`` layer if it matches
          ``in_channels == input_dim``.

        Special cases
        -------------
        - If the model name contains ``'TimeSeriesTransformer'``, delegate to
          :meth:`filter_layers_for_tst`.
        - For certain detection/segmentation/time-series models identified by
          substrings in :attr:`specified_models`, call
          :meth:`_filter_specified_layers`.

        Parameters
        ----------
        model : nn.Module
            Model to analyze.
        model_name : str
            Human-readable identifier of the model class.

        Returns
        -------
        list[nn.Module]
            Modules that should be ignored by the pruner.
        """
        ignored_layers = []
        model_layers = list(model.modules())
        if model_name.__contains__('TimeSeriesTransformer'):
            return self.filter_layers_for_tst(model)
        else:
            for layer in model_layers:
                # dont prune final fc layer
                if isinstance(layer, torch.nn.Linear) and layer.out_features == self.output_dim:
                    ignored_layers.append(layer)
                # dont prune input conv layer
                elif isinstance(layer, torch.nn.Conv1d):
                    if layer.in_channels == self.input_dim:
                        ignored_layers.append(layer)
                # elif isinstance(layer, torch.nn.BatchNorm1d):
                #     ignored_layers.append(layer)
            if model_name in self.specified_models:
                self._filter_specified_layers(model=model, model_name=model_name, ignored_layers=ignored_layers)

        return ignored_layers

    def fix_attention_layer(self):
        """
        Adjust attention head sizes after heavy pruning (heuristic).

        For modules exposing attributes ``attention_head_size`` / ``all_head_size``,
        reduce them by a factor of 4. This is a coarse fix intended for cases
        where attention dimensions were pruned significantly and need a quick
        alignment. Use with care—model-specific review is recommended.
        """
        for name, m in self.model.named_modules():
            if hasattr(m, "attention_head_size"):
                m.attention_head_size = m.attention_head_size // 4
            if hasattr(m, "all_head_size"):
                m.all_head_size = m.all_head_size // 4

    def validate_channel_groups(self):
        """
        Discover attention modules and return their head counts for grouping.

        For :class:`torchvision.models.VisionTransformer` models, maps every
        :class:`nn.MultiheadAttention` (or custom
        :class:`fedcore.models.network_modules.layers.attention_layers.MultiHeadAttention`)
        submodule to its number of heads. This is used by structured pruning
        to keep channels grouped per attention head.

        Returns
        -------
        dict[nn.Module, int]
            Mapping ``{attention_module: num_heads}``. Empty for non-ViT models.
        """
        channel_groups = {}
        if isinstance(self.model, VisionTransformer):
            for m in self.model.modules():
                if isinstance(m, nn.MultiheadAttention) or isinstance(m, MultiHeadAttention):
                    channel_groups[m] = m.num_heads
        return channel_groups
