"""Validation utilities for pruning and channel grouping.

This module defines :class:`PruningValidator`, which provides helper methods to:

* fix shape mismatches in pruned layers (e.g. BatchNorm, Conv, Linear);
* determine which layers should be ignored during pruning for different
  model families (classification, detection, time series, etc.);
* compute per-module channel groups for attention-based architectures
  (e.g. Vision Transformer) to be used by Torch-Pruning.
"""

import torch
from torch import nn
from torchvision.models import VisionTransformer

from fedcore.models.network_modules.layers.attention_layers import MultiHeadAttention


class PruningValidator:
    """Utility class for validating and configuring models for pruning.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be validated and analyzed.
    input_dim : int
        Input dimensionality used to detect input layers that should be
        excluded from pruning (e.g. first Conv/Linear layer).
    output_dim : int
        Output dimensionality used to detect final layers that should be
        excluded from pruning (e.g. last Linear classifier/regressor).
    """

    def __init__(self, model, input_dim, output_dim):
        self.model = model
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.specified_models = [
            "ssd",
            "raft_larget",
            "retinanet_resnet50_fpn_v2",
            "object_detection",
            "chronos",
            "fcos_resnet50_fpn",
            "keypointrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn_v2",
        ]

    @staticmethod
    def validate_pruned_layers(pruned_model):
        """Align parameter shapes of pruned layers.

        After structured pruning, some layer attributes (e.g. BatchNorm
        running statistics, Linear/Conv weight shapes) can become inconsistent
        with their corresponding metadata (``num_features``, ``in_channels``,
        ``in_features``). This method checks common layer types and trims
        parameters to match the expected dimensions.

        Parameters
        ----------
        pruned_model : torch.nn.Module
            Model that has been pruned.

        Returns
        -------
        torch.nn.Module
            The same model instance with adjusted parameter shapes.
        """
        """Make layer weight shapes aligment
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
        """Extend the ignored-layers list for specific detection/TS models.

        This helper adds model-specific heads and submodules that should not
        be pruned (e.g. detection heads, FPNs, ROI heads). The selection is
        based on substrings in ``model_name``.

        Parameters
        ----------
        model_name : str
            Name of the model class or identifier.
        model : torch.nn.Module
            Model instance whose submodules may be excluded from pruning.
        ignored_layers : list
            Current list of layers to ignore; extended in-place.

        Returns
        -------
        list
            Updated list of ignored layers.
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
        """Select layers to ignore for TimeSeriesTransformer-like models.

        For time series transformers, this method ignores Linear layers
        that directly operate on input or output dimensions, so that the
        main embedding and output projection layers are not pruned.

        Parameters
        ----------
        model : torch.nn.Module
            Time series transformer model.

        Returns
        -------
        list
            List of layers to be excluded from pruning.
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
        """Compute a list of layers that should be excluded from pruning.

        Generic rules:

        * final Linear layer with ``out_features == output_dim`` is ignored;
        * first Conv1d layer with ``in_channels == input_dim`` is ignored.

        For time series transformers (model name contains
        ``"TimeSeriesTransformer"``), :meth:`filter_layers_for_tst` is used
        instead. For certain detection/segmentation models, additional
        exclusions are applied via :meth:`_filter_specified_layers`.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose layers are analyzed.
        model_name : str
            Name of the model class or identifier.

        Returns
        -------
        list
            List of layers to be passed as ``ignored_layers`` to pruners.
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
        """Adjust attention head sizes after channel pruning.

        Some attention implementations keep cached attributes such as
        ``attention_head_size`` and ``all_head_size`` that can become
        inconsistent if hidden dimensions are pruned. This method divides
        these attributes by 4 (following the current pruning scheme) to
        keep them in sync with pruned weights.
        """
        for name, m in self.model.named_modules():
            if hasattr(m, "attention_head_size"):
                m.attention_head_size = m.attention_head_size // 4
            if hasattr(m, "all_head_size"):
                m.all_head_size = m.all_head_size // 4

    def validate_channel_groups(self):
        """Build channel-group mapping for attention modules.

        For Vision Transformer-like models, this method returns a dictionary
        that maps attention modules to the number of heads, which can be used
        as ``channel_groups`` in Torch-Pruning.

        Returns
        -------
        dict
            Mapping from attention modules to their number of heads. For
            non-VisionTransformer models, an empty dictionary is returned.
        """
        channel_groups = {}
        if isinstance(self.model, VisionTransformer):
            for m in self.model.modules():
                if isinstance(m, nn.MultiheadAttention) or isinstance(m, MultiHeadAttention):
                    channel_groups[m] = m.num_heads
        return channel_groups
