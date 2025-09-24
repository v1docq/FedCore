import torch
from torch import nn
from torchvision.models import VisionTransformer

from fedcore.repository.constant_repository import PRUNING_FUNC, PRUNING_LAYER_TYPE
from fedcore.models.network_modules.layers.attention_layers import MultiHeadAttention


class PruningValidator:
    def __init__(self, model, input_dim, output_dim):
        self.model = model
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.specified_models = ["ssd", "raft_larget", "retinanet_resnet50_fpn_v2",
                                 "object_detection", "chronos", "fcos_resnet50_fpn", "keypointrcnn_resnet50_fpn",
                                 "maskrcnn_resnet50_fpn_v2"]

    def validate_pruned_layers(self, pruned_model):
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
        ignored_layers = []
        list_of_layers = list(model.modules())
        for layer in list_of_layers:
            is_linear_layer = isinstance(layer, torch.nn.Linear)
            if is_linear_layer:
                if any([layer.in_features == self.input_dim, layer.out_features == self.output_dim]):
                    ignored_layers.append(layer)
        return ignored_layers

    def filter_ignored_layers(self, model, model_name):
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
        for name, m in self.model.named_modules():
            if hasattr(m, "attention_head_size"):
                m.attention_head_size = m.attention_head_size // 4
            if hasattr(m, "all_head_size"):
                m.all_head_size = m.all_head_size // 4

    def validate_channel_groups(self):
        channel_groups = {}
        if isinstance(self.model, VisionTransformer):
            for m in self.model.modules():
                if isinstance(m, nn.MultiheadAttention) or isinstance(m, MultiHeadAttention):
                    channel_groups[m] = m.num_heads
        return channel_groups
