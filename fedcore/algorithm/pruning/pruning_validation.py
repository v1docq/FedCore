import torch
from torch import nn
from torchvision.models import VisionTransformer

from fedcore.repository.constanst_repository import PRUNING_FUNC


class PruningValidator:
    def __init__(self, model, n_classes):
        self.model = model
        self.num_classes = n_classes

    def validate_pruned_layers(self, layer, pruning_fn):
        is_linear_layer = isinstance(layer, torch.nn.Linear)
        prune_fc_out = isinstance(pruning_fn, type(PRUNING_FUNC["linear_out"]))
        if is_linear_layer:
            is_fc_layer = layer.out_features == self.num_classes
        else:
            is_fc_layer = False
        return True if not all([is_linear_layer, is_fc_layer, prune_fc_out]) else False

    def filter_ignored_layers(self, model, model_name):
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == self.num_classes:
                ignored_layers.append(m)
            # elif isinstance(m, FrozenBatchNorm2d):
            #     ignored_layers.append(m)
        if model_name.__contains__("ssd"):
            ignored_layers.append(model.head)
        if model_name.__contains__("raft_larget"):
            ignored_layers.extend(
                [model.corr_block, model.update_block, model.mask_predictor]
            )
        if model_name.__contains__("faster_rcnn"):
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
        if model_name.__contains__("retinanet_resnet50_fpn_v2"):
            ignored_layers.extend(
                [
                    model.head.classification_head.cls_logits,
                    model.head.regression_head.bbox_reg,
                ]
            )
            # For ViT: Rounding the number of channels to the nearest multiple of num_heads
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
                if isinstance(m, nn.MultiheadAttention):
                    channel_groups[m] = m.num_heads
        return channel_groups
