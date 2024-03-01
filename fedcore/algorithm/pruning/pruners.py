from copy import deepcopy

import numpy as np
import torchvision
from fedot.core.data.data import InputData
from torch import nn
from torchvision.models import VisionTransformer

from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.repository.constanst_repository import PRUNERS, PRUNING_IMPORTANCE, PRUNING_LAYERS_IMPL


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get('epochs', 15)
        self.pruner_name = params.get('pruner_name', 'group_norm_pruner')
        self.pruning_ratio = params.get('pruning_ratio', 0.5)
        self.importance = params.get('importance', 'GroupNormImportance')
        self.importance_norm = params.get('importance_norm', 1)
        self.importance_reduction = params.get('importance_reduction', 'mean')
        self.importance_normalize = params.get('importance_normalize', 'mean')
        # importance criterion for parameter selections
        self.importance = PRUNING_IMPORTANCE[self.importance](group_reduction=self.importance_reduction,
                                                              normalizer=self.importance_normalize)
        self.pruner = None

    def __repr__(self):
        return self.pruner_name

    def _filter_ignored_layers(self, model, model_name):
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.num_classes:
                ignored_layers.append(m)
        if model_name.__contains__('ssd'):
            ignored_layers.append(model.head)
        if model_name.__contains__('raft_larget'):
            ignored_layers.extend(
                [model.corr_block, model.update_block, model.mask_predictor]
            )
        if model_name.__contains__('faster_rcnn'):
            ignored_layers.extend([
                model.rpn.head.cls_logits,
                model.rpn.head.bbox_pred,
                model.backbone.fpn,
                model.roi_heads
            ])
        if model_name.__contains__('fcos_resnet50_fpn'):
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg,
                                   model.head.regression_head.bbox_ctrness])
        if model_name.__contains__('keypointrcnn_resnet50_fpn'):
            ignored_layers.extend([model.rpn.head.cls_logits, model.backbone.fpn.layer_blocks, model.rpn.head.bbox_pred,
                                   model.roi_heads.box_head, model.roi_heads.box_predictor,
                                   model.roi_heads.keypoint_predictor])
        if model_name.__contains__('maskrcnn_resnet50_fpn_v2'):
            ignored_layers.extend([model.rpn.head.cls_logits, model.rpn.head.bbox_pred, model.roi_heads.box_predictor,
                                   model.roi_heads.mask_predictor])
        if model_name.__contains__('retinanet_resnet50_fpn_v2'):
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg])
            # For ViT: Rounding the number of channels to the nearest multiple of num_heads
        return ignored_layers

    def fit(self,
            input_data: InputData):
        self.model = input_data.target
        self.model_for_inference = deepcopy(self.model)
        self.num_classes = input_data.features.num_classes
        self.ignored_layers = self._filter_ignored_layers(self.model, str(self.model.__class__))
        self.model.cpu().eval()

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:

        # Pruner initialization
        self.pruner = PRUNERS[self.pruner_name]
        example_inputs = [b[0] for b in input_data.features.calib_dataloader]
        example_inputs = example_inputs[0]
        channel_groups = {}
        if isinstance(self.model, VisionTransformer):
            for m in self.model.modules():
                if isinstance(m, nn.MultiheadAttention):
                    channel_groups[m] = m.num_heads

        self.pruner = self.pruner(
            self.model,
            example_inputs,
            global_pruning=True,  # If False, a uniform ratio will be assigned to different layers.
            importance=self.importance,  # importance criterion for parameter selection
            iterative_steps=self.epochs,  # the number of iterations to achieve target ratio
            pruning_ratio=self.pruning_ratio,
            ignored_layers=self.ignored_layers,
            channel_groups=channel_groups,
            round_to=None,
            unwrapped_parameters=None

        )
        if True:
            base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
            for i in range(self.epochs):
                self.pruner.step()
                macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
                print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
                print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))
            ori_size = tp.utils.count_params(self.model)
        else:
            print("==============Before pruning=================")
            layer_channel_cfg = {}
            for module in self.model.modules():
                if module not in self.pruner.ignored_layers:
                    if isinstance(module, nn.Conv2d):
                        layer_channel_cfg[module] = module.out_channels
                    elif isinstance(module, nn.Linear):
                        layer_channel_cfg[module] = module.out_features
            self.pruner.step()
            print("==============After pruning=================")
            with torch.no_grad():
                params_after_prune = tp.utils.count_params(self.model)
                print("Params: %s => %s" % (ori_size, params_after_prune))
                out = self.model(example_inputs)
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.model if self.pruner is not None else self.predict_for_fit(input_data, output_mode)
