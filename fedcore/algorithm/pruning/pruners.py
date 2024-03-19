from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from torch import nn, optim
from torchvision.models import VisionTransformer
from torchvision.ops import FrozenBatchNorm2d

from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.repository.constanst_repository import PRUNERS, PRUNING_IMPORTANCE, PRUNING_LAYERS_IMPL, \
    PRUNER_REQUIRED_REG, PRUNER_REQUIRED_GRADS, PRUNER_WITHOUT_REQUIREMENTS


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        # finetune params
        self.criterion = params.get('loss', nn.CrossEntropyLoss())
        self.optimizer = params.get('optimizer', optim.Adam)
        self.learning_rate = params.get('lr', 0.001)

        # pruning gradients params
        self.criterion_for_grad = params.get('loss', nn.CrossEntropyLoss())
        self.learning_rate_for_grad = params.get('lr', 0.001)

        # pruning params
        self.pruner_name = params.get('pruner_name', 'meta_pruner')
        self.importance = params.get('importance', 'MagnitudeImportance')

        # pruning hyperparams
        self.pruning_ratio = params.get('pruning_ratio', 0.5)
        self.epochs = params.get('pruning_iterations', 2)
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
            # elif isinstance(m, FrozenBatchNorm2d):
            #     ignored_layers.append(m)
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
            ignored_layers.extend([model.rpn.head.cls_logits,
                                   model.backbone.fpn.layer_blocks,
                                   model.rpn.head.bbox_pred,
                                   model.roi_heads.box_head,
                                   model.roi_heads.box_predictor,
                                   model.roi_heads.keypoint_predictor])
        if model_name.__contains__('maskrcnn_resnet50_fpn_v2'):
            ignored_layers.extend([model.rpn.head.cls_logits, model.rpn.head.bbox_pred, model.roi_heads.box_predictor,
                                   model.roi_heads.mask_predictor])
        if model_name.__contains__('retinanet_resnet50_fpn_v2'):
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg])
            # For ViT: Rounding the number of channels to the nearest multiple of num_heads
        return ignored_layers

    def _fix_attention_layer(self):
        for name, m in self.model.named_modules():
            if hasattr(m, 'attention_head_size'):
                m.attention_head_size = m.attention_head_size // 4
            if hasattr(m, 'all_head_size'):
                m.all_head_size = m.all_head_size // 4

    def _finetune_pruned_model(self, input_data):
        self.finetune(finetune_object=self,
                      finetune_data=input_data)

    def _accumulate_grads(self, data, target, return_false=False):
        data, target = data.to(default_device()), target.to(default_device())
        out = self.model(data)
        loss = self.criterion_for_grad(out, target)
        loss.backward()
        if return_false:
            return loss

    def _pruner_iteration_func(self, pruner: callable, input_data):
        if isinstance(self.importance, tuple(PRUNER_WITHOUT_REQUIREMENTS.values())):
            return pruner
        elif isinstance(self.importance, tuple(PRUNER_REQUIRED_GRADS.values())):
            for i, (data, target) in enumerate(input_data.features.calib_dataloader):
                if i != 0:
                    print(f"Gradients accumulation iter- {i}")
                    print(f"==========================================")
                    # we using 1 batch as example of pruning quality
                    self._accumulate_grads(data, target)
            return pruner
        elif isinstance(self.importance, tuple(PRUNER_REQUIRED_REG.values())):
            pruner.update_regularizer()
            # <== initialize regularizer
            for i, (data, target) in enumerate(input_data.features.calib_dataloader):
                if i != 0:
                    print(f"Pruning reg iter- {i}")
                    print(f"==========================================")
                    # we using 1 batch as example of pruning quality
                    self.optimizer_for_grad.zero_grad()
                    loss = self._accumulate_grads(data, target, True)  # after loss.backward()
                    pruner.regularize(self.model, loss)  # <== for sparse training
                    self.optimizer_for_grad.step()
            return pruner

    def fit(self,
            input_data: InputData):
        self.model = input_data.target
        self.model_for_inference = deepcopy(self.model)
        self.optimizer_for_grad = optim.Adam(self.model.parameters(),
                                             lr=self.learning_rate_for_grad)
        self.num_classes = input_data.features.num_classes
        self.ignored_layers = self._filter_ignored_layers(self.model, str(self.model.__class__))
        self.model.cpu().eval()

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:

        # Pruner initialization
        self.pruner = PRUNERS[self.pruner_name]
        # list of tensors with dim size n_samples x n_channel x height x width
        all_batches = [b[0] for b in input_data.features.calib_dataloader]
        # take first batch
        first_batch = all_batches[0]

        channel_groups = {}
        if isinstance(self.model, VisionTransformer):
            for m in self.model.modules():
                if isinstance(m, nn.MultiheadAttention):
                    channel_groups[m] = m.num_heads

        self.pruner = self.pruner(
            self.model,
            first_batch,
            global_pruning=True,  # If False, a uniform ratio will be assigned to different layers.
            importance=self.importance,  # importance criterion for parameter selection
            iterative_steps=self.epochs,  # the number of iterations to achieve target ratio
            pruning_ratio=self.pruning_ratio,
            ignored_layers=self.ignored_layers,
            channel_groups=channel_groups,
            round_to=None,
            unwrapped_parameters=None

        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, first_batch)
        for i in range(self.epochs):
            self.pruner = self._pruner_iteration_func(self.pruner, input_data)
            self.pruner.step()
            print(f"Pruning iter - {i}")
            print(f"==========================================")

        print("==============After pruning=================")
        macs, nparams = tp.utils.count_ops_and_params(self.model, first_batch)
        print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))

        self._finetune_pruned_model(input_data)
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.model if self.pruner is not None else self.predict_for_fit(input_data, output_mode)
