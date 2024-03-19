from copy import deepcopy

import numpy as np
import torchvision
from fedot.core.data.data import InputData
from torch import nn, optim
from torchvision.models import VisionTransformer
from tqdm import tqdm

from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.data.data import TrainParams
from fedcore.metrics.metric_impl import calc_last_layer_loss, calc_intermediate_layers_attn_loss, \
    calc_intermediate_layers_feat_loss
from fedcore.repository.constanst_repository import PRUNERS, PRUNING_IMPORTANCE, PRUNING_LAYERS_IMPL


class BaseDistilator(BaseCompressionModel):
    """Class responsible for Distilation model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        # finetune params
        self.epochs = params.get('epochs', 15)
        self.criterion = params.get('loss', nn.CrossEntropyLoss())
        self.optimizer = params.get('optimizer', optim.Adam)
        self.learning_rate = params.get('lr', 0.001)

    def __repr__(self):
        return 'Distilation_model'

    def _init_distil_model(self, teacher_model):
        student_model = deepcopy(teacher_model)
        for module in student_model.segformer.encoder.block:
            del module[0]
        return student_model

    def finetune(self,
                 train_params: TrainParams
                 ):
        # metric = load_metric('mean_iou')

        self.base_model.to(train_params.device)
        self.student_model.to(train_params.device)

        self.base_model.eval()

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=train_params.learning_rate)
        step = 0
        for epoch in range(train_params.n_epochs):
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for idx, batch in pbar:
                self.student_model.train()
                # get the inputs;
                pixel_values = batch['pixel_values'].to(train_params.device)
                labels = batch['labels'].to(train_params.device)

                optimizer.zero_grad()

                # forward + backward + optimize
                student_outputs = self.student_model(
                    pixel_values=pixel_values,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                loss, student_logits = student_outputs.loss, student_outputs.logits

                with torch.no_grad():
                    teacher_output = self.base_model(
                        pixel_values=pixel_values,
                        labels=labels,
                        output_attentions=True,
                        output_hidden_states=True,
                    )

                last_layer_loss = calc_last_layer_loss(
                    student_logits,
                    teacher_output.logits,
                    train_params.last_layer_loss_weight,
                )

                student_attentions, teacher_attentions = student_outputs.attentions, teacher_output.attentions
                student_hidden_states, teacher_hidden_states = student_outputs.hidden_states, teacher_output.hidden_states

                intermediate_layer_att_loss = calc_intermediate_layers_attn_loss(
                    student_attentions,
                    teacher_attentions,
                    train_params.intermediate_attn_layers_weights,
                    train_params.student_teacher_attention_mapping,
                )

                intermediate_layer_feat_loss = calc_intermediate_layers_feat_loss(
                    student_hidden_states,
                    teacher_hidden_states,
                    train_params.intermediate_feat_layers_weights,
                )

                total_loss = loss * train_params.loss_weight + last_layer_loss
                if intermediate_layer_att_loss is not None:
                    total_loss += intermediate_layer_att_loss

                if intermediate_layer_feat_loss is not None:
                    total_loss += intermediate_layer_feat_loss

                step += 1

                total_loss.backward()
                optimizer.step()
                pbar.set_description(f'total loss: {total_loss.item():.3f}')

                for loss_value, loss_name in (
                        (loss, 'loss'),
                        (total_loss, 'total_loss'),
                        (last_layer_loss, 'last_layer_loss'),
                        (intermediate_layer_att_loss, 'intermediate_layer_att_loss'),
                        (intermediate_layer_feat_loss, 'intermediate_layer_feat_loss'),
                ):
                    if loss_value is None:  # для выключенной дистилляции атеншенов
                        continue
                    tb_writer.add_scalar(
                        tag=loss_name,
                        scalar_value=loss_value.item(),
                        global_step=step,
                    )

            # после модификаций модели обязательно сохраняйте ее целиком, чтобы подгрузить ее в случае чего
            torch.save(
                {
                    'model': self.student_model,
                    'state_dict': self.student_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                },
                f'{self.output_dir}/ckpt_{epoch}.pth',
            )

            eval_metrics = evaluate_model(self.student_model, valid_dataloader, id2label)

            for metric_key, metric_value in eval_metrics.items():
                if not isinstance(metric_value, float):
                    continue
                tb_writer.add_scalar(
                    tag=f'eval_{metric_key}',
                    scalar_value=metric_value,
                    global_step=epoch,
                )

    def _fit_distil_model(self, input_data):
        train_params = TrainParams(
            n_epochs=self.epochs,
            lr=self.learning_rate,
            batch_size=8,
            n_workers=8,
            device=default_device(),
            loss_weight=0.5,
            last_layer_loss_weight=0.5,
            intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
            intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
            student_teacher_attention_mapping={
                0: 1,
                1: 3,
                2: 5,
                3: 7
            },
        )
        self.finetune(train_params)
        pass

    def fit(self,
            input_data: InputData):
        self.base_model = input_data.target
        self.num_classes = input_data.features.num_classes
        self.student_model = self._init_distil_model(self.base_model)
        self._fit_distil_model(input_data)
        self.model.cpu().eval()

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:

        # Pruner initialization
        pass

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.predict_for_fit(input_data, output_mode)
