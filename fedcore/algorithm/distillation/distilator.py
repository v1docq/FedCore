from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from torch import nn, optim
from tqdm import tqdm
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.data.data import TrainParams
from fedcore.metrics.cv_metrics import (
    LastLayer,
    IntermediateAttention,
    IntermediateFeatures,
)


class BaseDistilator(BaseCompressionModel):
    """Class responsible for Distilation model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        # finetune params
        self.epochs = params.get("epochs", 15)
        self.criterion = params.get("loss", nn.CrossEntropyLoss())
        self.optimizer = params.get("optimizer", optim.Adam)
        self.learning_rate = params.get("lr", 0.001)

    def __repr__(self):
        return "Distilation_model"

    def _init_distil_model(self, teacher_model):
        student_model = deepcopy(teacher_model)
        for module in student_model.segformer.encoder.block:
            del module[0]
        return student_model

    def _calc_losses(self, loss, train_params, output_dict):
        last_layer_loss = LastLayer(
            output_dict["student_logits"],
            output_dict["teacher_logits"],
            train_params.last_layer_loss_weight,
        )
        intermediate_layer_att_loss = IntermediateAttention(
            output_dict["student_attentions"],
            output_dict["teacher_attentions"],
            train_params.intermediate_attn_layers_weights,
            train_params.student_teacher_attention_mapping,
        )

        intermediate_layer_feat_loss = IntermediateFeatures(
            output_dict["student_hidden_states"],
            output_dict["teacher_hidden_states"],
            train_params.intermediate_feat_layers_weights,
        )

        total_loss = output_dict["loss"] * train_params.loss_weight + last_layer_loss
        if intermediate_layer_att_loss is not None:
            total_loss += intermediate_layer_att_loss

        if intermediate_layer_feat_loss is not None:
            total_loss += intermediate_layer_feat_loss

        return total_loss

    def finetune(self, input_data: InputData, train_params: TrainParams):
        # metric = load_metric('mean_iou')

        self.base_model.to(default_device())
        self.student_model.to(default_device())

        self.base_model.eval()

        optimizer = torch.optim.AdamW(
            self.student_model.parameters(), lr=self.learning_rate
        )
        step = 0
        for epoch in range(self.epochs):
            pbar = tqdm(
                enumerate(input_data.features.train_dataloader),
                total=len(input_data.features.train_dataloader),
            )
            for idx, batch in pbar:
                self.student_model.train()
                optimizer.zero_grad()

                # get the inputs;
                pixel_values = batch["pixel_values"].to(default_device())
                labels = batch["labels"].to(default_device())

                # outputs
                student_outputs = self.student_model(
                    pixel_values=pixel_values,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                with torch.no_grad():
                    teacher_output = self.base_model(
                        pixel_values=pixel_values,
                        labels=labels,
                        output_attentions=True,
                        output_hidden_states=True,
                    )

                output_dict = {
                    "student_logits": student_outputs.logits,
                    "teacher_logits": teacher_output.logits,
                    "student_attentions": student_outputs.attentions,
                    "teacher_attentions": teacher_output.attentions,
                    "student_hidden_states": student_outputs.hidden_states,
                    "teacher_hidden_states": teacher_output.hidden_states,
                }

                total_loss = self._calc_losses(
                    student_outputs.loss, train_params, output_dict
                )
                step += 1

                total_loss.backward()
                optimizer.step()
                pbar.set_description(f"total loss: {total_loss.item():.3f}")

            # после модификаций модели обязательно сохраняйте ее целиком, чтобы подгрузить ее в случае чего
            torch.save(
                {
                    "model": self.student_model,
                    "state_dict": self.student_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                f"{self.output_dir}/ckpt_{epoch}.pth",
            )

    def _fit_distil_model(
        self, input_data: InputData, distilation_params: TrainParams = None
    ):
        if distilation_params is None:
            distilation_params = TrainParams(
                loss_weight=0.5,
                last_layer_loss_weight=0.5,
                intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
                intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
                student_teacher_attention_mapping={0: 1, 1: 3, 2: 5, 3: 7},
            )
        self.finetune(input_data, distilation_params)

    def fit(self, input_data: InputData):
        self.base_model = input_data.target
        self.num_classes = input_data.features.num_classes
        self.student_model = self._init_distil_model(self.base_model)
        self._fit_distil_model(input_data)
        self.model.cpu().eval()

    def predict_for_fit(
        self, input_data: InputData, output_mode: str = "default"
    ) -> np.array:

        # Pruner initialization
        pass

    def predict(self, input_data: InputData, output_mode: str = "default") -> np.array:
        return self.predict_for_fit(input_data, output_mode)
