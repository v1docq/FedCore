import numpy as np
from fedot.core.data.data import InputData
from torch import nn, optim
from tqdm import tqdm
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from typing import Optional
import torch
import logging
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.computational.devices import default_device
from fedcore.data.data import TrainParams
# from fedcore.metrics.cv_metrics import (
#     LastLayer,
#     IntermediateAttention,
#     IntermediateFeatures,
# )

LastLayer = None
IntermediateAttention = None
IntermediateFeatures = None


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
        self._distill_index = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self):
        return "Distilation_model"

    def _init_distil_model(self, teacher_model):
        """Initialize student model from teacher model without deepcopy.
        
        Strategy:
        1. Try to instantiate a new model of the same type
        2. Load teacher's state_dict into student
        3. Modify student architecture as needed (e.g., remove layers)
        4. If instantiation fails, save and reload from checkpoint
        """
        try:
            model_class = type(teacher_model)
            if hasattr(model_class, '__call__'):
                try:
                    student_model = model_class()
                except TypeError:
                    if hasattr(teacher_model, 'config'):
                        student_model = model_class(teacher_model.config)
                    else:
                        temp_checkpoint = self._registry.register_model(
                            fedcore_id=self._fedcore_id,
                            model=teacher_model,
                            stage="before",
                            mode=self.__class__.__name__
                        )
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        student_model = self._registry.load_model_from_latest_checkpoint(
                            self._fedcore_id, temp_checkpoint, device
                        )
                        if student_model is None:
                            raise RuntimeError("Failed to create student model from checkpoint")
                
                if hasattr(student_model, 'load_state_dict') and hasattr(teacher_model, 'state_dict'):
                    student_model.load_state_dict(teacher_model.state_dict())
            else:
                raise TypeError("Cannot instantiate model")
                
        except Exception as e:
            self.logger.warning(f"Warning: Standard initialization failed ({e}), using checkpoint-based approach")
            temp_checkpoint = self._registry.register_model(
                fedcore_id=self._fedcore_id,
                model=teacher_model,
                stage="after",
                mode=self.__class__.__name__
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            student_model = self._registry.load_model_from_latest_checkpoint(
                self._fedcore_id, temp_checkpoint, device
            )
            if student_model is None:
                raise RuntimeError("Failed to create student model")
        
        if hasattr(student_model, 'segformer') and hasattr(student_model.segformer, 'encoder'):
            for module in student_model.segformer.encoder.block:
                if len(module) > 0:
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
                enumerate(input_data.train_dataloader),
                total=len(input_data.train_dataloader),
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
        self.num_classes = input_data.num_classes
        self.model_before = self.base_model
        self.student_model = self._init_distil_model(self.base_model)
        self._model_registry = ModelRegistry()
        self._distill_index += 1
        
        if self._model_id_before:
            self._model_registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage="before",
                mode=self.__class__.__name__
            )
        
        self._fit_distil_model(input_data)
        
        self.model_after = self.student_model
        
        if self._model_id_after:
            self._model_registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_after,
                metrics={},
                stage="after",
                mode=self.__class__.__name__
            )
        
        self.student_model.cpu().eval()

    def predict_for_fit(
        self, input_data: InputData, output_mode: str = "default"
    ) -> np.array:

        # Pruner initialization
        pass

    def predict(self, input_data: InputData, output_mode: str = "default") -> np.array:
        return self.predict_for_fit(input_data, output_mode)
