from dataclasses import dataclass, field
import typing as tp

import numpy as np
import torch.utils.data
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class CompressionInputData:
    features: np.ndarray = None
    target: tp.Optional[np.ndarray] = None
    train_dataloader: torch.utils.data.DataLoader = None
    calib_dataloader: torch.utils.data.DataLoader = None
    task: Task = Task(TaskTypesEnum.classification)
    num_classes: int = None
    model = None
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)

    @property
    def shape(self):
        return (1, 1)


@dataclass
class CompressionOutputData:
    features: np.ndarray = None
    idx: list = None
    target: tp.Optional[np.ndarray] = None
    num_classes: int = None
    train_dataloader: torch.utils.data.DataLoader = None
    calib_dataloader: torch.utils.data.DataLoader = None
    task: Task = Task(TaskTypesEnum.classification)
    data_type: DataTypesEnum = DataTypesEnum.image
    model: callable = None
    predict: callable = None
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)

    @property
    def shape(self):
        return (1, 1)


class TrainParams:
    loss_weight: float
    last_layer_loss_weight: float
    intermediate_attn_layers_weights: tp.Tuple[float, float, float, float]
    intermediate_feat_layers_weights: tp.Tuple[float, float, float, float]
    student_teacher_attention_mapping: tp.Dict
