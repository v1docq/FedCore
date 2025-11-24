"""Data structures for compression workflows.

This module defines lightweight containers used to pass data between
FedCore compression components (pruning, quantization, low-rank, etc.).
"""

from dataclasses import dataclass, field
import typing as tp

import numpy as np
import torch.utils.data
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass(init=False)
class CompressionInputData(InputData):
    """Input container for model compression and fine-tuning pipelines.

    This structure aggregates both raw arrays and PyTorch dataloaders
    so that compression algorithms can reuse the same interface as
    training code.

    Attributes
    ----------
    features : np.ndarray
        Raw input features used for calibration or fallback processing.
        Shape is typically ``(n_samples, n_features, ...)``.
    target : Optional[np.ndarray]
        Ground-truth targets aligned with ``features``. May be ``None``
        for purely unsupervised or inference-only scenarios.
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for training data used during fine-tuning or
        gradient-based pruning/quantization (may be ``None`` if not used).
    val_dataloader : torch.utils.data.DataLoader
        Dataloader for validation data used for evaluation or calibration
        (e.g. static quantization). May be ``None``.
    test_dataloader : torch.utils.data.DataLoader
        Optional test dataloader used for final evaluation.
    task : Task
        FEDOT task descriptor (e.g. classification, regression,
        time series forecasting).
    num_classes : int
        Number of target classes for classification tasks.
        ``None`` for regression or if unknown.
    input_dim : int
        Dimensionality of a single input sample (e.g. feature count or
        channel count), if known.
    model : Any
        Optional reference to the underlying model associated with this
        dataset (e.g. a pretrained network).
    supplementary_data : SupplementaryData
        FEDOT supplementary metadata attached to the dataset
        (e.g. indexes, masks, service info).
    """

    idx = None
    features = None
    data_type = None
    train_dataloader: torch.utils.data.DataLoader = None
    val_dataloader: torch.utils.data.DataLoader = None
    test_dataloader: torch.utils.data.DataLoader = None
    input_dim: int = None
    num_classes: int = None
    model: tp.Any = None

    def __init__(self, idx=None, features=None, data_type=None,
                 train_dataloader=None, val_dataloader=None, test_dataloader=None,
                 input_dim=None, num_classes=None, model=None, **kwargs):
        if idx is None:
            idx = np.arange(1)
        if data_type is None:
            data_type = DataTypesEnum.image
        super().__init__(idx=idx, features=features, data_type=data_type, **kwargs)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = model

    @property
    def shape(self):
        """Tuple-like interface for shape compatibility.

        Notes
        -----
        Currently returns a dummy ``(1, 1)`` placeholder and is mainly
        used to satisfy interfaces expecting a ``.shape`` attribute.
        """
        return (1, 1)


@dataclass
class CompressionOutputData:
    """Output container for compressed model inference results.

    This structure mirrors FEDOT's :class:`OutputData`, but is tailored
    for FedCore compression workflows where both predictions and the
    compressed model reference may be needed.

    Attributes
    ----------
    features : np.ndarray
        Features used during prediction (may be preprocessed).
    idx : list
        Sample indices associated with predictions.
    target : Optional[np.ndarray]
        Ground-truth targets, if available.
    num_classes : int
        Number of target classes for classification problems.
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader corresponding to the dataset used to
        obtain these predictions (optional).
    val_dataloader : torch.utils.data.DataLoader
        Validation dataloader corresponding to the dataset used to
        obtain these predictions (optional).
    task : Task
        FEDOT task descriptor describing the prediction problem.
    data_type : DataTypesEnum
        Type of data (e.g. table, image, time series).
    model : callable
        Reference to the (possibly compressed) model used for inference.
    predict : callable
        Callable or object holding prediction outputs in a FedCore-
        compatible format (e.g. tensor or :class:`OutputData`-like).
    supplementary_data : SupplementaryData
        Additional metadata associated with predictions.
    """

    features: np.ndarray = None
    num_classes: int = None
    train_dataloader: torch.utils.data.DataLoader = None
    val_dataloader: torch.utils.data.DataLoader = None
    task: Task = Task(TaskTypesEnum.classification)
    data_type: DataTypesEnum = DataTypesEnum.image
    model: callable = None
    predict: callable = None
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)
    # ModelRegistry integration fields
    checkpoint_path: tp.Optional[str] = None
    model_id: tp.Optional[str] = None
    fedcore_id: tp.Optional[str] = None

    @property
    def shape(self):
        """Tuple-like interface for shape compatibility.

        Notes
        -----
        Currently returns a dummy ``(1, 1)`` placeholder to align with
        APIs expecting a ``.shape`` attribute.
        """
        return (1, 1)


class TrainParams:
    """Typed configuration stub for training hyperparameters.

    This class is used as a type container for knowledge-distillation
    and teacher–student training setups.

    Attributes
    ----------
    loss_weight : float
        Global coefficient for the main loss term.
    last_layer_loss_weight : float
        Additional coefficient for the loss applied to the final layer
        (e.g. head distillation).
    intermediate_attn_layers_weights : Tuple[float, float, float, float]
        Weights for distillation/regularization terms on intermediate
        attention layers.
    intermediate_feat_layers_weights : Tuple[float, float, float, float]
        Weights for distillation/regularization terms on intermediate
        feature (hidden representation) layers.
    student_teacher_attention_mapping : Dict
        Mapping between student and teacher attention layers that should
        be aligned (e.g. {student_layer_idx: teacher_layer_idx}).
    """

    loss_weight: float
    last_layer_loss_weight: float
    intermediate_attn_layers_weights: tp.Tuple[float, float, float, float]
    intermediate_feat_layers_weights: tp.Tuple[float, float, float, float]
    student_teacher_attention_mapping: tp.Dict
