from dataclasses import dataclass

import torch.utils.data
from fedot.core.data.data import InputData, OutputData


@dataclass
class CompressionInputData(InputData):
    train_dataloader: torch.utils.data.DataLoader = None
    calib_dataloader: torch.utils.data.DataLoader = None
    num_classes: int = None
    @property
    def shape(self):
        return (1, 1)


@dataclass
class CompressionOutputData(OutputData):
    train_dataloader: torch.utils.data.DataLoader = None
    calib_dataloader: torch.utils.data.DataLoader = None
    num_classes: int = None
    @property
    def shape(self):
        return (1, 1)

