from dataclasses import dataclass

import torch.utils.data
from fedot.core.data.data import InputData, OutputData


@dataclass
class CompressionInputData(InputData):
    train_dataloader: torch.utils.data.DataLoader = None
    calib_dataloader: torch.utils.data.DataLoader = None

@dataclass
class CompressionOutputData(OutputData):
    train_dataloader: torch.utils.data.DataLoader = None
    calib_dataloader: torch.utils.data.DataLoader = None
