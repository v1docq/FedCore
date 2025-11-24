import pytest 

from typing import *
import torch 

from fedcore.tools.ruler import PerformanceEvaluator 
from fedcore.metrics.quality import MetricFactory, LOADED_METRICS
from itertools import product


class PseudoDataloader:
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        feature_dim: int = 10,
        num_classes: int = 3,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        PyTorch-style pseudo dataloader that generates random data batches.
        
        Args:
            dataset_size: Total number of samples in the dataset
            batch_size: Number of samples per batch
            feature_dim: Dimension of input features
            num_classes: Number of output classes
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate indices
        self.indices = list(range(dataset_size))
        self.current_index = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Reset the dataloader for a new epoch."""
        self.current_index = 0
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of data."""
        if self.current_index >= self.dataset_size:
            raise StopIteration
        
        # Get batch indices
        end_index = min(self.current_index + self.batch_size, self.dataset_size)
        batch_indices = self.indices[self.current_index:end_index]
        actual_batch_size = len(batch_indices)
        
        # Generate pseudo data as PyTorch tensors
        # Features: random values from normal distribution
        batch_features = torch.randn(actual_batch_size, self.feature_dim, dtype=torch.float32)
        
        # Labels: random integers between 0 and num_classes-1
        batch_labels = torch.randint(0, self.num_classes, (actual_batch_size,), dtype=torch.long)
        
        self.current_index = end_index
        
        return batch_features, batch_labels
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

dataloader = PseudoDataloader(
        dataset_size=1000,
        batch_size=32,
        feature_dim=120,
        num_classes=5,
        shuffle=True,
        seed=42
    )

DEVICE = torch.device('cuda')
model = torch.nn.Linear(120, 40).to(DEVICE)


METRICS = ['Latency', 'Throughput', 'PowerConsumption', 'ModelSize']
DEVICES = ['CPU', 'CUDA']

@pytest.mark.parametrize('metric_name,device', product(METRICS, DEVICES))
def test_cpu(metric_name, device):
    if device == 'CPU':
        metric_name = device + metric_name
    assert metric_name is not None
    mtr = MetricFactory.get_metric(metric_name)
    assert mtr.get_value(model, dataloader) is not None, 'wrong `.metric` logic. the type-based decorator didn\'t cope with the input'
    try:
        mtr.metric(model, dataloader)
    except: pass
    else: raise