import torch
from torch.utils.data import Dataset


class DummyDatasetCLF(Dataset):
    def __init__(self, num_samples, channels=1, size=28, classes=10):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, channels, size, size)
        self.targets = torch.randint(0, 10, (num_samples,))
        self.classes = list(range(classes))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
