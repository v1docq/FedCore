from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from fedcore.architecture.utils.paths import wrap_with_project_root_path

def get_small_cifar10_train_and_val_loaders(batch_size = 10, train_size=100, val_size=100) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10(root=wrap_with_project_root_path("./data"), train=True, download=True, transform=transform)

    small_train_subset = Subset(dataset, range(train_size))
    small_val_subset = Subset(dataset, range(train_size, train_size + val_size))

    train_dataloader = DataLoader(small_train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(small_val_subset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader