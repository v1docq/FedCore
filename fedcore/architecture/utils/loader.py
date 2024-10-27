import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from fedcore.architecture.comptutaional.devices import default_device


def collate(batch):
    batch = tuple(zip(*batch))
    images, targets = batch
    images = list(image.to(default_device()) for image in images)
    targets = [
        {
            k: v.to(default_device()) if isinstance(v, torch.Tensor) else v
            for k, v in t.items()
        }
        for t in targets
    ]
    return images, targets


def transform():
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.PILToTensor(),
        ]
    )
    return transform


def get_loader(dataset, batch_size: int = 1, train: bool = False):
    if train:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
        )
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    return loader
