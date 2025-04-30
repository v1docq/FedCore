import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
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


def image_transform():
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
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
