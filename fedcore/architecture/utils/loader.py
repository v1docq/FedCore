import torch
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