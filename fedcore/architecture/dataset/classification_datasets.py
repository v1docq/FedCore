import os
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class YOLOClassificationDataset(Dataset):
    def __init__(self, 
                 path: str, 
                 transform: Optional[Callable] = None, 
                 cache: str = "none", 
                 train: bool = True):
        self.path = Path(path).resolve()
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.cache_ram = cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.train = train

        class_folders = [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
        for class_idx, class_name in enumerate(class_folders):
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            class_path = os.path.join(path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(IMG_EXTENSIONS):
                    img_path = os.path.join(class_path, img_name)
                    cache_path = Path(img_path).with_suffix(".npy")
                    self.samples.append([img_path, class_idx, cache_path, None])

        if self.cache_ram:
            self._cache_images_ram()
        elif self.cache_disk:
            self._cache_images_disk()

    def _cache_images_ram(self):
        for i, (img_path, class_idx, _, _) in enumerate(tqdm(self.samples, desc="Кэширование в RAM")):
            img = cv2.imread(img_path)
            if img is not None:
                self.samples[i][3] = img

    def _cache_images_disk(self):
        for i, (img_path, class_idx, cache_path, _) in enumerate(tqdm(self.samples, desc="Кэширование на диск")):
            if not cache_path.exists():
                img = cv2.imread(img_path)
                if img is not None:
                    np.save(str(cache_path), img)

    def __getitem__(self, idx):
        img_path, class_idx, cache_path, img = self.samples[idx]

        if self.cache_ram:
            if img is None:
                img = self.samples[idx][3] = cv2.imread(img_path)
        elif self.cache_disk:
            if cache_path.exists():
                img = np.load(str(cache_path))
            else:
                img = cv2.imread(img_path)
        else:
            img = cv2.imread(img_path)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        targets = class_idx
        return img, targets

    def __len__(self):
        return len(self.samples)

    def get_dataloader(self, params: dict = None):
        batch_size = params.get("train_bs", 100) if self.train else params.get("val_bs", 64)
        shuffle = params.get("train_shuffle", True) if self.train else params.get("val_shuffle", False) 
        num_workers = params.get("num_workers", 4)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)