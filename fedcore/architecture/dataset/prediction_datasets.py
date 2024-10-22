"""This module contains classes for wrapping data of various types
for passing it to the prediction method of computer vision models.
"""

import os
from typing import Callable, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import numpy as np
import torch.utils.data as data
import torchvision
from torchvision import transforms

from fedcore.architecture.utils.paths import PROJECT_PATH

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)
TRANSFORM_IMG = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
BATCH_SIZE = 32


class TorchVisionDataset(Dataset):
    def __init__(self, path, transform=TRANSFORM_IMG):
        # directory containing the images
        self.train_dir, self.val_dir = os.path.join(
            PROJECT_PATH, path, "train"
        ), os.path.join(PROJECT_PATH, path, "validation")
        # transform to be applied on images
        self.transform = transform

    def get_dataloader(self):
        train_data = torchvision.datasets.ImageFolder(
            root=self.train_dir, transform=TRANSFORM_IMG
        )
        train_data_loader = data.DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        test_data = torchvision.datasets.ImageFolder(
            root=self.val_dir, transform=TRANSFORM_IMG
        )
        test_data_loader = data.DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        return train_data_loader, test_data_loader


class CustomDatasetForImages(Dataset):
    # defining constructor
    def __init__(self, directory, annotations, transform=None):
        # directory containing the images
        self.directory = directory
        annotations_file_dir = os.path.join(self.directory, annotations)
        # loading the csv with info about images
        self.labels_mask = pd.read_csv(annotations_file_dir)
        # transform to be applied on images
        self.transform = transform
        self.labels = os.listdir(os.path.join(self.directory, "labels"))
        self.images_path = os.path.join(self.directory, "images")
        self.labels_path = os.path.join(self.directory, "labels")

    # getting the length
    def __len__(self):
        return len(self.labels)

    def bbox_converter(
        self, center_X, center_y, width, height, image_width, image_height
    ):
        x1 = int((center_X - width / 2) * image_width)
        x2 = int((center_X + width / 2) * image_width)
        x2 = x2 - x1
        y1 = int((center_y - height / 2) * image_height)
        y2 = int((center_y + height / 2) * image_height)
        y2 = y2 - y1
        return [x1, y1, x2, y2]

    # getting the data items
    def __getitem__(self, idx):
        # defining the image path
        image_path = os.path.join(self.images_path, self.labels[idx]).replace(
            ".txt", ".png"
        )
        label_path = os.path.join(self.labels_path, self.labels[idx])
        # reading the images
        image = read_image(image_path)
        image = image.to(torch.float32)
        annotation = np.loadtxt(label_path, ndmin=2)
        labels = annotation[:, 0] + 1
        boxes = annotation[:, 1:]
        c, h, w = image.shape
        boxes *= [w, h, w, h]
        area = boxes[:, 2] * boxes[:, 3]
        # x centre, y centre, w, h -> x1, y1, w, h
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]  # x1, y1, w, h -> x1, y1, x2, y2

        # apply the transform if not set to None
        if self.transform:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.zeros(annotation.shape[0], dtype=torch.int64),
        }
        # returning the image and label
        return image[np.newaxis, :, :, :], target

    @property
    def shape(self):
        return (len(self.labels), 1)

    @property
    def num_classes(self):
        return self.labels_mask.shape[0]


class PredictionNumpyDataset(Dataset):
    """Class for prediction on numpy arrays.

    Args:
        images: Numpy matrix of images.

    """

    def __init__(
        self,
        images: np.ndarray,
    ) -> None:
        self.images = torch.from_numpy(images).float()

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, id)``, where image is image tensor,
                and id is integer.

        """
        return self.images[idx], idx

    def __len__(self) -> int:
        """Return length of dataset"""
        return self.images.size()[0]


class PredictionFolderDataset(Dataset):
    """Class for prediction on images from folder.

    Args:
        image_folder: Path to image folder.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.

    """

    def __init__(
        self,
        image_folder: str,
        transform: Callable,
    ) -> None:
        self.root = image_folder
        self.images = []
        for address, dirs, files in os.walk(image_folder):
            for name in files:
                if name.lower().endswith(IMG_EXTENSIONS):
                    self.images.append(os.path.join(address, name))
        self.transform = transform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, id)``, where image is image tensor,
                and id is file name.

        """

        image = Image.open(os.path.join(self.root, self.images[idx])).convert("RGB")
        image = self.transform(image)
        return image, self.images[idx]

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.images)
