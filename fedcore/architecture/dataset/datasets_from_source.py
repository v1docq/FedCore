"""This module contains classes for wrapping data of various types
for passing it to the prediction method of computer vision models.
"""

import os
from enum import Enum

from typing import Callable, Tuple, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import numpy as np
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder
from torchvision.transforms import v2
from fedcore.architecture.dataset.task_specified.object_detection_datasets import YOLODataset, COCODataset
from fedcore.architecture.utils.paths import PROJECT_PATH, PATH_TO_DATA


class TorchVisionTransforms(Enum):
    STANDART_IMG_TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                                 transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                             std=[0.229, 0.224,
                                                                                                  0.225])])
    STANDART_TENSOR_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])
    STANDART_PIL_IMAGE_TRANSFORM = transforms.Compose(
        [transforms.PILToTensor(), transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])
    IMAGE_FLOAT32_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


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

BATCH_SIZE = 32


class DatasetFromTorchVision(Dataset):
    def __init__(self, path, transform=TorchVisionTransforms.STANDART_IMG_TRANSFORM):
        # directory containing the images
        self.train_dir, self.val_dir = (os.path.join(PROJECT_PATH, path, "train"),
                                        os.path.join(PROJECT_PATH, path, "validation"))
        # transform to be applied on images
        self.transform = transform

    def get_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(root=self.train_dir, transform=self.transform)
        # train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_dataset = torchvision.datasets.ImageFolder(root=self.val_dir, transform=self.transform)
        # test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        return train_dataset, test_dataset


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


class DatasetFromNumpy(Dataset):
    """Class for prediction on numpy arrays.

    Args:
        images: Numpy matrix of images.

    """

    def __init__(self, data_array: np.ndarray, transform) -> None:
        self.images = torch.from_numpy(data_array).float()

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


class DatasetFromFolder(DatasetFolder):
    """Class for prediction on images from folder.

        directory/
        ├── xxx.jpg
        ├── xxy.jpg

    Args:
        image_folder: Path to image folder.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.

    """

    def __init__(
            self,
            image_folder: str,
            transform: Callable = None,
    ) -> None:
        self.images = os.listdir(image_folder)
        self.root = image_folder
        self.transform = transform
        self._define_data_source(image_folder)

    def _define_data_source(self, image_folder):
        if self.images[0].lower().endswith(IMG_EXTENSIONS):
            self.is_dir_with_images = True
        elif any([x.__contains__('labels') for x in self.images]):
            path_to_images = os.path.join(image_folder, 'images')
            path_to_labels = os.path.join(image_folder, 'labels')
            self.images = [(os.path.join(path_to_images, img), os.path.join(path_to_labels, target))
                           for img, target in zip(os.listdir(path_to_images),
                                                  os.listdir(path_to_labels))]
            self.is_dir_with_images = False
        else:
            self.class_names, self.class_mapping = self.find_classes(image_folder)
            self.images = self.make_dataset(self.root, class_to_idx=self.class_mapping, extensions=IMG_EXTENSIONS)
            self.is_dir_with_images = False

    def _default_image_loader(self, path_to_image):
        return Image.open(path_to_image).convert("RGB")

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, id)``, where image is image tensor,
                and id is file name.

        """

        if self.is_dir_with_images:
            path_to_image, target = os.path.join(self.root, self.images[idx]), None
        else:
            path_to_image, target = self.images[idx]

        image = self.transform(self._default_image_loader(path_to_image))
        for ext in IMG_EXTENSIONS:
            if ext in target:
                target = self.transform(self._default_image_loader(target))
                break
        return image, target

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.images)


class AbstractDataset(Dataset):

    def __init__(self, data_source: Union[str, np.array],
                 annotation_source: str = None,
                 transformation_func: Callable = TorchVisionTransforms.STANDART_IMG_TRANSFORM.value):
        if isinstance(data_source, str):
            self.dataset_impl = DatasetFromFolder(data_source, transformation_func)
        elif isinstance(data_source, np.array):
            self.dataset_impl = DatasetFromNumpy(data_source, transformation_func)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        return self.dataset_impl.__getitem__(idx)


class ObjectDetectionDataset(AbstractDataset):

    def __init__(self, data_source: str,
                 annotation_source: str = None,
                 transformation_func: Callable = TorchVisionTransforms.IMAGE_FLOAT32_TRANSFORM.value):
        if data_source.__contains__('.yaml'):
            self.dataset_impl = YOLODataset(path=data_source,
                                            transform=transformation_func)
        else:
            self.dataset_impl = COCODataset(images_path=data_source,
                                            annotation_path=annotation_source,
                                            transform=transformation_func)



if __name__ == "__main__":
    #
    image_clf_source = os.path.join(PATH_TO_DATA, 'image_segmentation', "train")
    image_od_source = os.path.join(PATH_TO_DATA, 'object_detection', 'chips.yaml')
    image_od_annotation = os.path.join(PATH_TO_DATA, 'object_detection', "labels")
    torch_dataset = ObjectDetectionDataset(data_source=image_clf_source,
                                           annotation_source=image_od_annotation)
    # torch_dataset = AbstractDataset(data_source=image_clf_source)
    image, target = next(iter(torch_dataset))
    _ = 1
