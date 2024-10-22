"""This module contains classes for object detection task based on torch dataset."""

import json
import os
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import opendatasets as od
from opendatasets.utils.archive import extract_archive

from fedcore.architecture.utils.paths import (
    data_path,
    yolo_data_path,
    yolo_yaml_path,
    YOLO_DATA_URL,
    YOLO_YAML_URL,
)
from fedcore.architecture.utils.loader import transform

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


class COCODataset(Dataset):
    """Class-loader for COCO json.

    Args:
        images_path: Image folder path.
        json_path: Json file path.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.
        fix_zero_class: If ``True`` add 1 for each class label
            (0 represents always the background class).
        replace_to_binary: If ``True`` set label 1 for any class.

    """

    def __init__(
        self,
        images_path: str,
        json_path: str,
        transform: Callable,
        fix_zero_class: bool = False,
        replace_to_binary: bool = False,
    ) -> None:
        self.transform = transform
        self.classes = {}
        self.samples = []

        with open(json_path) as f:
            data = json.load(f)

        for category in data["categories"]:
            id = category["id"] + 1 if fix_zero_class else category["id"]
            self.classes[id] = category["name"]

        samples = {}
        for image in data["images"]:
            samples[image["id"]] = {
                "image": os.path.join(images_path, image["file_name"]),
                "area": [],
                "iscrowd": [],
                "labels": [],
                "boxes": [],
            }

        for annotation in tqdm(data["annotations"]):
            if annotation["area"] > 0:
                bbox = np.array(annotation["bbox"])
                bbox[2:] += bbox[:2]  # x, y, w, h -> x1, y1, x2, y2
                labels = annotation["category_id"]
                labels = labels + 1 if fix_zero_class else labels
                labels = 1 if replace_to_binary else labels
                samples[annotation["image_id"]]["labels"].append(labels)
                samples[annotation["image_id"]]["boxes"].append(bbox)
                samples[annotation["image_id"]]["area"].append(annotation["area"])
                samples[annotation["image_id"]]["iscrowd"].append(annotation["iscrowd"])

        for sample in samples.values():
            if len(sample["labels"]) > 0:
                self.samples.append(sample)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, targets)``, where image is image tensor,
                and targets is dict with keys: ``'boxes'``, ``'labels'``,
                ``'image_id'``, ``'area'``, ``'iscrowd'``.

        """
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        image = self.transform(image)
        if len(sample["boxes"]) != 0:
            targets = {
                "labels": torch.tensor(sample["labels"], dtype=torch.int64),
                "boxes": torch.tensor(np.stack(sample["boxes"]), dtype=torch.float32),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor(sample["area"], dtype=torch.float32),
                "iscrowd": torch.tensor(sample["iscrowd"], dtype=torch.int64),
            }
        else:
            targets = {
                "labels": torch.zeros(0, dtype=torch.int64),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        return image, targets

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)


def img2label_paths(img_path):
    """Define label path as a function of image path."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return sb.join(img_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


class YOLODataset(Dataset):
    """Class-loader for YOLO format (https://docs.ultralytics.com/datasets/detect/).

    Args:
        path: YAML file path.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.
        train: If True, creates dataset from training set, otherwise creates from test set.
        replace_to_binary: If ``True`` set label 1 for any class.

    """

    def __init__(
        self,
        path: str = None,
        dataset_name: str = None,
        transform: Callable = transform(),
        train: bool = True,
        replace_to_binary: bool = False,
        download: bool = False,
        log: bool = False,
    ) -> None:

        if dataset_name is not None:
            path_flag = os.path.isdir(data_path(dataset_name, log=log))
            if path_flag is False or download is True:
                dataset_url = f"{YOLO_DATA_URL}{dataset_name}.zip"
                yaml_url = f"{YOLO_YAML_URL}{dataset_name}.yaml"

                od.download(dataset_url, data_dir=data_path(dataset_name))
                od.download(yaml_url, data_dir=data_path(dataset_name))
                extract_archive(
                    from_path=str(yolo_data_path(dataset_name)),
                    to_path=str(data_path(dataset_name)),
                    remove_finished=True,
                )

            path = yolo_yaml_path(dataset_name)

        self.transform = transform
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.root = os.path.abspath(
            os.path.join(
                os.path.dirname(path), (data["train"] if train else data["test"])
            )
        )
        self.classes = {0: "background"}

        for k in data["names"]:
            id = k + 1
            self.classes[id] = data["names"][k]

        self.binary = replace_to_binary
        self.samples = []

        for file in os.listdir(self.root):
            if file.lower().endswith(IMG_EXTENSIONS):
                self.samples.append(
                    {
                        "image": os.path.join(self.root, file),
                        "label": img2label_paths(os.path.join(self.root, file)),
                    }
                )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, targets)``, where image is image tensor,
                and targets is dict with keys: ``'boxes'``, ``'labels'``,
                ``'image_id'``, ``'area'``, ``'iscrowd'``.

        """
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        image = self.transform(image)
        annotation = np.loadtxt(sample["label"], ndmin=2)
        labels = annotation[:, 0] + 1
        labels = np.ones_like(labels) if self.binary else labels
        boxes = annotation[:, 1:]
        if len(boxes) != 0:
            c, h, w = image.shape
            boxes *= [w, h, w, h]
            area = boxes[:, 2] * boxes[:, 3]
            # x centre, y centre, w, h -> x1, y1, w, h
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]  # x1, y1, w, h -> x1, y1, x2, y2

            targets = {
                "labels": torch.tensor(labels, dtype=torch.int64),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor(area, dtype=torch.float32),
                "iscrowd": torch.zeros(annotation.shape[0], dtype=torch.int64),
            }
        else:
            targets = {
                "labels": torch.zeros(0, dtype=torch.int64),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        return image, targets

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)


class UnlabeledDataset(Dataset):
    """Class-loader for custom dataset.

    Args:
        images_path: Image folder path.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.
    """

    def __init__(self, images_path: str, transform: Callable = transform()) -> None:
        self.transform = transform
        self.images_path = images_path

        self.samples = []
        for file in os.listdir(self.images_path):
            if file.lower().endswith(IMG_EXTENSIONS):
                self.samples.append(
                    {"image": os.path.join(self.images_path, file), "name": file}
                )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, targets)``, where image is image tensor,
                and targets is dict with keys: ``'name'``, ``'boxes'``, ``'labels'``,
                ``'image_id'``, ``'area'``, ``'iscrowd'``.

        """
        sample = self.samples[idx]
        image = Image.open(sample["image"])
        image = self.transform(image)
        targets = {
            "name": sample["name"],
            "labels": torch.zeros(0, dtype=torch.int64),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "image_id": torch.tensor([idx]),
            "area": torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
        return image, targets

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)
