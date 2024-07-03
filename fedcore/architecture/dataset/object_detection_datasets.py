"""This module contains classes for object detection task based on torch dataset."""

import json
import os
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import yaml
import imageio
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


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

        for category in data['categories']:
            id = category['id'] + 1 if fix_zero_class else category['id']
            self.classes[id] = category['name']

        samples = {}
        for image in data['images']:
            samples[image['id']] = {
                'image': os.path.join(images_path, image['file_name']),
                'area': [],
                'iscrowd': [],
                'labels': [],
                'boxes': [],
            }

        for annotation in tqdm(data['annotations']):
            if annotation['area'] > 0:
                bbox = np.array(annotation['bbox'])
                bbox[2:] += bbox[:2]  # x, y, w, h -> x1, y1, x2, y2
                labels = annotation['category_id']
                labels = labels + 1 if fix_zero_class else labels
                labels = 1 if replace_to_binary else labels
                samples[annotation['image_id']]['labels'].append(labels)
                samples[annotation['image_id']]['boxes'].append(bbox)
                samples[annotation['image_id']]['area'].append(
                    annotation['area'])
                samples[annotation['image_id']]['iscrowd'].append(
                    annotation['iscrowd'])

        for sample in samples.values():
            if len(sample['labels']) > 0:
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
        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)
        target = {
            'boxes': torch.tensor(np.stack(sample['boxes']), dtype=torch.float32),
            'labels': torch.tensor(sample['labels'], dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(sample['area'], dtype=torch.float32),
            'iscrowd': torch.tensor(sample['iscrowd'], dtype=torch.int64),
        }
        return image, target

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)


def img2label_paths(img_path):
    """Define label path as a function of image path."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'


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
        path: str,
        transform: Callable,
        train: bool = True,
        replace_to_binary: bool = False,
    ) -> None:

        self.transform = transform
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self.root = os.path.abspath(os.path.join(os.path.dirname(
            path), (data['train'] if train else data['val'])))
        self.classes = ['background']
        self.classes.extend(['object'] if replace_to_binary else data['names'])
        self.binary = replace_to_binary
        self.samples = []

        for file in os.listdir(self.root):
            if file.lower().endswith(IMG_EXTENSIONS):
                self.samples.append(
                    {
                        'image': os.path.join(self.root, file),
                        'label': img2label_paths(os.path.join(self.root, file))
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
        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)
        annotation = np.loadtxt(sample['label'], ndmin=2)
        labels = annotation[:, 0] + 1
        labels = np.ones_like(labels) if self.binary else labels
        boxes = annotation[:, 1:]
        c, h, w = image.shape
        boxes *= [w, h, w, h]
        area = boxes[:, 2] * boxes[:, 3]
        # x centre, y centre, w, h -> x1, y1, w, h
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]  # x1, y1, w, h -> x1, y1, x2, y2

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        _ = {
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(area, dtype=torch.float32),
            'iscrowd': torch.zeros(annotation.shape[0], dtype=torch.int64)
        }

        return image, labels, boxes, _

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)


class LabelEncoder():

    # init function with our switcher with label 
    def __init__(self) -> None:
        # switch statement for label encoding
        switcher = {
            "Car": 0.000,
            "Pedestrian": 0.111,
            "Cyclist": 0.222,
            "Tram": 0.333,
            "Truck": 0.444,
            "Van": 0.555,
            "Person_sitting": 0.666,
            "Misc": 0.777,
            "DontCare": 0.888
        }
        self.switcher = switcher

    # takes in a label and returns the encoded label
    def KittiLabelEncoder(self, label):
        # return the encoded label
        return self.switcher.get(label, "Invalid label")
    
    # We use self in python functions because in python methods are passed automatically
    # but not received automatically, so we need self to receive an instance of the method
    # (hence, if you remove self, you get "1 positional arguments expected but got 0")
    def len_classes(self):
        return len(self.switcher)


class KittiCustomDataset(Dataset):
    # initialize constructor
    def __init__(self, image_dir, annotations, transforms=None):
        self.image_dir  = image_dir
        self.annotations  = annotations
        self.transforms = transforms

        # sorted files
        self.image_files = sorted(os.listdir(image_dir))


    def __getitem__(self, index):
        # grab the image, label, and its bounding box coordinates
        image_path = os.path.join(self.image_dir, self.image_files[index])
        text = self.annotations
        # print("len text", len(text))
        # print("index: ", index)
        # print("annotations1", text)
        print(f"label index {index}", text[index][0])
        print(f"bbox index {index}", text[index][1])
        labels = text[index][0]
        bboxes = text[index][1]
        normalized_bboxes = []

        # loads the image
        image = imageio.imread(image_path, pilmode="RGB")
        # print("image shape: ", image.shape)
        # gets img_height and width
        img_height, img_width = image.shape[:2]

        print("wid, hgih ", img_height, img_width)
        for i in range(len(bboxes)):
            x_start, y_start, x_end, y_end = bboxes[i]      # we get the coordinates from the array...

            # ...normalize against image height & width...
            x_start = x_start / img_width
            y_start = y_start / img_height
            x_end   = x_end / img_width
            y_end   = y_end / img_height
            print(f"tuple: ({x_start}, {y_start}, {x_end}, {y_end})")   # ...normalize against image height & width...
            # ...and create a tuple of normalized coordinates
            normalized_bboxes.append((x_start,
                                    y_start,
                                    x_end,
                                    y_end))
        
        print("norm bbox: ", normalized_bboxes)
        # os.wait()

		# check to see if we have any image transformations to apply
		# and if so, apply them
        if self.transforms:
            image = self.transforms(image)

        # we need to one hot encode the labels that the model can perform better
        # first, get unique labels
        unique_labels = set(labels)
        unique_labels = list(unique_labels)
        # print("!!unique_labels ", unique_labels)

        encoded_labels = []
        # iterate through each item in unique_labels and encode them
        for label in unique_labels:
            # print("label: ", label)
            le = LabelEncoder()
            encoded_label = le.KittiLabelEncoder(label)
            encoded_labels.append(encoded_label)

        # print("bboxes: ", bboxes)
        # os.wait()
        labels = torch.from_numpy(np.asarray(encoded_labels))
        ret_bboxes = torch.from_numpy(np.asarray(normalized_bboxes))
        print("encoded_labels:", encoded_labels)
        print("_labels:", labels)
        # os.wait()

        return image, labels, ret_bboxes
        # return image, encoded_labels, bboxes


    def __len__(self):
        # return size of dataset
        return len(self.annotations)


# label.txt format for KITTI dataset
"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""