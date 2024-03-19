from transformers import SegformerFeatureExtractor
from PIL import Image

from torch.utils.data import Dataset

import numpy as np

from torch.utils.data import DataLoader


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        split_name = "train.txt" if self.train else "test.txt"

        with open(f"{root_dir}/{split_name}") as f:
            self.images = [line.replace("\n", "") for line in f.readlines()]

        self.annotations = [
            img_path.replace("clip_img", "matting")
            .replace("clip_0", "matting_0")
            .replace(".jpg", ".png")
            for img_path in self.images
        ]

        assert len(self.images) == len(
            self.annotations
        ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        segmentation_map = np.array(Image.open(self.annotations[idx]).split()[-1])
        segmentation_map[segmentation_map < 127] = 0
        segmentation_map[segmentation_map >= 127] = 1
        segmentation_map = Image.fromarray(segmentation_map)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt"
        )
        encoded_inputs["labels"][encoded_inputs["labels"] == 0] = 1
        encoded_inputs["labels"][encoded_inputs["labels"] == 255] = 0

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


def init_segmentation_dataloaders(root_dir: str,
                                  batch_size: int,
                                  num_workers: int):
    feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

    train_dataset = SemanticSegmentationDataset(
        root_dir=root_dir, feature_extractor=feature_extractor
    )
    valid_dataset = SemanticSegmentationDataset(
        root_dir=root_dir, feature_extractor=feature_extractor, train=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, valid_dataloader
