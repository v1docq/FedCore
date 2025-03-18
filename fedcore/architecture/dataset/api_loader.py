import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from fedcore.architecture.dataset.prediction_datasets import TorchVisionDataset
from fedcore.architecture.utils.paths import data_path, PROJECT_PATH
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import (
    default_device,
    DEFAULT_TORCH_DATASET,
)
from fedcore.repository.model_repository import BACKBONE_MODELS


class ApiLoader:
    def __init__(self, load_source, loader_params: dict = None):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                            )
        self.source = load_source
        if loader_params is None:
            loader_params = {'train_bs': 64,
                             'val_bs': 100,
                             'train_shuffle': True,
                             'val_shuffle': False}
        self.loader_params = loader_params

    def _init_pretrain_dataset(self, dataset: str = "CIFAR10"):
        train_dataset = DEFAULT_TORCH_DATASET[dataset](
            data_path(dataset), train=True, download=True, transform=self.transform
        )
        val_dataset = DEFAULT_TORCH_DATASET[dataset](
            data_path(dataset), train=False, download=True, transform=self.transform
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.loader_params['train_bs'],
            shuffle=self.loader_params['train_shuffle'],
            num_workers=1
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.loader_params['val_bs'],
            shuffle=self.loader_params['val_shuffle'],
            num_workers=1
        )
        return train_dataloader, val_dataloader

    def _init_pretrain_model(self, model_name):
        model = BACKBONE_MODELS[model_name](pretrained=True).to(default_device())
        return model

    def _get_loader(self, loader_type: str):
        loader_dict = {
            "torchvision": self._torchvision_loader,
            "benchmark": self._benchmark_loader,
            "directory": self._directory_loader,
        }
        return loader_dict[loader_type]

    def _benchmark_loader(self, path):
        train_dataloader, val_dataloader = self._init_pretrain_dataset(dataset=path)
        # torch_model = self._init_pretrain_model(supplementary_data["torch_model"])
        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def _torchvision_loader(self, supplementary_data, path):

        train_dataloader, val_dataloader = TorchVisionDataset(path).get_dataloader()
        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def _directory_loader(self, supplementary_data, path):
        # load data from directory
        annotations, path_to_model = None, None
        path_to_data = os.path.join(PROJECT_PATH, path)
        dir_list = os.listdir(path_to_data)
        for x in dir_list:
            if x.__contains__("dataset"):
                directory = os.path.join(path_to_data, x)
            elif x.__contains__("model"):
                model_dir = os.path.join(path_to_data, x)
                _ = [
                    y
                    for y in os.listdir(model_dir)
                    if y.__contains__(".pt") or y.__contains__(".h5")
                ][0]
                path_to_model = os.path.join(model_dir, _)
            elif x.__contains__("txt"):
                annotations = os.path.join(path_to_data, x)
        if path_to_model is None and supplementary_data is not None:
            path_to_model = supplementary_data["model_name"]
        # self.train_data = DataCheck(
        #     input_data=(directory, annotations, path_to_model),
        #     cv_dataset=self.cv_dataset,
        # ).check_input_data()

    def load_data(self, loader_type: str = None):
        self._get_loader(loader_type)(self.source)
        return self.train_data
