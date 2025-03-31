import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.dataset.prediction_datasets import TorchVisionDataset, CustomDatasetForImages
from fedcore.architecture.utils.paths import (
    data_path, PROJECT_PATH, PATH_TO_DATA
)
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import (
    DEFAULT_TORCH_DATASET,
)
from fedcore.repository.model_repository import BACKBONE_MODELS
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.object_detection_datasets import *
from fedcore.architecture.dataset.classification_datasets import *


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
                             'val_shuffle': False,
                             'dataset_type': 'directory'}
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
            "yolo_detection": self._yolo_od_loader,
            "yolo_classification": self._yolo_cls_loader,
            "coco_detection": self._coco_od_loader
        }
        return loader_dict[loader_type]

    def _benchmark_loader(self, source):
        train_dataloader, val_dataloader = self._init_pretrain_dataset(dataset=source)
        # torch_model = self._init_pretrain_model(supplementary_data["torch_model"])
        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def _torchvision_loader(self, source, supplementary_data=None):

        train_dataloader, val_dataloader = TorchVisionDataset(source).get_dataloader()
        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def _directory_loader(self, source, supplementary_data=None):
        # load data from directory
        annotations, path_to_model = None, None
        path_to_data = os.path.join(PROJECT_PATH, source)
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

    def _yolo_od_loader(self, source, supplementary_data=None):
        path_to_data = os.path.join(PATH_TO_DATA, source)
        test_dir = os.path.join(path_to_data, "test")
        train_bs = self.loader_params['train_bs']
        val_bs = self.loader_params['val_bs']

        train_dataloader = YOLODataset(path=path_to_data, 
                                 dataset_name=source,
                                 transform=self.transform, 
                                 train=True).get_dataloader(batch_size=train_bs)
        val_dataloader = YOLODataset(path=path_to_data, 
                                  dataset_name=source, 
                                  transform=self.transform,
                                  train=False).get_dataloader(batch_size=val_bs)
        try:
            test_dataloader = UnlabeledDatasetOD(images_path=test_dir).get_dataloader()
        except:
            test_dataloader = None

        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def _yolo_cls_loader(self, source, supplementary_data=None):
        path_to_data = os.path.join(PATH_TO_DATA, source)
        train_dir =  os.path.join(path_to_data, "train")
        val_dir = os.path.join(path_to_data, "val")

        train_dataloader = YOLOClassificationDataset(path=train_dir, 
                                 transform=self.transform, 
                                 train=True).get_dataloader(params=self.loader_params)
        val_dataloader = YOLOClassificationDataset(path=val_dir, 
                                  transform=self.transform,
                                  train=False).get_dataloader(params=self.loader_params)

        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def _coco_od_loader(self, source, supplementary_data=None):
        path_to_data = os.path.join(PATH_TO_DATA, source)
        test_dir = os.path.join(path_to_data, "test")
        train_bs = self.loader_params['train_bs']
        val_bs = self.loader_params['val_bs']

        train_dataloader = COCODataset(path=path_to_data, 
                                 dataset_name=source, 
                                 train=True).get_dataloader(batch_size=train_bs)
        val_dataloader = COCODataset(path=path_to_data, 
                                  dataset_name=source, 
                                  train=False).get_dataloader(batch_size=val_bs)
        
        try:
            test_dataloader = UnlabeledDataset(images_path=test_dir).get_dataloader()
        except:
            test_dataloader = None
            
        num_classes = len(train_dataloader.dataset.classes) if hasattr(train_dataloader.dataset, 'classes') else 1
        self.train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            num_classes=num_classes,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )
        self.train_data.supplementary_data.is_auto_preprocessed = True

    def load_data(self, loader_type: str = None):
        self._get_loader(loader_type)(source=self.source)
        return self.train_data
