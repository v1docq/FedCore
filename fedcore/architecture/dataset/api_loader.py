from functools import partial

from torch.utils.data import DataLoader

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.dataset.datasets_from_source import AbstractDataset, ObjectDetectionDataset, TimeSeriesDataset
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import (
    DEFAULT_TORCH_DATASET,
)
from fedcore.repository.model_repository import BACKBONE_MODELS
from fedcore.architecture.dataset.task_specified.object_detection_datasets import *
from fedcore.architecture.dataset.task_specified.classification_datasets import *
from torch.utils.data import random_split
from fedot.core.repository.tasks import Task, TaskTypesEnum


class ApiLoader:
    def __init__(self, load_source, loader_params: dict = None):
        transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if 'resize' in loader_params:
            transforms_.insert(1, 
                transforms.Resize(loader_params['resize'])
            )
        self.transform = transforms.Compose(transforms_)
        self.source = load_source

        self.default_loader_params = {'batch_size': 8,
                                      'shuffle': True,
                                      'num_workers': 1}
        self.loader_params = loader_params

        self.torch_dataset_dict = {'object_detection': ObjectDetectionDataset,
                                   'image': AbstractDataset,
                                   'time_series': TimeSeriesDataset,
                                   'benchmark': partial(lambda bench_name: DEFAULT_TORCH_DATASET[bench_name])}

    def _update_loader_params(self):
        self.loader_params = {**self.default_loader_params, **self.loader_params}
        unsupported_keys = ['split_ratio', 'data_type', 'is_train']
        for key in unsupported_keys:
            if key in self.loader_params.keys():
                del self.loader_params[key]

    def _init_pretrain_dataset(self, dataset: str = "CIFAR10"):
        train_dataset = DEFAULT_TORCH_DATASET[dataset](data_path(dataset), train=True,
                                                       download=True, transform=self.transform)
        return train_dataset

    def _init_pretrain_model(self, model_name):
        model = BACKBONE_MODELS[model_name](pretrained=True).to(default_device())
        return model

    def _convert_to_fedcore(self, torch_dataset) -> CompressionInputData:
        if 'split_ratio' in self.loader_params:
            train_dataset, val_dataset = random_split(torch_dataset, self.loader_params['split_ratio'])
        else:
            train_dataset, val_dataset = torch_dataset, torch_dataset

        self._update_loader_params()
        from torch.ao.quantization.utils import _normalize_kwargs
        train_dataloader = DataLoader(dataset=train_dataset, 
                                      **_normalize_kwargs(DataLoader.__init__, self.loader_params)
        )
        val_dataloader = DataLoader(dataset=val_dataset,**_normalize_kwargs(DataLoader.__init__, self.loader_params)
        )
        sample = next(iter(torch_dataset))[0]
        num_classes = len(torch_dataset.classes) if torch_dataset.classes is not None else 1
        input_dim = sample.shape[1] if len(sample.shape) > 2 else sample.shape[0]
        task = Task(TaskTypesEnum.classification) if num_classes != 1 else Task(TaskTypesEnum.regression)
        fedcore_train_data = CompressionInputData(
            features=np.zeros((2, 2)),
            input_dim=input_dim,
            num_classes=num_classes,
            task=task,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
        )
        fedcore_train_data.supplementary_data.is_auto_preprocessed = True
        return fedcore_train_data

    def load_data(self, loader_type: str = None) -> CompressionInputData:
        torch_dataset = self.torch_dataset_dict[loader_type](self.source)
        try:
            if loader_type.__contains__('benchmark'):
                if self.source.__contains__('CIFAR'):
                    torch_dataset = torch_dataset(data_path(self.source),
                                                train=self.loader_params['is_train'],
                                                download=True,
                                                transform=self.transform)
                else:
                    torch_dataset = torch_dataset(data_path(self.source),
                                                download=True,
                                                transform=self.transform)
        except:
            torch_dataset = torch_dataset(data_path(self.source),
                                        download=False,
                                        transform=self.transform)
        fedcore_data = self._convert_to_fedcore(torch_dataset)
        return fedcore_data
