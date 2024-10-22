#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Built-in dataloaders, datasets, transforms, filters for multiple framework backends."""


from fedcore.neural_compressor.data.dataloaders import DATALOADERS, DataLoader
from fedcore.neural_compressor.data.dataloaders.dataloader import check_dataloader
from fedcore.neural_compressor.data.dataloaders.default_dataloader import DefaultDataLoader
from fedcore.neural_compressor.data.datasets import (
    Datasets,
    Dataset,
    IterableDataset,
    dataset_registry,
    TensorflowImageRecord,
    COCORecordDataset,
)
from fedcore.neural_compressor.data.filters import FILTERS, Filter, filter_registry, LabelBalanceCOCORecordFilter
from fedcore.neural_compressor.data.transforms import (
    LabelShift,
    BilinearImagenetTransform,
    TensorflowResizeCropImagenetTransform,
)
from fedcore.neural_compressor.data.transforms import ParseDecodeCocoTransform, TensorflowShiftRescale
from fedcore.neural_compressor.data.transforms import TFSquadV1PostTransform, TFSquadV1ModelZooPostTransform
from fedcore.neural_compressor.data.transforms import (
    TRANSFORMS,
    BaseTransform,
    ComposeTransform,
    transform_registry,
    Postprocess,
)
from fedcore.neural_compressor.data.transforms import (
    TensorflowResizeWithRatio,
    ResizeTFTransform,
    RescaleTFTransform,
    NormalizeTFTransform,
)

__all__ = [
    "check_dataloader",
    "DataLoader",
    "DATALOADERS",
    "DefaultDataLoader",
    "Datasets",
    "Dataset",
    "IterableDataset",
    "COCORecordDataset",
    "dataset_registry",
    "TensorflowImageRecord",
    "TRANSFORMS",
    "BaseTransform",
    "ComposeTransform",
    "transform_registry",
    "Postprocess",
    "LabelShift",
    "ResizeTFTransform",
    "RescaleTFTransform",
    "TensorflowShiftRescale",
    "NormalizeTFTransform",
    "ParseDecodeCocoTransform",
    "BilinearImagenetTransform",
    "TensorflowResizeWithRatio",
    "TensorflowResizeCropImagenetTransform",
    "FILTERS",
    "Filter",
    "filter_registry",
    "LabelBalanceCOCORecordFilter",
    "TFSquadV1PostTransform",
    "TFSquadV1ModelZooPostTransform",
]
