"""Intel® Neural Compressor: An open-source Python library supporting common model."""

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

from .model import Model
from .dataloader import DataLoader, _generate_common_dataloader
from .postprocess import Postprocess
from .metric import Metric

__all__ = [
    "Model",
    "DataLoader",
    "Postprocess",
    "Metric",
    "_generate_common_dataloader",
]
