# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Tuple, Union

import tensorflow as tf

from fedcore.neural_compressor.common import logger
from fedcore.neural_compressor.common.base_config import (
    BaseConfig,
    ComposableConfig,
    config_registry,
)
from fedcore.neural_compressor.common.utils import log_quant_execution
from fedcore.neural_compressor.tensorflow.utils import (
    BaseModel,
    KerasModel,
    Model,
    algos_mapping,
)


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    return any(config.name == algo_name for config in configs_mapping.values())


@log_quant_execution
def quantize_model(
    model: Union[str, tf.keras.Model, BaseModel],
    quant_config: BaseConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
):
    """The main entry to quantize model.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.

    Returns:
        q_model: the quantized model.
    """
    q_model = Model(model)
    framework_name = "keras" if isinstance(q_model, KerasModel) else "tensorflow"
    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(
            quant_config, config_registry=registered_configs[framework_name]
        )
        logger.info(
            f"Parsed a config dict to construct the quantization config: {quant_config}."
        )
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.info(f"Quantize model with config: \n {quant_config.to_json_string()} \n")
    # select quantization algo according to config

    model_info = quant_config.get_model_info(model=q_model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            q_model = algo_func(
                q_model, configs_mapping, calib_dataloader, calib_iteration
            )
    return q_model
