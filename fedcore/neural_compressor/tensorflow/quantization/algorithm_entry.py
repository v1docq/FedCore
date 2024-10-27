# Copyright (c) 2023 Intel Corporation
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


from typing import Callable

from fedcore.neural_compressor.common.utils import SMOOTH_QUANT, STATIC_QUANT
from fedcore.neural_compressor.tensorflow.algorithms import KerasAdaptor
from fedcore.neural_compressor.tensorflow.quantization.config import (
    SmoothQuantConfig,
    StaticQuantConfig,
)
from fedcore.neural_compressor.tensorflow.utils import (
    BaseModel,
    KerasModel,
    framework_specific_info,
    register_algo,
)


@register_algo(name=STATIC_QUANT)
def static_quantize_entry(
    model: BaseModel,
    quant_config: StaticQuantConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
):
    """The main entry to apply static quantization.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.

    Returns:
        q_model: the quantized model.
    """
    keras_adaptor = KerasAdaptor(framework_specific_info)
    q_model = keras_adaptor.quantize(
        quant_config, model, calib_dataloader, calib_iteration
    )
    return q_model


@register_algo(name=SMOOTH_QUANT)
def smooth_quant_entry(
    model: BaseModel,
    smooth_quant_config: SmoothQuantConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
):
    assert not isinstance(
        model, KerasModel
    ), "INC don't support smooth quantization for Keras models now."

    from fedcore.neural_compressor.tensorflow.algorithms import SmoothQuant

    converter = SmoothQuant(smooth_quant_config, calib_dataloader, calib_iteration)
    sq_model = converter(model)

    return sq_model
