#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
# pylint:disable=import-error

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from fedcore.neural_compressor.common.base_config import (
    BaseConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from fedcore.neural_compressor.common.utils import (
    AWQ,
    DEFAULT_WHITE_LIST,
    FP8_QUANT,
    GPTQ,
    HQQ,
    OP_NAME_OR_MODULE_TYPE,
    RTN,
    SMOOTH_QUANT,
    STATIC_QUANT,
    TEQ,
)
from fedcore.neural_compressor.torch.utils import is_hpex_available, logger
from fedcore.neural_compressor.torch.utils.constants import (
    PRIORITY_AWQ,
    PRIORITY_GPTQ,
    PRIORITY_HQQ,
    PRIORITY_RTN,
    PRIORITY_TEQ,
)

__all__ = [
    "RTNConfig",
    "get_default_rtn_config",
    "GPTQConfig",
    "get_default_gptq_config",
    "HQQConfig",
    "get_default_hqq_config",
]


FRAMEWORK_NAME = "torch"
DTYPE_RANGE = Union[torch.dtype, List[torch.dtype]]


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


######################## RNT Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=RTN, priority=PRIORITY_RTN)
class RTNConfig(BaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    name = RTN
    params_list = [
        "dtype",
        "bits",
        "use_sym",
        "group_size",
        "group_dim",
        "use_full_range",
        "use_mse_search",
        "export_compressed_model",
        # layer wise params
        "use_layer_wise",
        "model_path",
        # double quant
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        group_dim: int = 1,
        use_full_range: bool = False,
        use_mse_search: bool = False,
        export_compressed_model: bool = False,
        # layer wise
        use_layer_wise: bool = False,
        model_path: str = "",
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = False,
        double_quant_group_size: int = 256,
        # Tuning space
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init RTN weight-only quantization config.

        Args:
            dtype (str): Data type for weights. Default is "int".
            bits (int): Number of bits used to represent weights. Default is 4.
            use_sym (bool): Indicates whether weights are symmetric. Default is True.
            group_size (int): Size of weight groups. Default is 32.
            group_dim (int): Dimension for grouping. Default is 1.
            use_full_range (bool): Enables full range for activations. Default is False.
            use_mse_search (bool): Enables mean squared error (MSE) search. Default is False.
            export_compressed_model (bool): Enables return model in int format or not. Defaults to False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            model_path (str): Model path that is used to load state_dict per layer.
            use_double_quant (bool): Enables double quantization. Default is False.
            double_quant_dtype (str): Data type for double_quant scale. Default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale. Default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric. Default is True.
            double_quant_group_size (int): Size of double_quant groups. Default is 32.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.group_dim = group_dim
        self.use_full_range = use_full_range
        self.use_mse_search = use_mse_search
        self.export_compressed_model = export_compressed_model
        self.use_layer_wise = use_layer_wise
        self.model_path = model_path
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        self._post_init()  # initialize global & local configuration

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        linear_rtn_config = RTNConfig(
            dtype=["int", "int8", "int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"],
            bits=[4, 1, 2, 3, 5, 6, 7, 8],
            use_sym=[True, False],
            group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
            group_dim=[1, 0],
            use_full_range=[False, True],
            use_mse_search=[False, True],
            use_layer_wise=[False, True],
            export_compressed_model=[False, True],
            use_double_quant=[False, True],
            double_quant_bits=[4, 1, 2, 3, 5, 6, 7, 8],
            double_quant_dtype=["int"],
            double_quant_use_sym=[True, False],
            double_quant_group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
        )
        operators = [torch.nn.Linear]
        supported_configs.append(
            OperatorConfig(config=linear_rtn_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "RTNConfig", List["RTNConfig"]]:
        return RTNConfig(
            dtype=["int4", "nf4"],
            use_sym=[True, False],
            group_size=[32, 128],
            use_mse_search=[False, True],
        )


def get_default_rtn_config() -> RTNConfig:
    """Generate the default rtn config.

    Returns:
        the default rtn config.
    """
    return RTNConfig()


def get_default_double_quant_config(type="BNB_NF4"):
    from fedcore.neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

    assert type in DOUBLE_QUANT_CONFIGS, "Supported double quant configs: {}".format(
        list(DOUBLE_QUANT_CONFIGS.keys())
    )
    return RTNConfig.from_dict(DOUBLE_QUANT_CONFIGS[type])


######################## GPTQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=GPTQ, priority=PRIORITY_GPTQ)
class GPTQConfig(BaseConfig):
    """Config class for GPTQ.

    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
    https://arxiv.org/abs/2210.17323
    """

    name = GPTQ
    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "use_sym",
        "group_size",
        "use_mse_search",
        "export_compressed_model",
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # layer wise params
        "use_layer_wise",
        "model_path",
        # gptq params
        "act_order",
        "percdamp",
        "block_size",
        "static_groups",
    ]

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        use_mse_search: bool = False,
        export_compressed_model: bool = False,
        # layer wise
        use_layer_wise: bool = False,
        model_path: str = "",
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = False,
        double_quant_group_size: int = 256,
        # gptq params
        act_order: bool = False,
        percdamp: float = 0.01,
        block_size: int = 2048,
        static_groups: bool = False,
        # Tuning space
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init RTN weight-only quantization config.

        Args:
            dtype (str): Data type for weights. Default is "int".
            bits (int): Number of bits used to represent weights. Default is 4.
            use_sym (bool): Indicates whether weights are symmetric. Default is True.
            group_size (int): Size of weight groups. Default is 32.
            use_mse_search (bool): Enables mean squared error (MSE) search. Default is False.
            export_compressed_model (bool): Enables return model in int format or not. Defaults to False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            model_path (str): Model path that is used to load state_dict per layer.
            use_double_quant (bool): Enables double quantization. Default is False.
            double_quant_dtype (str): Data type for double_quant scale. Default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale. Default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric. Default is True.
            double_quant_group_size (int): Size of double_quant groups. Default is 32.
            act_order (bool): Whether to sort Hessian's diagonal values to rearrange channel-wise
                              quantization order. Default is False.
            percdamp (float): Percentage of Hessian's diagonal values' average, which will be added to
                              Hessian's diagonal to increase numerical stability. Default is 0.01.
            block_size (int): Execute GPTQ quantization per block, block shape = [C_out, block_size].
                              Default is 128.
            static_groups (bool): Whether to calculate group wise quantization parameters in advance.
                                  This option mitigate actorder's extra computational requirements.
                                  Default is False.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.use_mse_search = use_mse_search
        self.export_compressed_model = export_compressed_model
        # layer wise
        self.use_layer_wise = use_layer_wise
        self.model_path = model_path
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        # gptq
        self.act_order = act_order
        self.percdamp = percdamp
        self.block_size = block_size
        self.static_groups = static_groups
        self._post_init()  # initialize global & local configuration

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_gptq_config = GPTQConfig()
        operators = [torch.nn.Linear]
        supported_configs.append(
            OperatorConfig(config=linear_gptq_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "GPTQConfig", List["GPTQConfig"]]:
        # TODO fwk owner needs to update it.
        return GPTQConfig(act_order=[True, False], use_sym=[False, True])


def get_default_gptq_config() -> GPTQConfig:
    """Generate the default gptq config.

    Returns:
        the default gptq config.
    """
    return GPTQConfig()


######################## AWQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=AWQ, priority=PRIORITY_AWQ)
class AWQConfig(BaseConfig):
    """Config class for AWQ.

    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.
    https://arxiv.org/abs/2306.00978
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "group_size",
        "group_dim",
        "use_sym",
        "use_full_range",
        "use_mse_search",
        "use_layer_wise",
        "export_compressed_model",
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # AWQ params
        "use_auto_scale",
        "use_auto_clip",
        "folding",
    ]
    name = AWQ

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        group_dim: int = 1,
        use_full_range: bool = False,
        use_mse_search: bool = False,
        use_layer_wise: bool = False,
        export_compressed_model: bool = False,
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = True,
        double_quant_group_size: int = 256,
        # awq
        use_auto_scale: bool = True,
        use_auto_clip: bool = True,
        folding: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init AWQ weight-only quantization config.

        Args:
            dtype (str): Data type for weights, default is "int".
            bits (int): Number of bits used to represent weights, default is 4.
            use_sym (bool): Indicates whether weights are symmetric, default is True.
            group_size (int): Size of weight groups, default is 32.
            group_dim (int): Dimension for grouping, default is 1.
            use_full_range (bool): Enables full range for activations, default is False.
            use_mse_search (bool): Enables mean squared error (MSE) search, default is False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            export_compressed_model (bool): Enables return model in int format or not. Defaults to False.
            use_double_quant (bool): Enables double quantization, default is False.
            double_quant_dtype (str): Data type for double_quant scale, default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale, default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric, default is True.
            double_quant_group_size (int): Size of double_quant groups, default is 32.
            use_auto_scale (bool): Enables best scales search based on activation distribution, default is True.
            use_auto_clip (bool):  Enables clip range search. Defaults to True.
            folding(bool): Allow insert mul before linear when the scale cannot be absorbed by last layer,
              default is False.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.group_dim = group_dim
        self.use_full_range = use_full_range
        self.use_mse_search = use_mse_search
        self.use_layer_wise = use_layer_wise
        self.export_compressed_model = export_compressed_model
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        self.use_auto_scale = use_auto_scale
        self.use_auto_clip = use_auto_clip
        self.folding = folding
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_awq_config = AWQConfig()
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(
            OperatorConfig(config=linear_awq_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "AWQConfig", List["AWQConfig"]]:
        # TODO fwk owner needs to update it.
        return AWQConfig(bits=[4, 6])


def get_default_awq_config() -> AWQConfig:
    """Generate the default awq config.

    Returns:
        the default awq config.
    """
    return AWQConfig()


######################## TEQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=TEQ, priority=PRIORITY_TEQ)
class TEQConfig(BaseConfig):
    """Config class for TEQ.

    TEQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.
    https://arxiv.org/abs/2306.00978
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "group_size",
        "group_dim",
        "use_sym",
        "use_full_range",
        "use_mse_search",
        "use_layer_wise",
        "export_compressed_model",
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # TEQ params
        "absorb_to_layer",
        "folding",
    ]
    name = TEQ

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        group_dim: int = 1,
        use_full_range: bool = False,
        use_mse_search: bool = False,
        use_layer_wise: bool = False,
        export_compressed_model: bool = False,
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = True,
        double_quant_group_size: int = 256,
        # teq
        absorb_to_layer: dict = {},
        folding: bool = True,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init TEQ weight-only quantization config.

        Args:
            dtype (str): Data type for weights, default is "int".
            bits (int): Number of bits used to represent weights, default is 4.
            use_sym (bool): Indicates whether weights are symmetric, default is True.
            group_size (int): Size of weight groups, default is 32.
            group_dim (int): Dimension for grouping, default is 1.
            use_full_range (bool): Enables full range for activations, default is False.
            use_mse_search (bool): Enables mean squared error (MSE) search, default is False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            export_compressed_model (bool): Enables return model in int format or not. Defaults to False.
            use_double_quant (bool): Enables double quantization, default is False.
            double_quant_dtype (str): Data type for double_quant scale, default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale, default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric, default is True.
            double_quant_group_size (int): Size of double_quant groups, default is 32.
            absorb_to_layer (bool): The layer dict that scale can be absorbed, default is {}.
            folding(bool): Allow insert mul before linear when the scale cannot be absorbed by last layer,
              default is False.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.group_dim = group_dim
        self.use_full_range = use_full_range
        self.use_mse_search = use_mse_search
        self.use_layer_wise = use_layer_wise
        self.export_compressed_model = export_compressed_model
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        self.absorb_to_layer = absorb_to_layer
        self.folding = folding
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_teq_config = TEQConfig()
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(
            OperatorConfig(config=linear_teq_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "TEQConfig", List["TEQConfig"]]:
        # TODO fwk owner needs to update it.
        return TEQConfig(bits=[4, 6])


def get_default_teq_config() -> TEQConfig:
    """Generate the default teq config.

    Returns:
        the default teq config.
    """
    return TEQConfig()


######################## Static Quant Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=STATIC_QUANT)
class StaticQuantConfig(BaseConfig):
    """Config class for static quantization."""

    name = STATIC_QUANT
    params_list = [
        "w_dtype",
        "w_sym",
        "w_granularity",
        "w_algo",
        "act_dtype",
        "act_sym",
        "act_granularity",
        "act_algo",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        w_dtype: str = "int8",
        w_sym: bool = True,
        w_granularity: str = "per_channel",
        w_algo: str = "minmax",
        act_dtype: str = "uint8",
        act_sym: bool = False,
        act_granularity: str = "per_tensor",
        act_algo: str = "kl",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init Static Quant Configs."""
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.w_sym = w_sym
        self.w_granularity = w_granularity
        self.w_algo = w_algo
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algo = act_algo
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_static_config = StaticQuantConfig()
        operators = [torch.nn.Linear]
        supported_configs.append(
            OperatorConfig(config=linear_static_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(
        model: torch.nn.Module, example_inputs
    ) -> List[Tuple[str, Callable]]:
        from fedcore.neural_compressor.torch.algorithms.static_quant import (
            get_quantizable_ops_recursively,
        )

        model_info, _, _, _ = get_quantizable_ops_recursively(
            model, example_inputs=example_inputs
        )
        return model_info

    @classmethod
    def get_config_set_for_tuning(
        cls,
    ) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:
        return StaticQuantConfig(act_sym=[True, False], act_algo=["kl", "minmax"])


def get_default_static_config() -> StaticQuantConfig:
    """Generate the default static quant config.

    Returns:
        the default static quant config.
    """
    return StaticQuantConfig()


######################## Smooth Quant Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=SMOOTH_QUANT)
class SmoothQuantConfig(BaseConfig):
    """Config class for smooth quantization."""

    name = SMOOTH_QUANT
    params_list = [
        "w_dtype",
        "w_sym",
        "w_granularity",
        "w_algo",
        "act_dtype",
        "act_sym",
        "act_granularity",
        "act_algo",
        "alpha",
        "folding",
        "scale_sharing",
        "auto_alpha_args",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        w_dtype: str = "int8",
        w_sym: bool = True,
        w_granularity: str = "per_channel",
        w_algo: str = "minmax",
        act_dtype: str = "uint8",
        act_sym: bool = False,
        act_granularity: str = "per_tensor",
        act_algo: str = "kl",
        alpha: float = 0.5,
        folding: bool = False,
        # below for autotune
        scale_sharing: bool = False,
        init_alpha: float = 0.5,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        alpha_step: float = 0.1,
        shared_criterion: str = "max",
        enable_blockwise_loss: bool = False,
        auto_alpha_args: dict = None,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init SmoothQuant Configs."""
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.w_sym = w_sym
        self.w_granularity = w_granularity
        self.w_algo = w_algo
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algo = act_algo
        self.alpha = alpha
        self.folding = folding
        # below for autotune
        self.scale_sharing = scale_sharing
        self.init_alpha = init_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.shared_criterion = shared_criterion
        self.enable_blockwise_loss = enable_blockwise_loss
        self.auto_alpha_args = {
            "init_alpha": self.init_alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_step": self.alpha_step,
            "shared_criterion": self.shared_criterion,
            "enable_blockwise_loss": self.enable_blockwise_loss,
        }
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_sq_config = SmoothQuantConfig()
        operators = [torch.nn.Linear]
        supported_configs.append(
            OperatorConfig(config=linear_sq_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(
        cls,
    ) -> Union[None, "SmoothQuantConfig", List["SmoothQuantConfig"]]:
        # TODO fwk owner needs to update it.
        return SmoothQuantConfig(alpha=[0.1, 0.5])


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default smoothquant config.

    Returns:
        the default smoothquant config.
    """
    return SmoothQuantConfig()


######################## HQQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=HQQ, priority=PRIORITY_HQQ)
class HQQConfig(BaseConfig):
    # Half-Quadratic Quantization (HQQ), more details:
    # Blog: https://mobiusml.github.io/hqq_blog/
    # Code: https://github.com/mobiusml/hqq

    name = HQQ
    params_list = [
        "bits",
        "group_size",
        "quant_zero",
        "quant_scale",
        "scale_quant_group_size",
        "skip_lm_head",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        quant_zero: bool = True,
        quant_scale: bool = False,
        scale_quant_group_size: int = 128,
        skip_lm_head: bool = True,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        super().__init__(white_list=white_list)
        self.bits = bits
        self.group_size = group_size
        self.quant_zero = quant_zero
        self.quant_scale = quant_scale
        self.scale_quant_group_size = scale_quant_group_size
        self.skip_lm_head = skip_lm_head
        self._post_init()

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        return filter_result

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        # TODO: to be refined
        supported_configs = []
        linear_hqq_config = HQQConfig()
        operators = [torch.nn.Linear]
        supported_configs.append(
            OperatorConfig(config=linear_hqq_config, operators=operators)
        )
        cls.supported_configs = supported_configs

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "HQQConfig", List["HQQConfig"]]:
        return HQQConfig(bits=[4, 8])


def get_default_hqq_config() -> HQQConfig:
    """Generate the default HQQ config.

    Returns:
        the default HQQ config.
    """
    return HQQConfig()


######################## FP8 Config ###############################
if is_hpex_available():

    @register_config(framework_name=FRAMEWORK_NAME, algo_name=FP8_QUANT)
    class FP8QConfig(BaseConfig):
        """Config class for FP8 quantization."""

        name = FP8_QUANT
        supported_configs: List[OperatorConfig] = []
        params_list = [
            "weight_dtype",
            "act_dtype",
            "act_algo",
            "approach",
            "device",
        ]

        def __init__(
            self,
            weight_dtype: DTYPE_RANGE = torch.float8_e4m3fn,
            act_dtype: DTYPE_RANGE = torch.float8_e4m3fn,
            act_algo: Union[str, List[str]] = "minmax",
            approach: Union[str, List[str]] = "static",
            device: Union[str, List[str]] = "hpu",
            white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        ):
            """Init FP8 config.

            Args:
            """
            super().__init__(white_list=white_list)
            self.weight_dtype = weight_dtype
            self.act_dtype = act_dtype
            self.act_algo = act_algo
            self.approach = approach
            self.device = device
            self._post_init()

        @classmethod
        def register_supported_configs(cls) -> List[OperatorConfig]:
            supported_configs = []
            fp8_config = FP8QConfig(
                weight_dtype=[torch.float8_e5m2, torch.float8_e4m3fn],
                act_dtype=[torch.float8_e5m2, torch.float8_e4m3fn],
                act_algo=["minmax", "kl"],
                approach=["static", "dynamic"],
                device=["hpu"],
            )
            from fedcore.neural_compressor.torch.algorithms.habana_fp8 import white_list

            operators = white_list
            supported_configs.append(
                OperatorConfig(config=fp8_config, operators=operators)
            )
            cls.supported_configs = supported_configs

        @staticmethod
        def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
            from fedcore.neural_compressor.torch.algorithms.habana_fp8 import white_list

            filter_result = []
            for op_name, module in model.named_modules():
                if isinstance(module, white_list):
                    pair = (op_name, type(module).__name__)
                    filter_result.append(pair)
            logger.debug(f"Get model info: {filter_result}")
            return filter_result

        @classmethod
        def get_config_set_for_tuning(
            cls,
        ) -> Union[None, "FP8QConfig", List["FP8QConfig"]]:
            # TODO fwk owner needs to update it.
            return FP8QConfig(act_dtype=[torch.float8_e4m3fn])

    def get_default_fp8_qconfig() -> FP8QConfig:
        """Generate the default gptq config.

        Returns:
            the default gptq config.
        """
        return FP8QConfig()

    ##################### Algo Configs End ###################################


register_supported_configs_for_fwk(fwk_name=FRAMEWORK_NAME)


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    registered_configs = config_registry.get_all_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})
