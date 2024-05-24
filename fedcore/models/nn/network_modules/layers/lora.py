import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Implements a Layer-wise Representation Adaptation (LoRA) which allows adapting pre-trained models
    to new tasks efficiently with minimal trainable parameters. By injecting trainable weights into
    the existing architecture at inference, LoRA aims to preserve the original model's knowledge while
    enabling specificity to the new task.

    LoRALayer can be applied to various base layers like Linear, Conv2D, and Embedding. This flexibility
    makes it relevant for tasks including but not limited to natural language processing, image
    recognition, and recommendation systems where efficient adaptation of large pre-trained models is desired.

    Adapter weights are trained to minimally perturb the original pre-trained weights, aiming to achieve
    efficient fine-tuning. This is particularly useful in scenarios where computational resources for training
    are limited, or when the new task data is scarce.

    Parameters:
        base_layer (nn.Module): The base layer to which LoRA adaptation is to be applied. It can be of type nn.Linear,
                                nn.Conv2d, or nn.Embedding.

        **kwargs: Arbitrary keyword arguments. This can include parameters for initializing the LoRA weights
                  and configuring the behavior of the LoRA layer.

    Args:
        adapter_layer_names: Names of the layers that may contain trainable adapter weights.
        other_param_names: Names of other parameters involved in LoRA configuration.
        r: Dictionary mapping adapter names to their corresponding ranks (dimensions of the trainable weights).
        lora_alpha: Dictionary mapping adapter names to their LoRA alpha values, influencing the learning amplitude.
        scaling: Dictionary mapping adapter names to scaling factors applied to the LoRA weights.
        lora_dropout: Contains dropout layers to be applied to LoRA augmentations, keyed by adapter names.
        lora_A: ModuleDict containing the `A` LoRA weights for input transformation.
        lora_B: ModuleDict containing the `B` LoRA weights for output transformation.
        lora_embedding_A: ParameterDict containing embedding-specific `A` weights.
        lora_embedding_B: ParameterDict containing embedding-specific `B` weights.
        _disable_adapters: Flag to disable LoRA adaptation.
        merged_adapters: Tracks adapters that have been merged for optimization.
        _caches: Used internally for caching computations across LoRA layers.
        kwargs: Stores all keyword arguments passed during initialization.

    Methods:
        update_layer(...): Updates or adds adapter weights to the layer.
        reset_lora_parameters(...): Resets LoRA parameters based on specified initialization strategies.
        set_scale(adapter, scale): Sets the scaling factor for a given LoRA adapter.
        scale_layer(scale): Scales the entire LoRA layer by a given factor.
        unscale_layer(scale=None): Reverses the effect of scale_layer by the given scale factor, or resets scaling.
        _check_forward_args(x, *args, **kwargs): Validates forward pass arguments for compatibility with LoRA configuration.
        _mixed_batch_forward(x, *args, adapter_names, **kwargs): Handles forward pass across mixed batches with specified adapters.

    Example:
        # Define a base layer (e.g., nn.Linear)
        base_linear_layer = nn.Linear(in_features=768, out_features=768)

        # Initialize LoRALayer with the base layer
        lora_layer = LoRALayer(base_layer=base_linear_layer, r=4)

        # Assuming x is the input tensor to the model
        adapted_output = lora_layer(x)

    Note:
        LoRALayer is part of a broader effort to make large model adaptation more efficient and accessible. It requires
        careful selection of hyperparameters like `r` (rank) and `lora_alpha` to balance between efficiency and task
        performance. While it can provide significant improvements in adaptation scenarios, the choice to use LoRA
        should be informed by the specific task's requirements and available computational resources.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim

        self.in_features = in_features
        self.out_features = out_features

        def update_layer(
                self,
                adapter_name,
                r,
                lora_alpha,
                lora_dropout,
                init_lora_weights,
        ):
            # This code works for linear layers, override for other layer types
            if r <= 0:
                raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
            # Actual trainable parameters
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

            if init_lora_weights == "loftq":
                self.loftq_init(adapter_name)
            elif init_lora_weights:
                self.reset_lora_parameters(adapter_name, init_lora_weights)

            self.set_adapter(self.active_adapters)

        def reset_lora_parameters(self, adapter_name, init_lora_weights):
            if init_lora_weights is False:
                return

            if adapter_name in self.lora_A.keys():
                if init_lora_weights is True:
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights=}")
                nn.init.zeros_(self.lora_B[adapter_name].weight)
            if adapter_name in self.lora_embedding_A.keys():
                # initialize a the same way as the default for nn.linear and b to zero
                nn.init.zeros_(self.lora_embedding_A[adapter_name])
                nn.init.normal_(self.lora_embedding_B[adapter_name])

        def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
            # calculate L2 norm of weight matrix, column-wise
            weight = weight + scaling * lora_weight
            weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
            return weight_norm

        def _cache_store(self, key: str, value: Any) -> None:
            self._caches[key] = value

        def _cache_pop(self, key: str) -> Any:
            value = self._caches.pop(key)
            return value

        def set_scale(self, adapter, scale):
            if adapter not in self.scaling:
                # Ignore the case where the adapter is not in the layer
                return
            self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

        def scale_layer(self, scale: float) -> None:
            if scale == 1:
                return

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                self.scaling[active_adapter] *= scale

        def unscale_layer(self, scale=None) -> None:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                if scale is None:
                    self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
                else:
                    self.scaling[active_adapter] /= scale

        def _check_forward_args(self, x, *args, **kwargs):
            """Check if the arguments are compatible with the configs and state of the model"""
            adapter_names = kwargs.get("adapter_names", None)
            if adapter_names is None:
                return

            if len(x) != len(adapter_names):
                msg = (
                    "Length of `adapter_names` should be the same as the number of inputs, but got "
                    f"{len(adapter_names)} and {len(x)} respectively."
                )
                raise ValueError(msg)

            if self.merged:
                # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
                # adapters. Therefore, it is better to raise an error in this case.
                msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
                raise ValueError(msg)

            unique_adapters = set(self.active_adapters)
            for adapter_name in unique_adapters:
                if self.use_dora.get(adapter_name, False):
                    msg = "Cannot pass `adapter_names` when DoRA is enabled."
                    raise ValueError(msg)

        def _mixed_batch_forward(
                self,
                x: torch.Tensor,
                *args: Any,
                adapter_names: list[str],
                **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
                lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
                result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

            return result


class Linear(nn.Module):
    """
    Represents a Linear LoRA (Layer-wise Representation Adaptation) layer that facilitates an efficient adaptation
    mechanism to the dense layers in the neural network models. Linear LoRA is a method to fine-tune only a small
    fraction of the model parameters, specifically designed to provide an efficient alternative to full fine-tuning.

    Linear LoRA adapts a dense layer by adding low-rank perturbations, which allows the base model to quickly
    adapt to new tasks. It introduces additional parameters known as A and B that are used to create the rank-r
    update to the original weight matrix.

    This technique is especially valuable in scenarios where computational resources are limited, and
    during transfer learning where the adapted model needs to maintain most of the base model's knowledge
    while gaining new capabilities for the task at hand.

    Args:
        base_layer: The original dense layer to which LoRA will be applied.
        adapter_name: A unique name assigned to this LoRA adapter for identification.
        r: The rank of the update matrix, which determines the number of trainable parameters to add.
        lora_alpha: A scaling factor that adjusts the strength of the LoRA update.
        lora_dropout: Dropout rate applied to the random feature mapping in LoRA.
        fan_in_fan_out: Set True if the layer stores weights in (fan_in, fan_out) order.
        is_target_conv_1d_layer: Set True if the target layer is a 1-dimensional convolutional layer.
        init_lora_weights: Strategy to initialize the LoRA weights. Can be a boolean
                                                         indicating whether to use default initialization or a string
                                                         specifying a particular initialization method.
        **kwargs: Arbitrary keyword arguments that are passed to the underlying LoRALayer.

    Attributes:
        fan_in_fan_out: Indicates if the weights order is fan-in fan-out, affecting how the delta weight is computed.
        _active_adapter: The name of the currently active adapter layer.
        is_target_conv_1d_layer: Indicates whether the target layer to replace is a 1-dimensional convolution layer.

    Methods:
        merge(safe_merge=False, adapter_names=None): Integrates the changes from the adaptable weights (LoRA layers) into
                                                     the base layer weights. Optionally can perform a safe merge to check
                                                     for NaNs.
        unmerge(): Reverts the base weights back to their state before any LoRA merge operations were performed.
        get_delta_weight(adapter): Computes the change needed to the base weights for the specified adapter.
        forward(x, *args, **kwargs): Performs a forward pass on the input tensor `x` using the adapted dense layer.

    Example:
        # Define a base dense layer
        base_layer = nn.Linear(in_features=512, out_features=512)

        # Wrap the base layer with Linear LoRA
        lora_linear = Linear(base_layer, adapter_name='example', r=4)

        # Forward pass through the LoRA layer
        adapted_output = lora_linear(input_tensor)

    Note:
        Linear LoRA provides an effective approach to adapt large-scale models like BERT for specific tasks without
        the need for extensive retraining. It's crucial to choose appropriate LoRA parameters based on the specific
        requirements of the task and the computational efficiency desired.
    """
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)

                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)

                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoRALayer):
    """
    Enhances an existing Embedding layer with Layer-wise Representation Adaptation (LoRA), enabling
    efficient adaptation for large-scale models. This class applies the LoRA mechanism specifically to
    embedding matrices, facilitating the adjustment of embeddings with minimal additional trainable parameters.

    LoRA in an embedding context introduces low-rank matrices that perturb the original embedding weights.
    The adaptation occurs through learned parameters that are optimized to capture task-relevant nuances without
    substantial modification to the original pre-trained embeddings.

    It is designed to work with pre-trained embeddings and is especially beneficial in scenarios where
    embeddings form a significant part of the model's parameters, such as large language models. LoRA
    allows these models to be fine-tuned for new tasks with a reduced risk of catastrophic forgetting.

    Args:
        base_layer: The original embedding layer to which LoRA will be applied.
        adapter_name: A unique identifier for the adapter layer.
        r: The rank of the update matrices which is a key factor determining the number of LoRA parameters.
        lora_alpha: A hyperparameter controlling the learning rate specific to the LoRA adapter.
        lora_dropout: Dropout rate to apply to the LoRA weight matrices during training.
        init_lora_weights: Initialization scheme for LoRA weights. It could be a boolean
                                                        to use the default scheme or a string to specify a method.
        **kwargs: Arbitrary keyword arguments to further customize the LoRA layer.

    Methods:
        update_layer(...): Initializes or updates the LoRA adapter parameters within the embedding layer.
        merge(safe_merge=False, adapter_names=None): Integrates LoRA parameters into the base embeddings, with an option to
                                                     perform a safe merge that validates the output for NaN values.
        unmerge(): Reverts the base embeddings to their original state by removing any merged LoRA adaptations.
        get_delta_weight(adapter): Computes the weight perturbation introduced by a specified adapter.
        forward(x, *args, **kwargs): Performs a forward pass, applying both the original embeddings and the
                                     LoRA adaptations to the input tensor `x`.

    Example:
        # Instantiate an Embedding layer
        base_embedding_layer = nn.Embedding(num_embeddings=10000, embedding_dim=768)

        # Wrap the Embedding layer with LoRALayer
        lora_embedding = Embedding(base_layer=base_embedding_layer, adapter_name='my_adapter', r=4)

        # Forward pass with the enhanced Embedding layer
        output = lora_embedding(input_indices)

    Note:
        When utilizing Embedding with LoRA, it's important to carefully consider the rank `r` as it directly impacts
        the number of adaptable parameters. Setting `r` too high may lead to over-parameterization, while setting it
        too low may not capture sufficient task-specific information. LoRA enables the model to adapt quickly to new
        tasks with minimal changes to the overall parameter landscape of the model.
    """
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)

        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, LoRALayer):
    """
    A specialized module that extends a standard 2D convolutional layer with LoRA (Layer-wise Representation Adaptation),
    allowing for efficient incremental learning and adaptation of pre-trained convolutional networks.

    By incorporating trainable low-rank matrices, Conv2d LoRA is capable of fine-tuning convolutional weights with a
    limited set of parameters, thus enabling quick adaptation to new tasks with minimal disruption to the existing
    learned representations in larger models.

    Conv2d LoRA is well-suited for applications involving image processing, computer vision, or any tasks that require
    convolutional neural network adaptations, allowing these networks to remain nimble and adaptable without extensive
    retraining or compromising the integrity of their pre-trained knowledge.

    Args:
        base_layer: The original convolutional layer to be adapted with LoRA.
        adapter_name: A name identifier for the LoRA adapter being applied.
        r: The rank for the adaptation, which influences the size of the LoRA matrices.
        lora_alpha: A scaling hyperparameter specific to the LoRA modulation.
        lora_dropout: The dropout rate applied to the LoRA matrices, aiding generalization.
        init_lora_weights: An initialization option for LoRA weights. If set to a boolean
                                                        value, it chooses between default and no initialization. A string
                                                        value specifies a particular method or scheme.
        **kwargs: Additional keyword arguments potentially influencing the behavior of the LoRA layer.

    Methods:
        update_layer(...): Sets up or updates the internal LoRA adapter configuration within the Conv2d layer.
        merge(safe_merge=False, adapter_names=None): Combines the LoRA weights with the base layer, offering an
                                                     optional "safe merge" mode to preclude numerical inconsistencies.
        unmerge(): Dissociates any previously merged LoRA weights from the base layer, effectively reverting to the
                   original layer weights.
        get_delta_weight(adapter): Calculates the weight modification for a specified adapter's LoRA matrices.
        forward(x, *args, **kwargs): Executes a forward pass, combining LoRA adaptations with base layer computations
                                     when processing input tensor `x`.

    Example:
        # Instantiate a Conv2d layer
        base_conv_layer = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Wrap the Conv2d layer with LoRA
        lora_conv2d = Conv2d(base_layer=base_conv_layer, adapter_name='conv_adapter', r=16)

        # Pass input through the enhanced Conv2d layer
        output = lora_conv2d(input_tensor)

    Note:
        Conv2d LoRA takes inspiration from practices in NLP such as adapting large Transformer models, applying these
        learnings to the domain of CNNs. The rank `r` should be chosen judiciously as it controls the trade-off between
        adaptability and parameter efficiency, which is critical in maintaining the balance between learning new
        information and preserving existing knowledge.
    """
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)

        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1, 1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)

                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)

                weight.data -= delta_weight

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                    F.conv2d(
                        weight_A.permute(1, 0, 2, 3),
                        weight_B,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, channel-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4D weight tensors of Conv2D
        weight_norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True).transpose(1, 0)
        return weight_norm

    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied.
        This should be added on top of the base layer output.
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        lora_weight = torch.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.lora_magnitude_vector[active_adapter]
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * (
            F.conv2d(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        return result_dora

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def transpose(weight, fan_in_fan_out: bool):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


class LoRAParametrization(nn.Module):
    """Class responsible for LoRA Update matrices.

    References:
        @inproceedings{
            hu2022lora,
            title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
            author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
            booktitle={International Conference on Learning Representations},
            year={2022},
            url={https://openreview.net/forum?id=nZeVKeeFYf9}
        }
        Original paper: https://arxiv.org/pdf/2106.09685
    """

    def __init__(self, features_in, features_out, rank=1, alpha=1, device="cpu"):
        super().__init__()
        # Section 4.1 from paper:
        #    We use a random Gaussian initialization for A and zero for B,
        #    so ∆W = B * A is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))

        nn.init.normal_(self.lora_A, mean=0, std=1)
        # Section 4.1 from paper:
        #   We then scale ∆W * x by α/r , where α is a constant in r. When optimizing with Adam,
        #   tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
        #   As a result, we simply set α to the first r we try and do not tune it.
        #   This scaling helps to reduce the need to re-tune hyperparameters when we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights) -> torch.Tensor:
        if self.enabled:
            # Return W + (B * A) * scale if lora is enabled instead of W * x + ∆W * x
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale

        return original_weights


def linear_layer_parameterization(layer: torch.Tensor, device: str, rank=1, lora_alpha=1) -> LoRAParametrization:
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights
    #   for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks)
    #   both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.

    features_in, features_out = layer.weight.shape
    return LoRAParametrization(features_in, features_out, rank=rank, alpha=lora_alpha, device=device)