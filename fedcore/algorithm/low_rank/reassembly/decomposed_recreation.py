"""Independent standard-layer reconstruction preserving the original operator."""
from copy import deepcopy
from torch import nn
from fedcore.models.network_impl.decomposed_layers import IDecomposed, DecomposedLinear, DecomposedConv1d, DecomposedConv2d, DecomposedEmbedding


def to_standard_module(layer):
    if not isinstance(layer,IDecomposed):
        return deepcopy(layer)
    weight = layer._get_composed_weight().detach()
    options = {'device':weight.device,'dtype':weight.dtype}
    if isinstance(layer,DecomposedLinear):
        result = nn.Linear(layer.in_features,layer.out_features,layer.bias is not None,**options)
    elif isinstance(layer,(DecomposedConv1d,DecomposedConv2d)):
        cls = nn.Conv1d if isinstance(layer,DecomposedConv1d) else nn.Conv2d
        result = cls(layer.in_channels,layer.out_channels,layer.kernel_size,layer.stride,layer.padding,
                     layer.dilation,layer.groups,layer.bias is not None,layer.padding_mode,**options)
    elif isinstance(layer,DecomposedEmbedding):
        result = nn.Embedding(layer.num_embeddings,layer.embedding_dim,layer.padding_idx,layer.max_norm,
                              layer.norm_type,layer.scale_grad_by_freq,layer.sparse,**options)
    else:
        raise ValueError(f'Unsupported decomposed layer: {type(layer).__name__}')
    result.weight.data.copy_(weight)
    result.weight.requires_grad_(layer._weight_requires_grad)
    if getattr(layer,'bias',None) is not None:
        result.bias.data.copy_(layer.bias.detach())
        result.bias.requires_grad_(layer.bias.requires_grad)
    result.train(layer.training)
    return result


def _recreate_embedding(layer):
    return deepcopy(layer)


_recreate_decomposed_linear = to_standard_module
_recreate_decomposed_embedding = to_standard_module
_recreate_decomposed_conv1d = to_standard_module
_recreate_decomposed_conv2d = to_standard_module
RECREATION_FUNCTIONS = {nn.Embedding:_recreate_embedding, DecomposedLinear:to_standard_module,
                       DecomposedEmbedding:to_standard_module,DecomposedConv1d:to_standard_module,
                       DecomposedConv2d:to_standard_module}
