from copy import deepcopy

from typing import Optional

import torch
import torch.nn.functional as F
from fastai.torch_core import Module
from fastcore.meta import delegates
from torch import nn, Tensor

from fedcore.architecture.abstraction.decorators import convert_to_torch_tensor
from fedcore.architecture.settings.computational import backend_methods as np
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_modules.activation import get_activation_fn
from fedcore.models.network_modules.layers.attention_layers import MultiHeadAttention
from fedcore.models.network_modules.layers.conv_layers import Conv1d, ConvBlock
from fedcore.models.network_modules.layers.linear_layers import (
    Add,
    BN1d,
    Concat,
    Noop,
    Transpose,
)


class EarlyStopping:
    def __init__(
            self,
            patience: int = 15,
            maximise_task: bool = False,
            verbose: bool = False,
            delta: float = 0.5):
        self.patience = patience
        self.verbose = verbose
        self.is_maximise_task = maximise_task
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss

        if self.is_maximise_task:
            patience_condition = self.best_loss + self.delta > loss > self.best_loss - self.delta
        else:
            patience_condition = self.best_loss - self.delta < loss < self.best_loss + self.delta
        if patience_condition:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(
        optimizer, scheduler, epoch, learning_rate, printout=True, lradj="3"
):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == "type1":
        lr_adjust = {epoch: ["learning_rate"] * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "type3":
        lr_adjust = {
            epoch: (
                learning_rate
                if epoch < 3
                else learning_rate * (0.9 ** ((epoch - 3) // 1))
            )
        }
    elif lradj == "constant":
        lr_adjust = {epoch: learning_rate}
    elif lradj == "3":
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate * 0.1}
    elif lradj == "4":
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate * 0.1}
    elif lradj == "5":
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate * 0.1}
    elif lradj == "6":
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate * 0.1}
    elif lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print("Updating learning rate to {}".format(lr))


class MovingAverage(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(
            self,
            kernel_size: int,  # the size of the window
    ):
        super().__init__()
        padding_left = (kernel_size - 1) // 2
        padding_right = kernel_size - padding_left - 1
        self.padding = torch.nn.ReplicationPad1d((padding_left, padding_right))
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: Tensor):
        """
        Args:
            x: torch.Tensor shape: [bs x seq_len x features]
        """
        return self.avg(self.padding(x))


class SeriesDecomposition(nn.Module):
    "Series decomposition block"

    def __init__(
            self,
            kernel_size: int,  # the size of the window
    ):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: Tensor):
        """Args:
        x: torch.Tensor shape: [bs x seq_len x features]
        """
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class SampaddingConv1D_BN(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.padding = nn.ConstantPad1d(
            (int((kernel_size - 1) / 2), int(kernel_size / 2)), 0
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv1d(x)
        x = self.batch_norm(x)
        return x


class ParameterizedLayer(Module):
    """Formerly build_layer_with_layer_parameter"""

    def __init__(self, layer_parameters):
        """
        layer_parameters format
            [in_channels, out_channels, kernel_size,
            in_channels, out_channels, kernel_size,
            ..., nlayers
            ]
        """
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            # in_channels, out_channels, kernel_size
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, x):
        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(x)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class InceptionModule(Module):
    def __init__(
            self, input_dim, number_of_filters, base_kernel_size, bottleneck, activation="ReLU"
    ):
        kernel_size_list = self._generate_kernel_size_list(base_kernel_size)
        if input_dim > 1:
            self.bottleneck = Conv1d(input_dim, number_of_filters, 1, bias=False)
            input_filters = number_of_filters
        else:
            self.bottleneck = Noop
            input_filters = input_dim
        conv_layers_list = [Conv1d(input_filters, number_of_filters, kernel_size, bias=False, )
                            for kernel_size in kernel_size_list]
        self.convs = nn.ModuleList(conv_layers_list)
        self.maxconvpool = nn.Sequential(
            *[
                nn.MaxPool1d(3, stride=1, padding=1),
                Conv1d(input_dim, number_of_filters, 1, bias=False),
            ]
        )
        self.concat = Concat()
        self.batch_norm = BN1d(number_of_filters * 4)
        self.activation = get_activation_fn(activation)

    def _generate_kernel_size_list(self, base_kernel_size):
        ks = [base_kernel_size // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        return ks

    def forward(self, input_tensor):
        x = self.bottleneck(input_tensor)  # step 1 - project to lower dimension
        conv_result = [conv_layer(x.float()) for conv_layer in
                       self.convs]  # step 2 - apply set of conv layers with different "scales" (kernel_size)
        max_conv_result = [self.maxconvpool(
            input_tensor.float())]  # step 3 - apply "standard" (small fixed kernel size) conv layer for input signal
        x = self.concat(conv_result + max_conv_result)  # step 4 - concat in one embedding
        normalized_output = self.batch_norm(x)  # step 5 - normalize output
        output = self.activation(normalized_output)  # step 6 - apply activation func for output
        return output


@delegates(InceptionModule.__init__)
class InceptionBlock(Module):
    def __init__(
            self,
            input_dim,
            number_of_filters=32,
            residual=True,
            depth=6,
            activation="ReLU",
            base_kernel_size=40,
            bottleneck=True,
            **kwargs
    ):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(self.depth):
            inception_module_dim = input_dim if d == 0 else number_of_filters * 4
            self.inception.append(
                InceptionModule(
                    input_dim=inception_module_dim,
                    number_of_filters=number_of_filters,
                    activation=activation,
                    base_kernel_size=base_kernel_size,
                    bottleneck=bottleneck)
            )
            add_skip_connection = all([self.residual, d % 3 == 2])
            if add_skip_connection:
                # input_filters = input_dim if d == 2 else number_of_filters * 4
                input_filters = input_dim
                output_filters = number_of_filters * 4
                self._add_skip_connection(input_filters, output_filters)
        self.add = Add()
        self.activation = get_activation_fn(activation)

    def _add_skip_connection(self, input_filters, output_filters):
        skip_connection_layer = nn.Sequential(nn.Conv1d(in_channels=input_filters, out_channels=output_filters,
                                                        kernel_size=1, stride=1, padding=0),
                                              nn.BatchNorm1d(num_features=output_filters))
        # if input_filters == output_filters:
        #     skip_connection_layer = BN1d(input_filters)
        # else:
        #     skip_connection_layer = ConvBlock(input_filters, output_filters, 1, act=None)
        self.shortcut.append(skip_connection_layer)

    def forward(self, input_tensor):
        # going through all inception block
        X = deepcopy(input_tensor)
        for d, l in enumerate(range(self.depth)):
            X = self.inception[d](X)
            add_skip_connection = all([self.residual, d % 3 == 2])
            # going through all inception block
            if add_skip_connection:
                residual_tensor = self.shortcut[d // 3](input_tensor)
                X = self.add(X, residual_tensor)
                X = self.activation(X)

        return X


class _TSTiEncoderLayer(nn.Module):
    def __init__(
            self,
            q_len,
            d_model,
            n_heads,
            d_k=None,
            d_v=None,
            d_ff=256,
            store_attn=False,
            norm="BatchNorm",
            attn_dropout=0,
            dropout=0.0,
            bias=True,
            activation="GELU",
            res_attention=False,
            pre_norm=False,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiHeadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        """
        Args:
            src: [bs x q_len x d_model]
        """

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)

        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)

        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(
            self,
            input_dim,
            patch_num,
            patch_len,
            n_layers=3,
            d_model=128,
            n_heads=16,
            d_k=None,
            d_v=None,
            d_ff=256,
            norm="BatchNorm",
            attn_dropout=0.0,
            dropout=0.0,
            act="GELU",
            store_attn=False,
            res_attention=True,
            pre_norm=False,
    ):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.layers = nn.ModuleList(
            [
                _TSTiEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=act,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(self, x: Tensor):
        """
        Args:
            x: [bs x nvars x patch_len x patch_num]
        """

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        try:
            pos_encooding = x + self.W_pos
        except Exception:
            W_pos = torch.empty((x.shape[1], x.shape[2]), device=default_device())
            nn.init.uniform_(W_pos, -0.02, 0.02)
            self.W_pos = nn.Parameter(W_pos)
            # x: [bs * nvars x patch_num x d_model]
            pos_encooding = x + self.W_pos
        # x: [bs * nvars x patch_num x d_model]
        x = self.dropout(pos_encooding)
        scores = None

        # Encoder
        if self.res_attention:
            for mod in self.layers:
                x, scores = mod(x, prev=scores)
        else:
            for mod in self.layers:
                x = mod(x)
        # x: [bs x nvars x patch_num x d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x d_model x patch_num]

        return x, scores


class RevIN(nn.Module):
    """Reversible Instance Normalization layer adapted from

    Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2021, September).
    Reversible instance normalization for accurate time-series forecasting against distribution shift.
    In International Conference on Learning Representations.
    Original code: https://github.com/ts-kim/RevIN
    """

    def __init__(
            self,
            input_dim: int,  # #features (aka variables or channels)
            affine: bool = True,  # flag to incidate if RevIN has learnable weight and bias
            subtract_last: bool = False,
            dim: int = 2,  # int or tuple of dimensions used to calculate mean and std
            eps: float = 1e-5,  # epsilon - parameter added for numerical stability
    ):
        super().__init__()
        self.input_dim, self.affine, self.subtract_last, self.dim, self.eps = (
            input_dim,
            affine,
            subtract_last,
            dim,
            eps,
        )
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, input_dim, 1))
            self.bias = nn.Parameter(torch.zeros(1, input_dim, 1))

    def forward(self, x: Tensor, mode: Tensor):
        """Args:

        x: rank 3 tensor with shape [batch size x input_dim x sequence length]
        mode: torch.tensor(True) to normalize data and torch.tensor(False) to reverse normalization
        """

        # Normalize
        if mode:
            return self.normalize(x)

        # Denormalize
        else:
            return self.denormalize(x)

    def normalize(self, x):
        if self.subtract_last:
            self.sub = x[..., -1].unsqueeze(-1).detach()
        else:
            self.sub = torch.mean(x, dim=-1, keepdim=True).detach()
        self.std = (
                torch.std(x, dim=-1, keepdim=True, unbiased=False).detach() + self.eps
        )
        if self.affine:
            x = x.sub(self.sub)
            x = x.div(self.std)
            x = x.mul(self.weight)
            x = x.add(self.bias)
            return x
        else:
            x = x.sub(self.sub)
            x = x.div(self.std)
            return x

    def denormalize(self, x):
        if self.affine:
            x = x.sub(self.bias)
            x = x.div(self.weight)
            x = x.mul(self.std)
            x = x.add(self.sub)
            return x
        else:
            x = x.mul(self.std)
            x = x.add(self.sub)
            return x


class Residual(nn.Module):
    """Implementation of residual connection: `output = x + fn(x)`."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class PreNorm(nn.Module):
    """Applies normalization before `fn`."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))
