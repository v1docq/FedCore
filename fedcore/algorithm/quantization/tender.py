"""Tender WbAb Linear reference with fixed global channel metadata.

Python integers deliberately check every partial accumulation before conversion
to int64. This is an executable oracle, not a hardware throughput implementation.
"""
import math
import torch
from torch import nn
from .tensor_formats import PackedTensor, QuantizedTensorSpec
from .portable_layers import _input


class TenderLinear(nn.Module):
    def __init__(self, in_features, out_features, groups, bias=True, *, weight_bits=8, activation_bits=8, accumulator_bits=64):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (in_features,out_features,groups))
                or in_features*out_features > 2**26 or groups > in_features
                or type(bias) is not bool or type(activation_bits) is not int or activation_bits not in (2,4,8)
                or type(accumulator_bits) is not int or accumulator_bits not in (16,32,64)):
            raise ValueError("Invalid bounded Tender profile")
        self.in_features, self.out_features, self.groups = in_features,out_features,groups
        self.activation_bits, self.accumulator_bits = activation_bits,accumulator_bits
        self.weight = PackedTensor((out_features,in_features),QuantizedTensorSpec(weight_bits,axis=0))
        self.register_buffer("centers",torch.zeros(in_features,dtype=torch.float64))
        self.register_buffer("activation_scale",torch.ones(1,dtype=torch.float64))
        self.register_buffer("exponents",torch.arange(groups-1,-1,-1,dtype=torch.int64))
        self.register_buffer("activation_indices",torch.arange(in_features,dtype=torch.int64))
        self.register_buffer("weight_indices",torch.arange(in_features,dtype=torch.int64))
        self.register_buffer("group_ptr",torch.tensor([*range(groups),in_features],dtype=torch.int64))
        self.register_buffer("center_correction",torch.zeros(out_features,dtype=torch.float64))
        self.register_buffer("bias",torch.zeros(out_features,dtype=torch.float64) if bias else None)

    @classmethod
    def calibrate(cls, weight, bias, rows, *, weight_bits=8, activation_bits=8, accumulator_bits=64):
        from .matrix_transforms import _matrix
        _matrix(weight); _matrix(rows)
        if rows.shape[1] != weight.shape[1]:
            raise ValueError("Tender calibration coordinates disagree")
        x = rows.detach().double()
        centers = x.amin(0)*.5+x.amax(0)*.5
        ranges = (x-centers).abs().amax(0)
        maximum = float(ranges.max())
        positive = ranges > 0
        if maximum:
            if bool((ranges[positive]<maximum/2**60).any()):
                raise ValueError('Tender channel-scale dynamic range exceeds reference profile')
            depth = torch.zeros_like(ranges,dtype=torch.int64)
            # Direct binary thresholds preserve exact halves and avoid the
            # reciprocal overflow/rounding of scalar divided by Tensor.
            for level in range(1,61):
                depth[positive & (ranges<=maximum/2**level)]=level
            deepest = int(depth.max())
            if deepest > 60:
                raise ValueError("Tender channel-scale dynamic range exceeds reference profile")
            # Exact half-maximum belongs to the next level. Zero channels share
            # the finest supported level, with exact centered activation zero.
            depth[~positive] = deepest
            exponents = deepest-depth
            base = maximum/(2**(activation_bits-1)-1)/2**deepest
        else:
            exponents,base = torch.zeros_like(ranges,dtype=torch.int64),1.
        if not math.isfinite(base) or base <= 0:
            raise ValueError("Tender scale underflow")
        if bool((exponents > 60).any()):
            raise ValueError("Tender channel-scale dynamic range exceeds reference profile")
        levels = sorted(set(exponents.tolist()),reverse=True)
        indices, ptr = [], [0]
        for level in levels:
            indices.extend(torch.where(exponents==level)[0].tolist())
            ptr.append(len(indices))
        layer = cls(weight.shape[1],weight.shape[0],len(levels),bias is not None,
                    weight_bits=weight_bits,activation_bits=activation_bits,accumulator_bits=accumulator_bits)
        layer.weight = PackedTensor.from_tensor(weight,QuantizedTensorSpec(weight_bits,axis=0))
        layer.centers = centers.clone()
        layer.activation_scale = torch.tensor([base],dtype=torch.float64)
        layer.exponents = torch.tensor(levels,dtype=torch.int64)
        layer.activation_indices = torch.tensor(indices,dtype=torch.int64)
        layer.weight_indices = layer.activation_indices.clone()
        layer.group_ptr = torch.tensor(ptr,dtype=torch.int64)
        layer.center_correction = layer.weight.decode()@centers
        if bias is not None:
            if bias.device.type != "cpu" or bias.shape != (weight.shape[0],) or not bool(torch.isfinite(bias).all()):
                raise ValueError("Invalid Tender bias")
            layer.bias = bias.detach().double().clone()
        layer.validate_state(layer.state_dict())
        return layer

    def activation_codes(self, inputs):
        _input(inputs,self.in_features)
        centered = inputs.double().reshape(-1,self.in_features)-self.centers
        codes = torch.zeros_like(centered,dtype=torch.int64)
        qmax = 2**(self.activation_bits-1)-1
        for g, exponent in enumerate(self.exponents.tolist()):
            ids = self.activation_indices[self.group_ptr[g]:self.group_ptr[g+1]]
            codes[:,ids] = (centered[:,ids]/(float(self.activation_scale[0])*2**exponent)).round().clamp(-qmax,qmax).long()
        return codes

    def integer_accumulator(self, inputs):
        codes, weights = self.activation_codes(inputs), self.weight.codes()
        lower,upper = -2**(self.accumulator_bits-1),2**(self.accumulator_bits-1)-1
        def checked(value):
            if not lower <= value <= upper:
                raise OverflowError("Tender intermediate accumulator overflow")
            return value
        result = []
        for row in codes.tolist():
            outputs = []
            for w in weights.tolist():
                accumulator = 0
                previous = None
                for g, exponent in enumerate(self.exponents.tolist()):
                    if previous is not None:
                        accumulator = checked(accumulator*2**(previous-exponent))
                    product = 0
                    start,end = int(self.group_ptr[g]),int(self.group_ptr[g+1])
                    for xi,wi in zip(self.activation_indices[start:end].tolist(),self.weight_indices[start:end].tolist()):
                        term = checked(row[xi]*w[wi])
                        product = checked(product+term)
                    accumulator = checked(accumulator+product)
                    previous = exponent
                outputs.append(accumulator)
            result.append(outputs)
        return torch.tensor(result,dtype=torch.int64).reshape(-1,self.out_features)

    def forward(self, inputs):
        accumulator = self.integer_accumulator(inputs)
        final_scale = self.activation_scale[0]*2**int(self.exponents[-1])
        # Separate mantissas/exponents: either multiplication order can overflow
        # or underflow although the final three-factor product is representable.
        a,ae=torch.frexp(accumulator.double())
        b,be=torch.frexp(final_scale)
        c,ce=torch.frexp(self.weight.scale)
        mantissas,exponents=torch.broadcast_tensors(a*b*c,ae+be+ce)
        try:
            # Torch 2.2 ldexp forms 2**exponent first, losing finite extreme
            # products. The scalar CPU oracle rounds only the complete product.
            products=[math.ldexp(float(m),int(e)) for m,e in zip(mantissas.flatten(),exponents.flatten())]
        except OverflowError as error:
            raise ValueError('Tender output overflow') from error
        output = torch.tensor(products,dtype=torch.float64).reshape(accumulator.shape)+self.center_correction
        if self.bias is not None:
            output = output+self.bias
        output = output.reshape(*inputs.shape[:-1],self.out_features).to(inputs.dtype)
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Tender output overflow")
        return output

    def config(self):
        return {"in_features":self.in_features,"out_features":self.out_features,"groups":self.groups,
            "bias":self.bias is not None,"weight_bits":self.weight.spec.bits,"activation_bits":self.activation_bits,
            "accumulator_bits":self.accumulator_bits,"profile":"fixed_global_channel_linear",
            "overflow":"refuse","center_weight":"decoded","version":1}

    @classmethod
    def from_config(cls, config):
        keys = {"in_features","out_features","groups","bias","weight_bits","activation_bits","accumulator_bits","profile","overflow","center_weight","version"}
        if (type(config) is not dict or set(config) != keys or type(config["version"]) is not int or config["version"] != 1
                or config["profile"] != "fixed_global_channel_linear" or config["overflow"] != "refuse" or config["center_weight"] != "decoded"):
            raise ValueError("Unsupported Tender profile; row-chunk/per-head/all require another adapter")
        return cls(**{key:config[key] for key in ("in_features","out_features","groups","bias","weight_bits","activation_bits","accumulator_bits")})

    def validate_state(self,state,prefix=""):
        expected = {name:value for name,value in self._buffers.items() if value is not None}
        for name,template in expected.items():
            value = state[prefix+name]
            if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != template.dtype
                    or value.shape != template.shape or not bool(torch.isfinite(value).all())):
                raise ValueError("Tender metadata shape, dtype or finite-value failure")
        exponents,ptr,ai,wi = (state[prefix+name] for name in ("exponents","group_ptr","activation_indices","weight_indices"))
        if (int(ptr[0]) != 0 or int(ptr[-1]) != self.in_features or bool((ptr[1:] <= ptr[:-1]).any())
                or not torch.equal(ai.sort().values,torch.arange(self.in_features)) or not torch.equal(ai,wi)
                or bool((exponents < 0).any()) or bool((exponents > 60).any())
                or bool((exponents[1:] >= exponents[:-1]).any()) or float(state[prefix+"activation_scale"][0]) <= 0):
            raise ValueError("Tender indices/groups/scales are inconsistent")
        self.weight.validate_state(state,prefix+"weight.")
        packed,scale = self.weight.packed,self.weight.scale
        try:
            self.weight.packed,self.weight.scale = state[prefix+"weight.packed"],state[prefix+"weight.scale"]
            correction = self.weight.decode()@state[prefix+"centers"]
        finally:
            self.weight.packed,self.weight.scale = packed,scale
        if not torch.allclose(correction,state[prefix+"center_correction"],rtol=1e-12,atol=1e-12):
            raise ValueError("Tender center correction must use the declared decoded weight")
