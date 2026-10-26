"""Inspectable CPU backend storage and a restricted converted-FX interpreter.

Converted Linear packed objects are canonicalized into registered codes and
quantization parameters. Prepacking is an explicit inference temporary. Thus
the persistent inventory is exact and no opaque C++ object is hidden from it.
"""
from copy import deepcopy
import torch
from torch import nn


class RegisteredBackendLinear(nn.Module):
    def __init__(self,in_features,out_features,channels,bias=True,*,fused_relu=False,backend='fbgemm',per_channel=True):
        super().__init__()
        if (any(type(v) is not int or v<1 for v in (in_features,out_features,channels))
                or in_features*out_features>2**26 or channels not in (1,out_features)
                or type(bias) is not bool or type(fused_relu) is not bool or type(per_channel) is not bool
                or per_channel and channels!=out_features or not per_channel and channels!=1
                or backend not in torch.backends.quantized.supported_engines or backend=='none'):
            raise ValueError('Invalid registered CPU backend Linear')
        self.in_features,self.out_features,self.channels = in_features,out_features,channels
        self.fused_relu,self.backend = fused_relu,backend
        self.per_channel=per_channel
        self.register_buffer('codes',torch.zeros(out_features,in_features,dtype=torch.int8))
        self.register_buffer('scales',torch.ones(channels,dtype=torch.float64))
        self.register_buffer('zero_points',torch.zeros(channels,dtype=torch.int64))
        self.register_buffer('bias',torch.zeros(out_features,dtype=torch.float32) if bias else None)
        self.register_buffer('output_scale',torch.ones(1,dtype=torch.float64))
        self.register_buffer('output_zero_point',torch.zeros(1,dtype=torch.int64))

    @classmethod
    def from_converted(cls,layer,backend):
        weight,bias = layer.weight(),layer.bias()
        if weight.dtype != torch.qint8:
            raise ValueError('Only qint8 backend storage is admitted')
        per_channel = weight.qscheme() in (torch.per_channel_affine,torch.per_channel_symmetric)
        result = cls(layer.in_features,layer.out_features,layer.out_features if per_channel else 1,bias is not None,
                     fused_relu=type(layer).__name__=='LinearReLU',backend=backend,per_channel=per_channel)
        result.codes = weight.int_repr().clone()
        result.scales = weight.q_per_channel_scales().clone() if per_channel else torch.tensor([weight.q_scale()],dtype=torch.float64)
        result.zero_points = weight.q_per_channel_zero_points().long().clone() if per_channel else torch.tensor([weight.q_zero_point()],dtype=torch.int64)
        if per_channel and weight.q_per_channel_axis()!=0:raise ValueError('Backend output-channel axis required')
        if bias is not None:result.bias = bias.detach().clone().float()
        result.output_scale = torch.tensor([layer.scale],dtype=torch.float64)
        result.output_zero_point = torch.tensor([layer.zero_point],dtype=torch.int64)
        result.validate_state(result.state_dict())
        return result

    def forward(self,x):
        if not x.is_quantized or x.dtype!=torch.quint8 or x.device.type!='cpu' or x.shape[-1]!=self.in_features:
            raise ValueError('Converted backend boundary requires CPU quint8 activations')
        weight = (torch._make_per_tensor_quantized_tensor(self.codes,float(self.scales[0]),int(self.zero_points[0]))
                  if not self.per_channel else torch._make_per_channel_quantized_tensor(self.codes,self.scales,self.zero_points,0))
        previous = torch.backends.quantized.engine
        try:
            torch.backends.quantized.engine = self.backend
            packed = torch.ops.quantized.linear_prepack(weight,self.bias)
            kernel = torch.ops.quantized.linear_relu if self.fused_relu else torch.ops.quantized.linear
            return kernel(x,packed,float(self.output_scale[0]),int(self.output_zero_point[0]))
        finally:
            torch.backends.quantized.engine = previous

    def config(self):
        return {'in_features':self.in_features,'out_features':self.out_features,'channels':self.channels,
                'bias':self.bias is not None,'fused_relu':self.fused_relu,'backend':self.backend,'per_channel':self.per_channel}

    def validate_state(self,state,prefix=''):
        for name,expected in self._buffers.items():
            if expected is None:continue
            value = state[prefix+name]
            if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.shape!=expected.shape
                    or value.dtype!=expected.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Backend buffer contract violated')
        if bool((state[prefix+'scales']<=0).any()) or bool((state[prefix+'output_scale']<=0).any()):
            raise ValueError('Backend scales must be positive')
        if bool(((state[prefix+'zero_points'] < -128)|(state[prefix+'zero_points'] > 127)).any()) or not 0<=int(state[prefix+'output_zero_point'][0])<=255:
            raise ValueError('Backend zero-point outside storage range')


def _encode_arg(arg):
    from torch.fx import Node
    if isinstance(arg,Node):return {'node':arg.name}
    if isinstance(arg,torch.dtype):
        if arg not in (torch.quint8,torch.qint8):raise ValueError('Unsupported FX dtype')
        return {'dtype':str(arg)[6:]}
    if type(arg) in (tuple,list):return [_encode_arg(value) for value in arg]
    if arg is None or type(arg) in (bool,int,float,str):return arg
    raise ValueError('Unsupported converted FX argument')


class RegisteredBackendGraph(nn.Module):
    def __init__(self,instructions,operators,constants):
        super().__init__()
        self.instructions = tuple(instructions)
        self.operators = nn.ModuleDict(operators)
        self.constants = nn.Module()
        for key,value in constants.items():self.constants.register_buffer(key,value.clone())
        self.validate_graph()

    @classmethod
    def from_converted(cls,graph,backend):
        from torch.fx import GraphModule
        if not isinstance(graph,GraphModule):raise ValueError('Converted FX GraphModule required')
        instructions,operators,constants,operator_ids = [],{},{},{}
        for node in graph.graph.nodes:
            target = node.target
            if node.op=='call_module':
                key = 'operator_'+str(len(operators))
                layer = graph.get_submodule(target)
                if id(layer) in operator_ids:
                    target=operator_ids[id(layer)]
                    instructions.append({'name':node.name,'op':node.op,'target':target,'args':_encode_arg(node.args)})
                    continue
                operator_ids[id(layer)]=key
                if type(layer).__module__.startswith(('torch.ao.nn.intrinsic.quantized','torch.ao.nn.quantized','torch.nn.quantized')) and type(layer).__name__ in ('Linear','LinearReLU'):
                    operators[key] = RegisteredBackendLinear.from_converted(layer,backend)
                elif type(layer) in (nn.Linear,nn.ReLU,nn.Tanh,nn.Identity,nn.Sigmoid):operators[key] = deepcopy(layer)
                else:raise ValueError('Unsupported converted backend operator')
                target = key
            elif node.op=='get_attr':
                value = getattr(graph,target)
                if type(value) is not torch.Tensor:raise ValueError('Only registered tensor FX constants allowed')
                target = 'constant_'+str(len(constants));constants[target]=value.detach().clone()
            elif node.op=='call_function':
                if target is not torch.quantize_per_tensor:raise ValueError('Unsupported backend FX function')
                target = 'quantize_per_tensor'
            elif node.op=='call_method':
                if target!='dequantize':raise ValueError('Unsupported backend FX method')
            elif node.op not in ('placeholder','output'):raise ValueError('Unsupported backend FX node')
            instructions.append({'name':node.name,'op':node.op,'target':target,'args':_encode_arg(node.args)})
        return cls(instructions,operators,constants)

    def validate_graph(self):
        names=set();placeholders=0
        def check(arg):
            if type(arg) is dict:
                if set(arg)=={'node'} and arg['node'] in names:return
                if set(arg)=={'dtype'} and arg['dtype'] in ('qint8','quint8'):return
                raise ValueError('Unclosed backend argument')
            if type(arg) is list:
                for value in arg:check(value)
            elif arg is not None and type(arg) not in (str,bool,int,float):raise ValueError('Invalid backend literal')
        for instruction in self.instructions:
            if (type(instruction) is not dict or set(instruction)!={'name','op','target','args'}
                    or instruction['name'] in names):raise ValueError('Invalid backend instruction')
            check(instruction['args'])
            op,target = instruction['op'],instruction['target']
            if op=='placeholder':placeholders+=1
            elif op=='call_module' and target in self.operators:pass
            elif op=='get_attr' and target in self.constants._buffers:pass
            elif op=='call_function' and target=='quantize_per_tensor':pass
            elif op=='call_method' and target=='dequantize':pass
            elif op=='output':pass
            else:raise ValueError('Unknown backend instruction')
            names.add(instruction['name'])
        if placeholders!=1 or not self.instructions or self.instructions[-1]['op']!='output':raise ValueError('One-input completed backend graph required')

    def forward(self,inputs):
        values={}
        def decode(arg):
            if type(arg) is dict:
                return values[arg['node']] if 'node' in arg else getattr(torch,arg['dtype'])
            if type(arg) is list:return tuple(decode(item) for item in arg)
            return arg
        for node in self.instructions:
            args = decode(node['args']);op,target=node['op'],node['target']
            if op=='placeholder':value=inputs
            elif op=='get_attr':value=getattr(self.constants,target)
            elif op=='call_module':value=self.operators[target](*args)
            elif op=='call_function':value=torch.quantize_per_tensor(*args)
            elif op=='call_method':value=args[0].dequantize()
            elif op=='output':return args[0]
            values[node['name']]=value
