"""Fixed constructors for portable quantization; no dynamic class imports."""
import torch
from .tensor_formats import PackedTensor, QuantizedTensorSpec
from .portable_layers import PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear


def _types():
    from .tender import TenderLinear
    from .backend import RegisteredBackendLinear,RegisteredBackendGraph
    return (PackedTensor, PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear, TenderLinear,RegisteredBackendLinear,RegisteredBackendGraph)


def validate_descriptor_budget(description,max_bytes=128*1024**2):
    """Bound total declared quantized allocations before constructing any part."""
    def constant_size(item):
        if (type(item) is not dict or set(item)!={'shape','dtype'} or item['dtype'] not in ('float32','float64','int64','int32')
                or type(item['shape']) is not list or len(item['shape'])>2
                or any(type(v) is not int or v<0 for v in item['shape'])):
            raise ValueError('Invalid bounded backend constant')
        elements=__import__('math').prod(item['shape'])
        if elements>2**20:raise ValueError('Invalid bounded backend constant')
        return elements*({'float32':4,'float64':8,'int64':8,'int32':4}[item['dtype']])
    def backend_operator_size(item):
        if type(item) is not dict:raise ValueError('Invalid backend operator descriptor')
        kind=item.get('type')
        if kind=='QuantizationGraph' and item.get('kind')=='RegisteredBackendLinear':return size(item)
        if kind in ('ReLU','Tanh','Identity','Sigmoid'):return 0
        if kind=='Linear':
            args=item['args'];i,o=args['in_features'],args['out_features']
            if any(type(n) is not int or n<1 for n in (i,o)) or type(args['bias']) is not bool:
                raise ValueError('Invalid backend float dimensions')
            # The fixed Linear constructor allocates float32 before loading state.
            return (i*o+(o if args['bias'] else 0))*(torch.finfo(torch.get_default_dtype()).bits//8)
        raise ValueError('Unsupported converted backend operator descriptor')
    def packed(shape,spec):
        if len(shape)!=2 or any(type(n) is not int or n<1 for n in shape):raise ValueError('Invalid declared matrix shape')
        return (shape[0]*shape[1]*spec.bits+7)//8+(1 if spec.axis is None else shape[spec.axis])*8
    def size(value):
        if type(value) is list:return sum(size(item) for item in value)
        if type(value) is not dict:return 0
        if value.get('type')!='QuantizationGraph':return sum(size(item) for item in value.values())
        kind,c=value['kind'],value['config']
        if kind=='PackedTensor':return packed(c['shape'],QuantizedTensorSpec.parse(c['spec']))
        if kind in ('PortableQuantizedLinear','PortableQuantizedFactorLinear'):
            i,o=c['in_features'],c['out_features']
            if any(type(n) is not int or n<1 for n in (i,o)):raise ValueError('Invalid declared dimensions')
            bias=o*8 if c['bias'] else 0
            if kind=='PortableQuantizedLinear':return packed((o,i),QuantizedTensorSpec.parse(c['spec']))+bias
            r=c['rank']
            return packed((o,r),QuantizedTensorSpec.parse(c['left_spec']))+packed((r,i),QuantizedTensorSpec.parse(c['right_spec']))+bias
        if kind=='QuantizedSumLinear':return size(c['base'])+size(c['correction'])
        if kind=='TenderLinear':
            i,o,g=c['in_features'],c['out_features'],c['groups']
            if any(type(n) is not int or n<1 for n in (i,o,g)):raise ValueError('Invalid Tender dimensions')
            return packed((o,i),QuantizedTensorSpec(c['weight_bits'],axis=0))+24*i+16*g+16*o+32
        if kind=='RegisteredBackendLinear':
            i,o,k=c['in_features'],c['out_features'],c['channels']
            if any(type(n) is not int or n<1 for n in (i,o,k)):raise ValueError('Invalid backend dimensions')
            return i*o+k*16+o*4+16
        if kind=='RegisteredBackendGraph':
            if type(c['constants']) is not dict or type(c['operators']) is not dict:
                raise ValueError('Backend graph needs explicit constant/operator mappings')
            return sum(constant_size(item) for item in c['constants'].values())+sum(backend_operator_size(item) for item in c['operators'].values())
        return sum(size(item) for item in c.values())
    if size(description)>max_bytes:raise ValueError('Quantization checkpoint declared allocation budget exceeded')


def validate_quantized_storage(model,state=None):
    """Private compact parts may alias only through the same declared module."""
    from fedcore.algorithm.low_rank.topology import module_paths
    paths=module_paths(model);expected={};private=set()
    types=_types()
    for path,layer in paths.items():
        for name,tensor in (*layer._parameters.items(),*layer._buffers.items()):
            if tensor is None:continue
            key=path+'.'+name if path else name
            expected[key]=tensor
            if type(layer) in types:private.add(key)
    values=expected if state is None else state
    groups={}
    for name,value in values.items():
        if not isinstance(value,torch.Tensor) or value.is_quantized:continue
        storage=value.untyped_storage()
        if storage.nbytes():groups.setdefault((str(value.device),storage._cdata),[]).append(name)
    for names in groups.values():
        if private.intersection(names) and len({id(expected.get(name)) for name in names})>1:
            raise ValueError('Private quantization components cannot share undeclared physical storage')


def describe_quantized(layer):
    # Tender is added by P3; importing this adapter before then stays harmless.
    try:
        types = _types()
    except ImportError:
        types = (PackedTensor, PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear)
    if type(layer) not in types:
        return None
    if (any(name in layer.__dict__ for name in ("forward", "validate_state", "state_dict", "decode", "codes"))
            or layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks):
        raise ValueError("Modified portable representation is not checkpointable")
    if type(layer) is PackedTensor:
        layer.validate_state(layer.state_dict())
        config = layer.config()
    elif type(layer) is QuantizedSumLinear:
        config = {"base": describe_quantized(layer.base), "correction": describe_quantized(layer.correction)}
    elif type(layer) is PortableQuantizedLinear:
        config = {"in_features": layer.in_features, "out_features": layer.out_features,
                  "bias": layer.bias is not None, "spec": layer.weight.spec.payload()}
    elif type(layer) is PortableQuantizedFactorLinear:
        config = {"in_features": layer.in_features, "out_features": layer.out_features, "rank": layer.rank,
                  "bias": layer.bias is not None, "left_spec": layer.left.spec.payload(), "right_spec": layer.right.spec.payload()}
    elif type(layer).__name__ == 'RegisteredBackendGraph':
        from fedcore.tools.registry.checkpoint_manager import describe_model
        layer.validate_graph()
        config = {'instructions':list(layer.instructions),
                  'operators':{key:describe_model(value) for key,value in layer.operators.items()},
                  'constants':{key:{'shape':list(value.shape),'dtype':str(value.dtype)[6:]} for key,value in layer.constants._buffers.items()}}
    else:
        layer.validate_state(layer.state_dict())
        config = layer.config()
    return {"type": "QuantizationGraph", "schema": "quantization_graph_v1", "kind": type(layer).__name__, "config": config}


def build_quantized(description):
    if (type(description) is not dict or set(description) != {"type", "schema", "kind", "config"}
            or description["type"] != "QuantizationGraph" or description["schema"] != "quantization_graph_v1"):
        raise ValueError("Unknown quantization checkpoint schema")
    kind, config = description["kind"], description["config"]
    if type(config) is not dict:
        raise ValueError("Mapping config required")
    validate_descriptor_budget(description)
    if kind == "PackedTensor" and set(config) == {"shape", "spec"}:
        return PackedTensor(config["shape"], QuantizedTensorSpec.parse(config["spec"]))
    if kind == "PortableQuantizedLinear" and set(config) == {"in_features", "out_features", "bias", "spec"}:
        return PortableQuantizedLinear(config["in_features"], config["out_features"], config["bias"], spec=QuantizedTensorSpec.parse(config["spec"]))
    if kind == "PortableQuantizedFactorLinear" and set(config) == {"in_features", "out_features", "rank", "bias", "left_spec", "right_spec"}:
        return PortableQuantizedFactorLinear(config["in_features"], config["out_features"], config["rank"], config["bias"],
            left_spec=QuantizedTensorSpec.parse(config["left_spec"]), right_spec=QuantizedTensorSpec.parse(config["right_spec"]))
    if kind == "QuantizedSumLinear" and set(config) == {"base", "correction"}:
        return QuantizedSumLinear(build_quantized(config["base"]), build_quantized(config["correction"]))
    if kind == "TenderLinear":
        from .tender import TenderLinear
        return TenderLinear.from_config(config)
    if kind == 'RegisteredBackendLinear' and set(config)=={'in_features','out_features','channels','bias','fused_relu','backend','per_channel'}:
        from .backend import RegisteredBackendLinear
        return RegisteredBackendLinear(**config)
    if kind == 'RegisteredBackendGraph' and set(config)=={'instructions','operators','constants'}:
        from .backend import RegisteredBackendGraph
        from fedcore.tools.registry.checkpoint_manager import _build_model
        constants = {}
        for key,item in config['constants'].items():
            constants[key]=torch.zeros(item['shape'],dtype=getattr(torch,item['dtype']))
        operators = {key:_build_model(item,{},'') for key,item in config['operators'].items()}
        return RegisteredBackendGraph(config['instructions'],operators,constants)
    raise ValueError("Unknown portable constructor or fields")


def validate_quantized_state(layer, state, prefix=""):
    if type(layer) is PackedTensor:
        layer.validate_state(state, prefix)
    if type(layer) in (PortableQuantizedLinear, PortableQuantizedFactorLinear):
        if layer.bias is not None:
            bias = state[prefix+"bias"]
            if bias.dtype != torch.float64 or bias.shape != (layer.out_features,) or not bool(torch.isfinite(bias).all()):
                raise ValueError("Invalid portable bias")
    if type(layer).__name__ == "TenderLinear" and type(layer).__module__ == "fedcore.algorithm.quantization.tender":
        layer.validate_state(state, prefix)
    from .backend import RegisteredBackendLinear
    if type(layer) is RegisteredBackendLinear:
        layer.validate_state(state,prefix)
