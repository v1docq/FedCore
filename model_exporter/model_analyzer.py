import torch
import torch.nn as nn
import json
from typing import List, Dict, Any
from collections import defaultdict

class ModelAnalyzer:
    def __init__(self, device_arch: Dict[str, Any]):
        self.device_arch = device_arch
        # PyTorch операции в формате ONNX
        self.op_mapping = {
            # Convolutional Layers
            'Conv2d': 'Conv',
            'ConvTranspose2d': 'ConvTranspose',
            'Conv1d': 'Conv',
            'Conv3d': 'Conv',
            'ConvTranspose1d': 'ConvTranspose',
            'ConvTranspose3d': 'ConvTranspose',
            
            # Activation Functions
            'ReLU': 'Relu',
            'LeakyReLU': 'LeakyRelu',
            'Sigmoid': 'Sigmoid',
            'Tanh': 'Tanh',
            'Softmax': 'Softmax',
            'LogSoftmax': 'LogSoftmax',
            'ELU': 'Elu',
            'SELU': 'Selu',
            'CELU': 'Celu',
            'GELU': 'Gelu',
            'HardSigmoid': 'HardSigmoid',
            'HardSwish': 'HardSwish',
            
            # Pooling Layers
            'MaxPool2d': 'MaxPool',
            'AvgPool2d': 'AvgPool',
            'MaxPool1d': 'MaxPool',
            'AvgPool1d': 'AvgPool',
            'MaxPool3d': 'MaxPool',
            'AvgPool3d': 'AvgPool',
            
            # Linear Layers
            'Linear': 'Gemm',
            'Bilinear': 'Gemm',
            
            # Normalization
            'BatchNorm1d': 'BatchNormalization',
            'BatchNorm2d': 'BatchNormalization',
            'BatchNorm3d': 'BatchNormalization',
            'GroupNorm': 'GroupNorm',
            'LayerNorm': 'LayerNorm',
            'InstanceNorm1d': 'InstanceNorm',
            'InstanceNorm2d': 'InstanceNorm',
            'InstanceNorm3d': 'InstanceNorm',
            
            # Element-wise Operations
            'Add': 'Add',
            'Sub': 'Sub',
            'Mul': 'Mul',
            'Div': 'Div',
            'Pow': 'Pow',
            'Mod': 'Mod',
            'Abs': 'Abs',
            'Neg': 'Neg',
            'Ceil': 'Ceil',
            'Floor': 'Floor',
            'Round': 'Round',
            'Sqrt': 'Sqrt',
            'Rsqrt': 'Rsqrt',
            'Exp': 'Exp',
            'Log': 'Log',
            'LogSoftmax': 'LogSoftmax',
            'Softplus': 'Softplus',
            'Softsign': 'Softsign',
            'Elu': 'Elu',
            'Selu': 'Selu',
            'Celu': 'Celu',
            'HardSigmoid': 'HardSigmoid',
            'HardSwish': 'HardSwish',
            
            # Reduction Operations
            'ReduceMean': 'ReduceMean',
            'ReduceSum': 'ReduceSum',
            'ReduceMax': 'ReduceMax',
            'ReduceMin': 'ReduceMin',
            'ReduceProd': 'ReduceProd',
            'ReduceL1': 'ReduceL1',
            'ReduceL2': 'ReduceL2',
            'ReduceLogSum': 'ReduceLogSum',
            'ReduceLogSumExp': 'ReduceLogSumExp',
            'ReduceSumSquare': 'ReduceSumSquare',
            'ReduceAny': 'ReduceAny',
            'ReduceAll': 'ReduceAll',
            'CumSum': 'CumSum',
            'CumProd': 'CumProd',
            
            # Mathematical Operations
            'Sum': 'Sum',
            'Mean': 'Mean',
            'Min': 'Min',
            'Max': 'Max',
            'Prod': 'Prod',
            'All': 'All',
            'Any': 'Any',
            'BitwiseAnd': 'BitwiseAnd',
            'BitwiseOr': 'BitwiseOr',
            'BitwiseXor': 'BitwiseXor',
            'BitwiseNot': 'BitwiseNot',
            'BitShift': 'BitShift',
            
            # Neural Network Layers
            'LSTM': 'LSTM',
            'GRU': 'GRU',
            'RNN': 'RNN',
            'Embedding': 'Embedding',
            'Dropout': 'Dropout',
            'AlphaDropout': 'Dropout',
            'FeatureAlphaDropout': 'Dropout',
            
            # Reshaping Operations
            'Flatten': 'Flatten',
            'Reshape': 'Reshape',
            'Transpose': 'Transpose',
            'Unsqueeze': 'Unsqueeze',
            'Squeeze': 'Squeeze',
            'Expand': 'Expand',
            'Tile': 'Tile',
            'Repeat': 'Tile',
            
            # Indexing Operations
            'Gather': 'Gather',
            'GatherND': 'GatherND',
            'GatherElements': 'GatherElements',
            'ScatterElements': 'ScatterElements',
            'ScatterND': 'ScatterND',
            'Slice': 'Slice',
            'Pad': 'Pad',
            'Where': 'Where',
            'Range': 'Range',
            'ArgMax': 'ArgMax',
            'ArgMin': 'ArgMin',
            'TopK': 'TopK',
            
            # Comparison Operations
            'Equal': 'Equal',
            'Less': 'Less',
            'Greater': 'Greater',
            'LessOrEqual': 'LessOrEqual',
            'GreaterOrEqual': 'GreaterOrEqual',
            'Not': 'Not',
            'And': 'And',
            'Or': 'Or',
            'Xor': 'Xor',
            
            # Special Operations
            'Upsample': 'Upsample',
            'Resize': 'Resize',
            'Split': 'Split',
            'Concat': 'Concat',
            'Cast': 'Cast',
            'ConstantOfShape': 'ConstantOfShape',
            'Shape': 'Shape',
            'Unique': 'Unique',
            'IsInf': 'IsInf',
            'IsNaN': 'IsNaN',
            'IsFinite': 'IsFinite',
            'Erf': 'Erf',
            'Dilations': 'Dilations',
            'NonMaxSuppression': 'NonMaxSuppression',
            'QuantizeLinear': 'QuantizeLinear',
            'DequantizeLinear': 'DequantizeLinear',
            'DynamicQuantizeLinear': 'DynamicQuantizeLinear',
            'QLinearConv': 'QLinearConv',
            'QLinearMatMul': 'QLinearMatMul',
            'MatMulInteger': 'MatMulInteger',
            'ConvInteger': 'ConvInteger',
            'DeconvInteger': 'DeconvInteger',
            'Clip': 'Clip',
            'Softmax': 'Softmax',
            'LogSoftmax': 'LogSoftmax',
            'Softplus': 'Softplus',
            'Softsign': 'Softsign',
            'Elu': 'Elu',
            'Selu': 'Selu',
            'Celu': 'Celu',
            'HardSigmoid': 'HardSigmoid',
            'HardSwish': 'HardSwish',
        }
    
    def get_layer_type(self, layer) -> str:
        layer_name = type(layer).__name__
        # Используем PyTorch названия как основные, но сопоставляем с ONNX форматом
        return self.op_mapping.get(layer_name, layer_name)
    
    def analyze_model_structure(self, model: nn.Module) -> List[Dict]:
        layers_info = []
        
        # Для torch script моделей нужно использовать специальную обработку
        try:
            # Попробуем получить модули через named_modules()
            for name, module in model.named_modules():
                # Пропускаем пустые модули и последовательности
                if len(list(module.children())) == 0 and not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                    layer_type = self.get_layer_type(module)
                    # Теперь неподдерживаемыми считаются операции, которых нет в списке supported_ops
                    is_supported = layer_type in self.device_arch.get('supported_ops', [])
                    
                    layers_info.append({
                        'name': name,
                        'type': layer_type,
                        'module': module,
                        'supported': is_supported,
                        'module_object': module
                    })
        except Exception as e:
            # Если возникла ошибка, попробуем альтернативный способ
            print(f"Error analyzing model structure: {e}")
            # Попробуем обойти проблему с torch script
            for name, module in model.named_modules():
                try:
                    layer_type = self.get_layer_type(module)
                    is_supported = layer_type in self.device_arch.get('supported_ops', [])
                    layers_info.append({
                        'name': name,
                        'type': layer_type,
                        'module': module,
                        'supported': is_supported,
                        'module_object': module
                    })
                except Exception:
                    # Если не получилось определить тип, добавляем как есть
                    layers_info.append({
                        'name': name,
                        'type': str(type(module).__name__),
                        'module': module,
                        'supported': False,
                        'module_object': module
                    })
        
        return layers_info
    
    def find_split_points(self, layers_info: List[Dict]) -> List[int]:
        if not layers_info:
            return []
        
        split_points = []
        
        for i in range(len(layers_info) - 1):
            current_layer = layers_info[i]
            next_layer = layers_info[i + 1]
            
            if current_layer['supported'] and not next_layer['supported']:
                split_points.append(i + 1)
            elif not current_layer['supported'] and next_layer['supported']:
                split_points.append(i + 1)
        
        if layers_info and not layers_info[-1]['supported']:
            split_points.append(len(layers_info))
        
        split_points = sorted(list(set(split_points)))
        return split_points
    
    def get_model_parts_info(self, model: nn.Module) -> Dict[str, Any]:
        layers_info = self.analyze_model_structure(model)
        split_points = self.find_split_points(layers_info)
        
        parts_info = []
        start_idx = 0
        
        for i, split_point in enumerate(split_points):
            if split_point > start_idx:
                part_layers = layers_info[start_idx:split_point]
                supported_count = sum(1 for layer in part_layers if layer['supported'])
                total_count = len(part_layers)
                
                parts_info.append({
                    'part_index': i,
                    'start_layer': start_idx,
                    'end_layer': split_point,
                    'layers_count': total_count,
                    'supported_layers': supported_count,
                    'unsupported_layers': total_count - supported_count,
                    'layers': part_layers,
                    'is_npu_part': i % 2 == 0
                })
            start_idx = split_point
        
        if start_idx < len(layers_info):
            part_layers = layers_info[start_idx:]
            supported_count = sum(1 for layer in part_layers if layer['supported'])
            total_count = len(part_layers)
            
            parts_info.append({
                'part_index': len(parts_info),
                'start_layer': start_idx,
                'end_layer': len(layers_info),
                'layers_count': total_count,
                'supported_layers': supported_count,
                'unsupported_layers': total_count - supported_count,
                'layers': part_layers,
                'is_npu_part': len(parts_info) % 2 == 0
            })
        
        return {
            'model_layers': layers_info,
            'split_points': split_points,
            'parts_info': parts_info,
            'total_layers': len(layers_info),
            'supported_layers': sum(1 for layer in layers_info if layer['supported']),
            'unsupported_layers': sum(1 for layer in layers_info if not layer['supported'])
        }
