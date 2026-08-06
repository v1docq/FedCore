import torch
import torch.nn as nn
import os
from typing import List, Dict, Any
from model_analyzer import ModelAnalyzer

class ModelSplitter:
    def __init__(self, device_arch: Dict[str, Any]):
        self.device_arch = device_arch
        self.analyzer = ModelAnalyzer(device_arch)
    
    def split_model(self, model: nn.Module, parts_info: Dict[str, Any]) -> List[Dict]:
        """
        Разделяет модель на части по точкам разделения
        """
        parts = []
        layers_info = parts_info['model_layers']
        
        start_idx = 0
        for i, part_info in enumerate(parts_info['parts_info']):
            end_idx = part_info['end_layer']
            
            # Создаем копию модели с нужными слоями
            part_model = self._create_part_model(model, layers_info, start_idx, end_idx)
            
            parts.append({
                'part_index': i,
                'model': part_model,
                'is_npu_part': part_info['is_npu_part'],
                'layers_info': part_info
            })
            
            start_idx = end_idx
            
        return parts
    
    def _create_part_model(self, model: nn.Module, layers_info: List[Dict], start_idx: int, end_idx: int) -> nn.Module:
        """
        Создает часть модели из исходной модели
        """
        # Создаем новую модель с теми же слоями
        part_layers = layers_info[start_idx:end_idx]
        
        # Используем OrderedDict для создания новой модели
        layer_dict = {}
        for layer in part_layers:
            layer_dict[layer['name']] = layer['module']
        
        # Создаем модуль с этими слоями
        if len(layer_dict) == 1:
            return list(layer_dict.values())[0]
        else:
            return nn.Sequential(*list(layer_dict.values()))
    
    def get_parts_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Получает информацию о частях модели
        """
        return self.analyzer.get_model_parts_info(model)
