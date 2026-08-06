import torch
import torch.nn as nn
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Настройка логирования
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"export_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ModelExporter:
    def __init__(self, device_arch: Dict[str, Any]):
        self.device_arch = device_arch
        self.opset_versions = [22, 21, 20, 19, 18, 17, 16, 14, 12, 11, 10, 8] # от 7 до 22 включительно
    
    def export_parts(self, parts: List[Dict], export_dir: str, format_type: str = None) -> List[str]:
        """
        Экспортирует части модели в указанный формат
        """
        exported_files = []
        
        # Получаем форматы из описания архитектуры
        cpu_framework = self.device_arch.get('cpu_framework', '')
        npu_framework = self.device_arch.get('npu_framework', '')
        
        for part in parts:
            part_index = part['part_index']
            part_model = part['model']
            is_npu_part = part['is_npu_part']
            
            # Определяем формат экспорта в зависимости от типа части и архитектуры
            if is_npu_part:
                # Для NPU используем npu_framework
                export_format = npu_framework
                filename = f"model_part_{part_index}_npu.pt"
            else:
                # Для CPU используем cpu_framework
                export_format = cpu_framework
                filename = f"model_part_{part_index}_cpu.pt"
            
            # Если формат не torchscript, изменяем имя файла
            if export_format != 'torchscript':
                filename = filename.replace('.pt', f'.{export_format}')
            
            export_path = os.path.join(export_dir, filename)
            
            try:
                if export_format == 'torchscript':
                    # Экспорт в TorchScript
                    self._export_torchscript(part_model, export_path)
                elif export_format == 'onnx':
                    # Экспорт в ONNX с разными версиями opset
                    self._export_onnx_with_versions(part_model, export_path)
                elif export_format == 'tflite':
                    # Экспорт в TensorFlow Lite
                    self._export_tflite(part_model, export_path)
                elif export_format == 'tensorrt':
                    # Экспорт в TensorRT
                    self._export_tensorrt(part_model, export_path)
                elif export_format == 'openvino':
                    # Экспорт в OpenVINO
                    self._export_openvino(part_model, export_path)
                elif export_format == 'tvm':
                    # Экспорт в TVM
                    self._export_tvm(part_model, export_path)
                elif export_format == 'tensorflow':
                    # Экспорт в TensorFlow
                    self._export_tensorflow(part_model, export_path)
                else:
                    # По умолчанию - torchscript
                    self._export_torchscript(part_model, export_path)
                
                exported_files.append(export_path)
                logger.info(f"Exported {filename}")
                
            except Exception as e:
                logger.error(f"Error exporting {filename}: {e}")
                # Если произошла ошибка, попробуем экспортировать как torchscript
                try:
                    fallback_path = os.path.join(export_dir, f"model_part_{part_index}_fallback.pt")
                    self._export_torchscript(part_model, fallback_path)
                    exported_files.append(fallback_path)
                    logger.info(f"Fallback exported to {fallback_path}")
                except Exception as fallback_error:
                    logger.error(f"Fallback export also failed: {fallback_error}")
        
        return exported_files
    
    def _export_torchscript(self, model: nn.Module, path: str):
        """
        Экспорт в TorchScript формат
        """
        model.eval()
        try:
            # Пробуем с помощью torch.jit.script
            traced_model = torch.jit.script(model)
            traced_model.save(path)
            logger.info(f"Successfully exported to TorchScript: {path}")
        except Exception as e:
            logger.warning(f"Script export failed, trying trace: {e}")
            try:
                # Если не получилось, используем torch.jit.trace
                # Создаем примерный входной тензор
                example_input = self._get_example_input(model)
                traced_model = torch.jit.trace(model, example_input)
                traced_model.save(path)
                logger.info(f"Successfully exported to TorchScript (trace): {path}")
            except Exception as e2:
                logger.error(f"Trace export also failed: {e2}")
                # Сохраняем как обычный файл
                torch.save(model, path)
                logger.info(f"Saved as regular PyTorch model: {path}")
    
    def _export_onnx_with_versions(self, model: nn.Module, path: str):
        """
        Экспорт в ONNX формат с разными версиями opset
        """
        import onnx
        import torch.onnx
        
        model.eval()
        example_input = self._get_example_input(model)
        
        # Попробуем разные версии opset
        success = False
        for opset_version in self.opset_versions:
            try:
                # Генерируем путь для конкретной версии opset
                opset_path = path.replace('.onnx', f'_opset{opset_version}.onnx')
                
                torch.onnx.export(
                    model,
                    example_input,
                    opset_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                logger.info(f"Successfully exported to ONNX with opset {opset_version}: {opset_path}")
                success = True
                break  # Успешно экспортировано, выходим из цикла
            except Exception as e:
                logger.warning(f"Failed to export with opset {opset_version}: {e}")
                continue
        
        if not success:
            raise Exception("Failed to export with any opset version")
    
    def _export_tflite(self, model: nn.Module, path: str):
        """
        Экспорт в TensorFlow Lite формат
        """
        try:
            import torch
            import tensorflow as tf
            import numpy as np
            import onnx
            from onnx_tf.backend import prepare
            
            model.eval()
            example_input = self._get_example_input(model)
            
            # Экспорт в TorchScript сначала
            traced_model = torch.jit.trace(model, example_input)
            
            # Конвертация в TensorFlow Lite
            # Сначала нужно экспортировать в ONNX, а затем в TFLite
            onnx_path = path.replace('.tflite', '.onnx')
            for opset_version in self.opset_versions:
                try:
                    torch.onnx.export(
                        model,
                    example_input,
                        onnx_path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output']
                    )
                    onnx_model = onnx.load(onnx_path)
                    tf_rep = prepare(onnx_model)
                    tf_rep.export_graph(path)
            
                    logger.info(f"Successfully exported to TFLite: {path}")
                    break
                except Exception as e:
                    logger.error(f"Error exporting to TFLite: {e}")
                    continue
        except ImportError as e:
            logger.error(f"TFLite export requires onnx and onnx-tf. Please install: pip install onnx onnx-tf. Error: {e}")
            raise
 
    def _export_tensorrt(self, model: nn.Module, path: str):
        """
        Экспорт в TensorRT формат
        """
        try:
            import torch
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            model.eval()
            example_input = self._get_example_input(model)
            
            # Экспорт в ONNX сначала
            onnx_path = path.replace('.engine', '.onnx')
            for opset_version in self.opset_versions:
                try:
                    torch.onnx.export(
                        model,
                    example_input,
                        onnx_path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output']
                    )
                    
                    # Создание TensorRT engine
                    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
                    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                    parser = trt.OnnxParser(network, builder.logger)
                    
                    with open(onnx_path, 'rb') as model_file:
                        parser.parse(model_file.read())
                    
                    config = builder.create_builder_config()
                    config.max_workspace_size = 1 << 30  # 1GB
                    
                    engine = builder.build_engine(network, config)
                    
                    # Сохранение engine
                    with open(path, 'wb') as f:
                        f.write(engine.serialize())
                    
                    logger.info(f"Successfully exported to TensorRT: {path}")
                    break
                except Exception as e:
                    logger.error(f"Error exporting to TensorRT: {e}")
                    continue
        except ImportError as e:
            logger.error(f"TensorRT export requires tensorrt and pycuda. Please install: pip install nvidia-tensorrt. Error: {e}")
            raise

    
    def _export_openvino(self, model: nn.Module, path: str):
        """
        Экспорт в OpenVINO формат
        """
        try:
            import openvino
            from openvino.tools import mo
            
            model.eval()
            example_input = self._get_example_input(model)
            
            # Экспорт в ONNX сначала
            onnx_path = path.replace('.xml', '.onnx')
            for opset_version in self.opset_versions:
                try:
                    torch.onnx.export(
                        model,
                    example_input,
                        onnx_path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output']
                    )
                    
                    # Конвертация в OpenVINO
                    mo.main([
                        '--input_model', onnx_path,
                        '--output_dir', os.path.dirname(path),
                        '--model_name', os.path.basename(path).replace('.xml', '')
                    ])
            
                    logger.info(f"Successfully exported to OpenVINO: {path}")
                    break
                except Exception as e:
                    logger.error(f"Error exporting to OpenVINO: {e}")
                    continue
        except ImportError as e:
            logger.error(f"OpenVINO export requires openvino. Please install: pip install openvino. Error: {e}")
            raise

    
    def _export_tvm(self, model: nn.Module, path: str):
        """
        Экспорт в TVM формат
        """
        try:
            import tvm
            from tvm import relay
            import numpy as np
            
            model.eval()
            example_input = self._get_example_input(model)
            
            # Экспорт в Relay
            shape_dict = {'input': example_input.shape}
            mod, params = relay.frontend.from_pytorch(model, shape_dict)
            
            # Компиляция
            target = "llvm"  # Можно изменить на "cuda" для GPU
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
            
            # Сохранение
            lib.export_library(path)
            
            logger.info(f"Successfully exported to TVM: {path}")
        except ImportError as e:
            logger.error(f"TVM export requires tvm. Please install: pip install tvm. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error exporting to TVM: {e}")
            raise
    
    def _export_tensorflow(self, model: nn.Module, path: str):
        """
        Экспорт в TensorFlow формат
        """
        try:
            import tensorflow as tf
            import torch
            import numpy as np
            import onnx
            from onnx_tf.backend import prepare
            
            model.eval()
            example_input = self._get_example_input(model)
            
            # Экспорт в ONNX сначала
            onnx_path = path.replace('.pb', '.onnx')
            for opset_version in self.opset_versions:
                try:
                    torch.onnx.export(
                        model,
                        example_input,
                        onnx_path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output']
                    )
                    onnx_model = onnx.load(onnx_path)
                    tf_rep = prepare(onnx_model)
                    
                    # Сохранение как TensorFlow SavedModel
                    tf_rep.export_graph(path)
                    
                    logger.info(f"Successfully exported to TensorFlow: {path}")
                    break
                except Exception as e:
                    logger.error(f"Error exporting to TensorFlow: {e}")
                    continue
        except ImportError as e:
            logger.error(f"TensorFlow export requires onnx and onnx-tf. Please install: pip install onnx onnx-tf. Error: {e}")
            raise

    
    def _get_example_input(self, model: nn.Module):
        """
        Создает примерный входной тензор для трассировки
        """
        # Простой подход - создаем примерный тензор
        return torch.randn(1, 3, 224, 224)  # Для примера
