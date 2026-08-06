import torch
import torch.nn as nn
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from model_exporter import ModelExporter
from model_splitter import ModelSplitter
from model_analyzer import ModelAnalyzer
from log_analizer import LogAnalyzer

# Настройка логирования
def setup_logging(log_dir="results/logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"model_logic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ModelManager:
    def __init__(self):
        self.device_arch = self.load_default_architecture()
        self.exporter = ModelExporter(self.device_arch)
        self.splitter = ModelSplitter(self.device_arch)
        self.analyzer = ModelAnalyzer(self.device_arch)
        self.log_analyzer = LogAnalyzer()
    
    def load_default_architecture(self):
        """Загружает архитектуру по умолчанию"""
        try:
            arch_path = "device_architectures/rk3588s_arch.json"
            with open(arch_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading device architecture: {e}")
            return {
                "name": "Default",
                "cpu_framework": "onnx",
                "npu_framework": "openvino",
                "supported_ops": [],
                "unsupported_ops": []
            }
    
    def load_architecture(self, arch_file):
        """Загружает указанную архитектуру"""
        try:
            with open(arch_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading architecture {arch_file}: {e}")
            return self.load_default_architecture()
    
    def save_model(self, model, model_path):
        """Сохраняет модель"""
        try:
            torch.save(model, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path):
        """Загружает модель"""
        try:
            model = torch.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None
    
    def export_model(self, model, export_format, export_dir, model_name):
        """Экспортирует модель в указанный формат"""
        try:
            os.makedirs(export_dir, exist_ok=True)
            
            if export_format == 'torchscript':
                export_path = os.path.join(export_dir, f"{model_name}.pt")
                self.exporter._export_torchscript(model, export_path)
                result = {"message": "Model exported to TorchScript", "file": export_path}
            elif export_format == 'onnx':
                export_path = os.path.join(export_dir, f"{model_name}.onnx")
                self.exporter._export_onnx_with_versions(model, export_path)
                result = {"message": "Model exported to ONNX", "file": export_path}
            elif export_format == 'tflite':
                export_path = os.path.join(export_dir, f"{model_name}.tflite")
                self.exporter._export_tflite(model, export_path)
                result = {"message": "Model exported to TFLite", "file": export_path}
            elif export_format == 'tensorrt':
                export_path = os.path.join(export_dir, f"{model_name}.engine")
                self.exporter._export_tensorrt(model, export_path)
                result = {"message": "Model exported to TensorRT", "file": export_path}
            elif export_format == 'openvino':
                export_path = os.path.join(export_dir, f"{model_name}.xml")
                self.exporter._export_openvino(model, export_path)
                result = {"message": "Model exported to OpenVINO", "file": export_path}
            elif export_format == 'tvm':
                export_path = os.path.join(export_dir, f"{model_name}.so")
                self.exporter._export_tvm(model, export_path)
                result = {"message": "Model exported to TVM", "file": export_path}
            elif export_format == 'tensorflow':
                export_path = os.path.join(export_dir, f"{model_name}.pb")
                self.exporter._export_tensorflow(model, export_path)
                result = {"message": "Model exported to TensorFlow", "file": export_path}
            else:
                return {"error": "Unsupported export format"}
            
            logger.info(f"Model exported successfully: {export_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error during model export: {e}")
            return {"error": str(e)}
    
    def export_parts(self, model, export_dir, architecture_file=None):
        """Экспортирует части модели"""
        try:
            if architecture_file:
                device_arch = self.load_architecture(architecture_file)
                # Создаем новый экспортер с новой архитектурой
                temp_exporter = ModelExporter(device_arch)
                # Используем оригинальные компоненты для разбиения
                parts_info = self.splitter.get_parts_info(model)
                parts = self.splitter.split_model(model, parts_info)
                exported_files = temp_exporter.export_parts(parts, export_dir)
            else:
                parts_info = self.splitter.get_parts_info(model)
                parts = self.splitter.split_model(model, parts_info)
                exported_files = self.exporter.export_parts(parts, export_dir)
            
            result = {
                "message": "Model parts exported successfully",
                "exported_files": exported_files,
                "parts_count": len(exported_files)
            }
            
            logger.info(f"Model parts exported successfully: {len(exported_files)} files")
            return result
            
        except Exception as e:
            logger.error(f"Error during parts export: {e}")
            return {"error": str(e)}
    
    def analyze_model(self, model):
        """Анализирует модель"""
        try:
            parts_info = self.splitter.get_parts_info(model)
            
            # Преобразуем все несериализуемые объекты в строки
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {key: make_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, '__class__'):
                    return str(obj.__class__.__name__)
                else:
                    return obj
            
            # Применяем сериализацию ко всем частям информации
            processed_parts_info = make_serializable(parts_info)
            
            result = {
                "model_analysis": processed_parts_info,
                "total_layers": processed_parts_info['total_layers'],
                "supported_layers": processed_parts_info['supported_layers'],
                "unsupported_layers": processed_parts_info['unsupported_layers']
            }
            
            logger.info(f"Model analyzed successfully: {processed_parts_info['total_layers']} layers")
            return result
            
        except Exception as e:
            logger.error(f"Error during model analysis: {e}")
            return {"error": str(e)}
    
    def analyze_log(self, log_file_path):
        """Анализирует лог файл"""
        try:
            if not os.path.exists(log_file_path):
                return {"error": "Log file not found"}
            
            analysis = self.log_analyzer.analyze_problems(log_file_path)
            problems = self.log_analyzer.find_problematic_layers(log_file_path)
            report = self.log_analyzer.generate_detailed_report(log_file_path)
            
            result = {
                "analysis": analysis,
                "problems": problems,
                "report": report
            }
            
            logger.info(f"Log analyzed successfully: {log_file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error during log analysis: {e}")
            return {"error": str(e)}
    
    def get_supported_ops(self):
        """Получает список поддерживаемых операций"""
        try:
            supported_ops = self.device_arch.get('supported_ops', [])
            return {
                "supported_operations": supported_ops,
                "count": len(supported_ops)
            }
        except Exception as e:
            logger.error(f"Error getting supported operations: {e}")
            return {"error": str(e)}
    
    def get_architectures(self):
        """Получает список всех доступных архитектур"""
        try:
            import glob
            arch_files = glob.glob("device_architectures/*.json")
            architectures = []
            for arch_file in arch_files:
                try:
                    with open(arch_file, 'r') as f:
                        arch_data = json.load(f)
                        architectures.append({
                            "name": arch_data.get("name", os.path.basename(arch_file)),
                            "file": arch_file,
                            "cpu_framework": arch_data.get("cpu_framework", "Unknown"),
                            "npu_framework": arch_data.get("npu_framework", "Unknown")
                        })
                except Exception as e:
                    logger.error(f"Error loading architecture {arch_file}: {e}")
            return {"architectures": architectures}
        except Exception as e:
            logger.error(f"Error getting architectures: {e}")
            return {"error": str(e)}

# Инициализация менеджера моделей
model_manager = ModelManager()
