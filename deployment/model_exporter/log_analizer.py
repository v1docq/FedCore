import re
import json
from collections import defaultdict
from typing import List, Dict, Tuple
import os

class LogAnalyzer:
    def __init__(self):
        self.error_patterns = {
            'channel_mismatch': r'expected input\[.*?\] to have (\d+) channels, but got (\d+) channels instead',
            'matrix_multiply': r'mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)',
            'opset_version': r'Unsupported ONNX opset version: (\d+)',
            'shape_mismatch': r'shape mismatch:.*?expected.*?(\d+).*?but got.*?(\d+)'
        }
        
        self.problematic_operations = [
            'Conv', 'Gemm', 'MatMul', 'BatchNormalization', 'Add', 'Sub', 'Mul', 'Div'
        ]
        
    def parse_log_file(self, log_file_path: str) -> List[Dict]:
        """Анализирует лог файл и возвращает список ошибок"""
        errors = []
        
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Log file not found: {log_file_path}")
            
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_error = None
        for line in lines:
            timestamp = line.split(',')[0] if ',' in line else ''
            
            # Проверяем на ошибки
            if 'ERROR' in line and 'OpenVINO' in line:
                error_info = self._extract_error_info(line)
                if error_info:
                    errors.append({
                        'timestamp': timestamp,
                        'type': 'error',
                        'message': line.strip(),
                        'details': error_info
                    })
            
            # Проверяем на предупреждения
            elif 'WARNING' in line and ('opset' in line or 'Failed to export' in line):
                warning_info = self._extract_warning_info(line)
                if warning_info:
                    errors.append({
                        'timestamp': timestamp,
                        'type': 'warning',
                        'message': line.strip(),
                        'details': warning_info
                    })
            
            # Проверяем на информацию о завершении экспорта
            elif 'Exported' in line and ('openvino' in line or 'onnx' in line):
                export_info = self._extract_export_info(line)
                if export_info:
                    errors.append({
                        'timestamp': timestamp,
                        'type': 'export',
                        'message': line.strip(),
                        'details': export_info
                    })
        
        return errors
    
    def _extract_error_info(self, line: str) -> Dict:
        """Извлекает информацию об ошибке из строки лога"""
        error_info = {}
        
        # Проверяем на ошибку с количеством каналов
        channel_match = re.search(self.error_patterns['channel_mismatch'], line)
        if channel_match:
            error_info['type'] = 'channel_mismatch'
            error_info['expected_channels'] = int(channel_match.group(1))
            error_info['actual_channels'] = int(channel_match.group(2))
            return error_info
        
        # Проверяем на ошибку с умножением матриц
        matrix_match = re.search(self.error_patterns['matrix_multiply'], line)
        if matrix_match:
            error_info['type'] = 'matrix_multiply'
            error_info['shape1'] = (int(matrix_match.group(1)), int(matrix_match.group(2)))
            error_info['shape2'] = (int(matrix_match.group(3)), int(matrix_match.group(4)))
            return error_info
        
        # Проверяем на ошибку с версией opset
        opset_match = re.search(self.error_patterns['opset_version'], line)
        if opset_match:
            error_info['type'] = 'unsupported_opset'
            error_info['version'] = int(opset_match.group(1))
            return error_info
        
        return error_info
    
    def _extract_warning_info(self, line: str) -> Dict:
        """Извлекает информацию о предупреждении"""
        warning_info = {}
        
        # Проверяем на ошибку с версией opset
        opset_match = re.search(self.error_patterns['opset_version'], line)
        if opset_match:
            warning_info['type'] = 'unsupported_opset'
            warning_info['version'] = int(opset_match.group(1))
            return warning_info
        
        return warning_info
    
    def _extract_export_info(self, line: str) -> Dict:
        """Извлекает информацию об экспорте"""
        export_info = {}
        
        if 'openvino' in line:
            export_info['format'] = 'openvino'
            export_info['status'] = 'success'
        elif 'onnx' in line:
            export_info['format'] = 'onnx'
            export_info['status'] = 'success'
        
        return export_info
    
    def analyze_problems(self, log_file_path: str) -> Dict:
        """Анализирует лог файл и возвращает отчет о проблемах"""
        errors = self.parse_log_file(log_file_path)
        analysis = {
            'total_errors': len(errors),
            'error_types': defaultdict(int),
            'channel_mismatch_errors': [],
            'matrix_multiply_errors': [],
            'opset_errors': [],
            'problematic_layers': [],
            'export_status': {
                'success': 0,
                'failed': 0
            }
        }
        
        # Собираем статистику по типам ошибок
        for error in errors:
            error_type = error['details'].get('type', 'unknown')
            analysis['error_types'][error_type] += 1
            
            if error_type == 'channel_mismatch':
                analysis['channel_mismatch_errors'].append(error)
            elif error_type == 'matrix_multiply':
                analysis['matrix_multiply_errors'].append(error)
            elif error_type == 'unsupported_opset':
                analysis['opset_errors'].append(error)
            
            # Определяем статус экспорта
            if error['type'] == 'export':
                analysis['export_status']['success'] += 1
            elif error['type'] == 'error' and 'Failed' in error['message']:
                analysis['export_status']['failed'] += 1
        
        return analysis
    
    def find_problematic_layers(self, log_file_path: str, model_layers_info: List[Dict] = None) -> Dict:
        """Ищет слои, связанные с проблемами в логах"""
        errors = self.parse_log_file(log_file_path)
        problematic_layers = []
        
        # Создаем словарь для быстрого поиска по типам ошибок
        error_types = defaultdict(list)
        for error in errors:
            error_type = error['details'].get('type', 'unknown')
            error_types[error_type].append(error)
        
        # Анализируем ошибки с mismatch каналов
        if error_types['channel_mismatch']:
            # В данном случае мы не можем точно определить слой без дополнительной информации
            # но можем указать, что проблема с размерностью
            problematic_layers.append({
                'type': 'channel_mismatch',
                'description': 'Channel dimension mismatch detected',
                'error_count': len(error_types['channel_mismatch']),
                'details': error_types['channel_mismatch']
            })
        
        # Анализируем ошибки с матричным умножением
        if error_types['matrix_multiply']:
            problematic_layers.append({
                'type': 'matrix_multiply',
                'description': 'Matrix multiplication shape mismatch detected',
                'error_count': len(error_types['matrix_multiply']),
                'details': error_types['matrix_multiply']
            })
        
        # Анализируем ошибки с opset
        if error_types['unsupported_opset']:
            problematic_layers.append({
                'type': 'unsupported_opset',
                'description': 'Unsupported ONNX opset version',
                'error_count': len(error_types['unsupported_opset']),
                'details': error_types['unsupported_opset']
            })
        
        return {
            'problematic_layers': problematic_layers,
            'total_problems': len(problematic_layers),
            'error_summary': dict(error_types)
        }
    
    def generate_detailed_report(self, log_file_path: str, model_layers_info: List[Dict] = None) -> str:
        """Генерирует подробный отчет по проблемам"""
        analysis = self.analyze_problems(log_file_path)
        problems = self.find_problematic_layers(log_file_path, model_layers_info)
        
        report = []
        report.append("=" * 60)
        report.append("LOG ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Log file: {os.path.basename(log_file_path)}")
        report.append(f"Total errors: {analysis['total_errors']}")
        report.append("")
        
        # Статистика по типам ошибок
        report.append("ERROR TYPE STATISTICS:")
        report.append("-" * 30)
        for error_type, count in analysis['error_types'].items():
            report.append(f"  {error_type}: {count}")
        report.append("")
        
        # Описание проблемных слоев
        report.append("PROBLEMATIC COMPONENTS:")
        report.append("-" * 30)
        for problem in problems['problematic_layers']:
            report.append(f"  Type: {problem['type']}")
            report.append(f"  Description: {problem['description']}")
            report.append(f"  Error count: {problem['error_count']}")
            report.append("")
        
        # Подробная информация о конкретных ошибках
        if analysis['channel_mismatch_errors']:
            report.append("CHANNEL MISMATCH ERRORS:")
            report.append("-" * 30)
            for error in analysis['channel_mismatch_errors']:
                report.append(f"  Time: {error['timestamp']}")
                report.append(f"  Message: {error['message']}")
                report.append(f"  Expected channels: {error['details']['expected_channels']}")
                report.append(f"  Actual channels: {error['details']['actual_channels']}")
                report.append("")
        
        if analysis['matrix_multiply_errors']:
            report.append("MATRIX MULTIPLICATION ERRORS:")
            report.append("-" * 30)
            for error in analysis['matrix_multiply_errors']:
                report.append(f"  Time: {error['timestamp']}")
                report.append(f"  Message: {error['message']}")
                report.append(f"  Shape 1: {error['details']['shape1']}")
                report.append(f"  Shape 2: {error['details']['shape2']}")
                report.append("")
        
        if analysis['opset_errors']:
            report.append("OPSET VERSION ERRORS:")
            report.append("-" * 30)
            for error in analysis['opset_errors']:
                report.append(f"  Time: {error['timestamp']}")
                report.append(f"  Message: {error['message']}")
                report.append(f"  Unsupported version: {error['details']['version']}")
                report.append("")
        
        report.append("EXPORT STATUS:")
        report.append("-" * 30)
        report.append(f"Successful exports: {analysis['export_status']['success']}")
        report.append(f"Failed exports: {analysis['export_status']['failed']}")
        
        return "\n".join(report)
    
    def save_report(self, log_file_path: str, output_file: str = None):
        """Сохраняет отчет в файл"""
        if not output_file:
            output_file = log_file_path.replace('.log', '_analysis_report.txt')
        
        report = self.generate_detailed_report(log_file_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Analysis report saved to: {output_file}")

# Пример использования
if __name__ == "__main__":
    # Инициализируем анализатор
    analyzer = LogAnalyzer()
    
    # Путь к лог файлу
    log_file = "export_log_20251129_033233.log"
    
    try:
        # Генерируем отчет
        report = analyzer.generate_detailed_report(log_file)
        print(report)
        
        # Сохраняем отчет в файл
        analyzer.save_report(log_file)
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
    except Exception as e:
        print(f"Error analyzing log file: {e}")
