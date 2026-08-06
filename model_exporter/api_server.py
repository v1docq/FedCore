import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
from model_logic import model_manager

import uuid
from werkzeug.utils import secure_filename 

# Настройка логирования
log_dir = "results/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"api_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Разрешенные типы файлов
ALLOWED_EXTENSIONS = {'pt', 'pth', 'onnx', 'tflite', 'engine', 'xml', 'pb', 'json'}
UPLOAD_FOLDER = 'results/models'
# ALLOWED_EXTENSIONS = {'pt', 'pth', 'onnx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Эндпоинт для получения Web UI
@app.route('/')
def index():
    """Возвращает Web UI"""
    try:
        # Читаем содержимое web_ui.html
        with open('templates/web_ui.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except Exception as e:
        logger.error(f"Error loading web UI: {e}")
        return "Web UI not found", 404

# Эндпоинт для получения Web UI через API
@app.route('/webui', methods=['GET'])
def get_webui():
    """Возвращает Web UI через API"""
    try:
        # Читаем содержимое web_ui.html
        with open('templates/web_ui.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except Exception as e:
        logger.error(f"Error loading web UI: {e}")
        return jsonify({"error": "Web UI not found"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка файлов на сервер"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Создаем уникальное имя файла чтобы избежать конфликтов
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            return jsonify({
                "message": "File uploaded successfully",
                "filename": unique_filename,
                "original_name": filename,
                "path": file_path
            })
        else:
            return jsonify({"error": "Invalid file type"}), 400
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/files', methods=['GET'])
def get_files():
    """Получение списка загруженных файлов"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_info = {
                    "name": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
                files.append(file_info)
        
        return jsonify({"files": files})
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return jsonify({"error": str(e)}), 500
    
# Эндпоинт для проверки состояния сервера
@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

# Эндпоинт для экспорта модели
@app.route('/export', methods=['POST'])
def export_model():
    """Экспорт модели в указанный формат"""
    try:
        # Получаем параметры из запроса
        data = request.get_json()
        
        # Проверяем наличие модели (в виде тензора или файла)
        if 'model_path' not in data:
            return jsonify({"error": "model_path must be provided"}), 400
        
        # Загружаем модель
        model_path = data['model_path']
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404
        
        model = torch.load(model_path)
        
        # Получаем параметры экспорта
        export_format = data.get('format', 'torchscript')
        export_dir = data.get('export_dir', 'results/exports')
        model_name = data.get('model_name', 'model')
        
        # Экспортируем модель
        result = model_manager.export_model(model, export_format, export_dir, model_name)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during model export: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для экспорта частей модели
@app.route('/export_parts', methods=['POST'])
def export_parts():
    """Экспорт частей модели с разделением"""
    try:
        data = request.get_json()
        
        # Получаем параметры
        model_path = data.get('model_path')
        export_dir = data.get('export_dir', 'results/exports')
        architecture_file = data.get('architecture_file')
        
        # Загружаем модель
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404
        
        model = torch.load(model_path)
        
        # Экспортируем части
        result = model_manager.export_parts(model, export_dir, architecture_file)
        
        if "error" in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during parts export: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для анализа модели
@app.route('/analyze_model', methods=['POST'])
def analyze_model():
    """Анализ модели"""
    try:
        data = request.get_json()
        
        # Получаем путь к модели
        model_path = data.get('model_path')
        if not model_path:
            return jsonify({"error": "Model path not provided"}), 400
        
        # Проверяем существование файла
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404
        
        # Загружаем модель
        model = torch.load(model_path)
        
        # Анализируем модель
        result = model_manager.analyze_model(model)
        
        if "error" in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during model analysis: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для анализа лог файла
@app.route('/analyze_log', methods=['POST'])
def analyze_log():
    """Анализ лог файла"""
    try:
        data = request.get_json()
        log_file_path = data.get('log_file')
        
        if not log_file_path or not os.path.exists(log_file_path):
            return jsonify({"error": "Log file not found"}), 404
        
        # Анализируем лог
        result = model_manager.analyze_log(log_file_path)
        
        if "error" in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during log analysis: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для получения списка поддерживаемых операций
@app.route('/supported_ops', methods=['GET'])
def get_supported_ops():
    """Получение списка поддерживаемых операций"""
    try:
        result = model_manager.get_supported_ops()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting supported operations: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для получения информации об архитектуре
@app.route('/architecture', methods=['GET'])
def get_architecture():
    """Получение информации об архитектуре"""
    try:
        device_arch = model_manager.device_arch
        return jsonify({
            "architecture": device_arch,
            "name": device_arch.get("name", "Unknown"),
            "cpu_framework": device_arch.get("cpu_framework", "Unknown"),
            "npu_framework": device_arch.get("npu_framework", "Unknown")
        })
    except Exception as e:
        logger.error(f"Error getting architecture info: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для получения списка всех доступных архитектур
@app.route('/architectures', methods=['GET'])
def get_architectures():
    """Получение списка всех доступных архитектур"""
    try:
        result = model_manager.get_architectures()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting architectures: {e}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для получения лог файла
@app.route('/logs/<log_file>', methods=['GET'])
def get_log_file(log_file):
    """Получение лог файла"""
    try:
        # Проверяем, что файл существует
        log_path = os.path.join("results/logs", log_file)
        if not os.path.exists(log_path):
            return jsonify({"error": "Log file not found"}), 404
        
        return send_file(log_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error retrieving log file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Создаем директории
    os.makedirs("device_architectures", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/exports", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("api_logs", exist_ok=True)
    
    logger.info("Starting REST API server. ..")
    app.run(host='0.0.0.0', port=5000, debug=False)
