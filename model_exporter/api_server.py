import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
from model_logic import model_manager
from loader_bundle import LoaderBundle
from fedcore_ops import (
    detect_capabilities,
    load_torch_module,
    run_operation,
    export_via_fedcore,
    example_input_from_loader,
    load_dataloader_from_bundle,
)

from werkzeug.utils import secure_filename
import shutil
import sys
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
LOADER_FOLDER = 'results/loaders'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LOADER_FOLDER'] = LOADER_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOADER_FOLDER, exist_ok=True)


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
    """Загрузка моделей / архитектур на сервер."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return jsonify({
                "message": "File uploaded successfully",
                "filename": filename,
                "original_name": filename,
                "path": file_path,
                "kind": "model_or_arch",
            })
        else:
            return jsonify({"error": "Invalid file type"}), 400
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload_loader', methods=['POST'])
def upload_loader():
    """Загрузка FedCore loader bundle (.pt) — данные, не модель."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename) or not file.filename.lower().endswith(('.pt', '.pth')):
            return jsonify({"error": "Loader must be a .pt / .pth bundle"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['LOADER_FOLDER'], filename)
        file.save(file_path)

        try:
            meta = LoaderBundle.inspect(file_path).to_dict()
        except Exception as inspect_error:
            os.remove(file_path)
            return jsonify({
                "error": f"Invalid loader bundle: {inspect_error}"
            }), 400

        return jsonify({
            "message": "Loader uploaded successfully",
            "filename": filename,
            "original_name": filename,
            "path": file_path,
            "kind": "loader",
            "meta": meta,
        })
    except Exception as e:
        logger.error(f"Error uploading loader: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/loaders', methods=['GET'])
def list_loaders():
    """Список загруженных loader bundles + краткие метаданные."""
    try:
        loaders = []
        folder = app.config['LOADER_FOLDER']
        if not os.path.isdir(folder):
            return jsonify({"loaders": []})

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if not os.path.isfile(file_path):
                continue
            entry = {
                "name": filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "meta": None,
            }
            try:
                entry["meta"] = LoaderBundle.inspect(file_path).to_dict()
            except Exception as e:
                entry["meta_error"] = str(e)
            loaders.append(entry)
        return jsonify({"loaders": loaders})
    except Exception as e:
        logger.error(f"Error listing loaders: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/inspect_loader', methods=['POST'])
def inspect_loader():
    """Метаданные одного loader bundle (без тензоров в ответе)."""
    try:
        data = request.get_json() or {}
        loader_path = data.get('loader_path') or data.get('path')
        if not loader_path:
            filename = data.get('filename')
            if filename:
                loader_path = os.path.join(app.config['LOADER_FOLDER'], filename)
        if not loader_path or not os.path.exists(loader_path):
            return jsonify({"error": "Loader file not found"}), 404
        meta = LoaderBundle.inspect(loader_path).to_dict()
        return jsonify({"meta": meta, "path": loader_path})
    except Exception as e:
        logger.error(f"Error inspecting loader: {e}")
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

# Эндпоинт для экспорта модели (FedCore: ONNX / TensorRT / TorchScript)
@app.route('/export', methods=['POST'])
def export_model():
    """Export via FedCore (fedcore.tools.export / FedCore.export)."""
    try:
        data = request.get_json() or {}
        if 'model_path' not in data:
            return jsonify({"error": "model_path must be provided"}), 400

        model_path = data['model_path']
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404

        model = load_torch_module(model_path)
        export_format = data.get('format', 'onnx')
        export_dir = data.get('export_dir', 'results/exports')
        model_name = data.get('model_name', 'model')
        loader_path = data.get('loader_path')

        loader = None
        if loader_path and os.path.exists(loader_path):
            loader = load_dataloader_from_bundle(loader_path)
        dummy = example_input_from_loader(loader)

        result = export_via_fedcore(
            model,
            framework=export_format,
            export_dir=export_dir,
            model_name=model_name,
            example_input=dummy,
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Error during FedCore export")
        return jsonify({"error": str(e) or repr(e)}), 500


@app.route('/model_capabilities', methods=['POST'])
def model_capabilities():
    """Detect FedCore ops available for a loaded .pt module."""
    try:
        data = request.get_json() or {}
        model_path = data.get('model_path')
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404
        model = load_torch_module(model_path)
        caps = detect_capabilities(model, kind=data.get('kind', 'auto'))
        return jsonify(caps.to_dict())
    except Exception as e:
        logger.exception("Error detecting capabilities")
        return jsonify({"error": str(e) or repr(e)}), 500


@app.route('/fedcore_op', methods=['POST'])
def fedcore_op():
    """Run FedCore operation: quantize / prune / low_rank / export_*."""
    try:
        data = request.get_json() or {}
        operation = data.get('operation')
        model_path = data.get('model_path')
        if not operation or not model_path:
            return jsonify({"error": "operation and model_path are required"}), 400
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404

        result = run_operation(
            operation,
            model_path,
            loader_path=data.get('loader_path'),
            export_dir=data.get('export_dir', 'results/exports'),
            model_name=data.get('model_name', 'model'),
            pruning_ratio=float(data.get('pruning_ratio', 0.3)),
            kind=data.get('kind', 'auto'),
        )
        return jsonify(result)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        logger.exception("Error running FedCore op")
        return jsonify({"error": str(e) or repr(e)}), 500

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
        
        # Full nn.Module checkpoints (not bare state_dict)
        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            model = torch.load(model_path, map_location="cpu")

        if isinstance(model, dict) and isinstance(model.get("model"), nn.Module):
            model = model["model"]
        elif isinstance(model, dict) and "state_dict" in model and not isinstance(model, nn.Module):
            return jsonify({
                "error": (
                    "Checkpoint contains state_dict only, not a full nn.Module. "
                    "Save with torch.save(model, path) or include the module object."
                )
            }), 400

        arch_file = data.get("architecture_file") or data.get("device_profile")
        if arch_file and not model_manager.set_device_architecture(arch_file):
            return jsonify({"error": f"Device profile not found: {arch_file}"}), 404

        result = model_manager.analyze_model(model)

        if "error" in result:
            return jsonify(result), 500

        try:
            kind = data.get("kind", "auto")
            result["capabilities"] = detect_capabilities(model, kind=kind).to_dict()
        except Exception as cap_err:
            logger.warning(f"capabilities detection failed: {cap_err}")
            result["capabilities"] = {
                "operations": [],
                "kind": "unknown",
                "suggested_kind": "other",
                "findings": [],
            }

        return jsonify(result)
        
    except Exception as e:
        logger.exception("Error during model analysis")
        return jsonify({"error": str(e) or repr(e)}), 500

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


@app.route('/device_profile_ops', methods=['GET', 'POST'])
def device_profile_ops():
    """Ops declared by a device profile JSON from device_architectures/."""
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
            arch_file = data.get('architecture_file') or data.get('file')
        else:
            arch_file = request.args.get('architecture_file') or request.args.get('file')

        if not arch_file:
            return jsonify({"error": "architecture_file is required"}), 400

        path = arch_file
        if not os.path.exists(path):
            candidate = os.path.join("device_architectures", os.path.basename(path))
            if os.path.exists(candidate):
                path = candidate
            else:
                return jsonify({"error": f"Device profile not found: {arch_file}"}), 404

        with open(path, "r", encoding="utf-8") as f:
            arch_data = json.load(f)

        supported = list(arch_data.get("supported_ops") or [])
        unsupported = list(arch_data.get("unsupported_ops") or [])
        # stable unique order
        seen = set()
        supported_unique = []
        for op in supported:
            if op not in seen:
                seen.add(op)
                supported_unique.append(op)

        return jsonify({
            "name": arch_data.get("name", os.path.basename(path)),
            "file": path.replace("\\", "/"),
            "cpu_framework": arch_data.get("cpu_framework"),
            "npu_framework": arch_data.get("npu_framework"),
            "supported_ops": supported_unique,
            "unsupported_ops": unsupported,
            "supported_count": len(supported_unique),
            "unsupported_count": len(unsupported),
        })
    except Exception as e:
        logger.error(f"Error reading device profile ops: {e}")
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

def _reset_session_uploads():
    """Session uploads are not kept across server restarts."""
    for folder in (UPLOAD_FOLDER, LOADER_FOLDER):
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    os.makedirs("device_architectures", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("results/exports", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("api_logs", exist_ok=True)
    _reset_session_uploads()

    logger.info("Starting REST API server. ..")
    app.run(host='0.0.0.0', port=5000, debug=False)
