# Model Splitter and Exporter

Этота утилита предоставляет решение для анализа, разделения и экспорта нейронных сетей в различные форматы для разных устройств (CPU, NPU). Поддерживает экспорт в TorchScript, ONNX, TensorFlow Lite, TensorRT, OpenVINO, TVM и TensorFlow.

## Возможности

- **Анализ модели**: Определение поддерживаемых и неподдерживаемых операций
- **Разделение модели**: Автоматическое разделение на части для CPU и NPU
- **Экспорт в различные форматы**: TorchScript, ONNX, TFLite, TensorRT, OpenVINO, TVM, TensorFlow
- **Графический интерфейс**: Удобное управление через GUI
- **REST API**: Веб-интерфейс для удаленного управления
- **Логирование**: Полное логирование всех операций

## Структура проекта

```
model_exporter/
├── model_exporter.py        # Основной класс экспортера
├── model_logic.py           # Логика работы с моделями
├── model_splitter.py        # Класс для разделения моделей
├── model_splitter_gui.py    # Графический интерфейс
├── model_analyzer.py        # Анализ модели
├── log_analizer.py          # Анализ логов
├── api_server.py            # REST API сервер
├── main.py                  # Точка входа
├── requirements.txt         # Зависимости
├── docker-compose.yml       # Docker конфигурация
└── templates/               # HTML шаблоны
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Docker запуск

```bash
docker-compose up -d
```

## Использование

### Графический интерфейс

```bash
python main.py
```

### REST API

Запуск сервера:
```bash
python api_server.py
```

API доступен по адресу: `http://localhost:5000`

### Примеры запросов

#### Загрузка файла
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@model.pt"
```

#### Экспорт модели
```bash
curl -X POST http://localhost:5000/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "results/models/12345_model.pt",
    "format": "onnx",
    "export_dir": "results/exports",
    "model_name": "exported_model"
  }'
```

#### Анализ модели
```bash
curl -X POST http://localhost:5000/analyze_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "results/models/12345_model.pt"
  }'
```

## Конфигурация устройств

Файлы архитектур находятся в `device_architectures/`:

```json
{
  "name": "RK3588S",
  "cpu_framework": "onnx",
  "npu_framework": "openvino",
  "supported_ops": [
    "Conv",
    "Gemm",
    "Relu",
    "MaxPool",
    "AvgPool",
    "BatchNormalization",
    "Add",
    "Sub",
    "Mul",
    "Div"
  ],
}
```

## Логирование

Все операции логируются в `results/logs/`. Логи содержат информацию о:
- Успешных и неудачных экспортах
- Ошибках при работе с моделями
- Статистике по операциям

## Поддерживаемые форматы

| Формат | Описание |
|--------|----------|
| **TorchScript** | Стандартный формат PyTorch |
| **ONNX** | Открытый формат обмена |
| **TFLite** | Формат TensorFlow Lite |
| **TensorRT** | Формат NVIDIA TensorRT |
| **OpenVINO** | Формат Intel OpenVINO |
| **TVM** | Формат Apache TVM |
| **TensorFlow** | Формат TensorFlow |
