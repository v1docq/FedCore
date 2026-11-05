# Реализация гетерогенного CV-сжатия

Реализация находится в отдельной ветке `integration/heterogeneous-compression`. Merge `7431c1c` объединяет SVD/CV `8cfa54d2abdc1e56fc3dbbd457a06588fe4d6e52` и все локальные quant-коммиты до `56bda18662790981822490324748dcd1b33efb9b`. Исходные рабочие копии сохранены. Сохранены также более поздние предохранители CV и исправление фильтрации невалидных оценок квантования.

Утверждённая постановка и бюджет: [PLAN_EVO_ONLY.md](PLAN_EVO_ONLY.md).

Подготовлен [следующий пакет послойной диагностики](A100_LAYERWISE_DIAGNOSTIC.md): восемь калибровочных изображений, 12 наблюдаемых выходов, 28 вызовов и одна сборка в предлагаемом окне до 600 секунд. Локально прошли 159 проверок, одна пропущена. Это подготовка; новое GPU-окно не запускалось, критерий принятия Q-only сохранён.

05.11.2026 завершено отдельное сравнение сохранённых Q-only ONNX/engine на 2000 калибровочных изображениях за 28,75 секунды. Исходный отказ воспроизведён: 9201 из 20 000 логитов нарушают допуск, argmax совпадает в 99,6% случаев. Q-only остаётся непринятым. GPU освобождена; вторичная ошибка наблюдателя и её последующее CPU-исправление описаны в [отчёте диагностики](A100_QONLY_DIAGNOSTIC_RESULT.md). Суммарно учтено 1175,81 секунды из 7200.

Разрешённый повтор 04.11.2026 завершил обучение M0 и три рецепта. Dense+200 SGD получил 94,05%, SVD — 94,00%, P75 — 93,525% на Vsearch. Q-only собран в INT8, но не прошёл согласование логитов с ONNX Runtime. Статус пилота `incomplete`; суммарный расход с первым запуском — 1147,06 с. [Результаты повтора](A100_CONTAINER_REPEAT_RESULT.md).

04.11.2026 выполнен первый удалённый контейнерный запуск. Dense прошёл TensorRT-допуск, P→SVD→PTQ остановился на неподдержанном INT32 bias. Израсходовано 39,43 секунды, SGD не начинались. Исправление сохраняет FP32 bias и проверено на CPU. [Результат и квитанции](A100_CONTAINER_PILOT_RESULT.md), [контейнерный профиль](CONTAINER_A100.md). Один повтор до 7160 секунд предложен владельцу.

03.11.2026 пользователь разрешил один новый пилот A100 до двух часов. Исполнитель и локальные проверки подготовлены; SSH недоступен, аппаратный допуск и вычисления ещё не начаты. Границы, рецепты и точка продолжения: [A100_PILOT.md](A100_PILOT.md).

## Публичные компоненты

| Файл | Назначение |
|---|---|
| `fedcore/algorithm/heterogeneous/contracts.py` | Представления, версии, координаты и допустимая композиция областей |
| `cv.py` | Конечная G1, компиляция, сериализация и согласованное построение float-кандидата |
| `policy.py`, `search.py` | E0/E1/E2 поверх общего GOLEM/SPEA2, одинаковый оплаченный префикс |
| `backend.py` | ONNX Q/DQ, ORT, TensorRT 10, самостоятельная загрузка и реальный размер пакета |
| `selection.py`, `protocol.py` | Выбор финалистов, неполные исходы и счётный бюджет |
| `fedcore/experiments/heterogeneous_data.py` | Только train-корпус CIFAR-10, роли и детерминированное преобразование изображений |
| `heterogeneous_runtime.py` | Фиксированное восстановление, оценка загруженного артефакта и журнал частичных расходов |
| `heterogeneous_series.py` | Полная последовательность независимых M0, контролей, поисков, общей фиксации, Vreport и аппаратных границ |
| `heterogeneous_host.py`, `heterogeneous_direct_host.py` | Ограниченный транспорт и доказуемая принадлежность наблюдаемых процессов |
| `heterogeneous_device.py`, `heterogeneous_deadline.py` | Один GPU UUID и внешний срок собственной systemd/cgroup unit |

## Безопасная подготовка манифеста

После установки пакета команда ниже только записывает конфигурацию и 27 заданий; обучение, CUDA, SSH и systemd не запускаются:

```powershell
python scripts/plan_heterogeneous_cv.py --output panel-plan.json
```

Экспортные зависимости уже выделены в существующую optional-группу `export`: ONNX 1.16.2 и ONNX Runtime 1.18.1. Они не устанавливаются при импорте. Для реального TensorRT нужен отдельный проверенный GPU-профиль; автоматического обновления основного окружения нет.

## Пример соединения одного поиска

Следующий код описывает вычислительный вход для будущего разрешённого запуска. Это не команда возобновления остановленного кластера.

```python
from fedcore.algorithm.heterogeneous import default_alternatives, prepare_space
from fedcore.algorithm.heterogeneous.backend import BackendSpec
from fedcore.algorithm.heterogeneous.policy import SearchArm
from fedcore.algorithm.heterogeneous.search import common_initial, run_search
from fedcore.algorithm.low_rank.compression_graph import RuntimeOptions
from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig
from fedcore.experiments.heterogeneous_runtime import (
    CandidateEvaluator, RecoveryConfig, recovery_tape,
)

# source: завершённая независимая M0 на CPU, в eval-режиме.
# data: CifarTrainingData; deadline: общий абсолютный срок time.perf_counter.
calibration = data.calibration_tensor()
space = prepare_space(source, calibration, alternatives=default_alternatives(),
    options=RuntimeOptions(max_workspace_bytes=512*1024**2,
                           max_peak_bytes=4*1024**3))
config = GrammarSearchConfig(seed=101, population_size=8, generations=64,
    max_evaluations=64, max_proposals=640, max_seconds=search_seconds)
recovery = RecoveryConfig()
tape = recovery_tape(data.roles.fit, model_seed=model_seed, config=recovery)
backend = BackendSpec(runtime="tensorrt", device_uuid=target_gpu_uuid)
evaluator = CandidateEvaluator(source, calibration, tape, data.training,
    data.search_batches, fresh_job_directory, deadline, backend,
    validation_identity=data.fingerprint+":Vsearch:"+data.roles.sha256,
    validation_rows=4000, recovery_config=recovery, device="cuda:0")
result = run_search(space, config, evaluator, arm=SearchArm.STAGED,
    fitness_context=evaluator.validation_identity,
    initial_genotypes=common_initial(space, 8, config.seed),
    context_guard=check_frozen_source_and_data,
    absolute_deadline=deadline)
```

Входы должны быть подготовлены контроллером с утверждённым бюджетом. Восстановление и исходное обучение обязаны связать GPU UUID до первой инициализации CUDA. Отдельный каталог нужен для каждого поиска: прошлые артефакты и квитанции не перезаписываются.

## Учёт неуспехов

`progress.jsonl` хранит начало попытки, переходы стадий, начало и завершение каждого SGD-шага. `receipt.json` атомарно сохраняет исход, фактические расходы и качество. При аварийной остановке незавершённый шаг остаётся явно начатым, а не исчезает из расхода. Общий поисковый счётчик отдельно списывает попытку до компиляции.

`load_deployment` сверяет хеш основного файла и состав пакета. Для PTQ сохраняются источник M0, версия восстановленных весов, калибровочные данные и координаты. Реальная проверка ONNX/ORT на CPU не считается аппаратным допуском A100.

## Условия аппаратного допуска

Для проведённого пилота владелец разрешил [контейнерный профиль GNU timeout](CONTAINER_A100.md), который сохраняет ограничение `host_pid_identity_proven=false`. Он требует свободной A100, привязки CUDA по UUID, единственного неизменного процесса GPU и внешнего срока для фиксированной группы собственных процессов. Каждое отдельное окно и повтор имеют собственное разрешение; неиспользованный остаток не разрешает автоматический новый запуск. Часы основной панели определяются после измерения стоимости пилота.

`run_series` соединяет вычислительные этапы и требует исходный срок контроллера, отдельный предел поиска и заранее зафиксированный протокол измерений. Число повторов, прогревы и критерии A/A определяются после пилота. Внешние аппаратные функции измерения и pre/post A/A передаются явно; исполнитель проверяет их сырые результаты и не заменяет их таймером CPU. Вариант `SeriesEffects` с подставными исполнителями предназначен для проверок счётчиков и порядка стадий и помечается как не являющийся аппаратной проверкой.

Отдельный direct-host профиль использует собственные guard/worker внутри подтверждённой cgroup: `SystemdDeadlineLauncher` готовит границу, `DirectHostBindingProvider` связывает процессы. Этот профиль не следует смешивать с ограниченными гарантиями контейнерного запуска. `query_device_gpu_rows` наблюдает объявленный UUID, а `require_cuda_binding` сравнивает UUID текущего контекста через CUDA Driver API. Эти API не обнаруживают доступ к кластеру и не предоставляют разрешение на запуск. Источник определения UUID: [CUDA Driver API](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-driver-api/group__CUDA__DEVICE.html).

Документация API: [TensorRT Q/DQ](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html), [Engine Inspector](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html). Применение конкретной версии проверяется на её фактическом формате inspector; неизвестный формат приводит к отказу.
