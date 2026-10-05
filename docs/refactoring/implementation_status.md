# Реализация задач аудита FedCore

Рабочая ветка: `fedcore_refactoring`. База main: `91e573462a01e77ef948a6e9175da5e736908359`. Изменения подготовлены для проверки; исходные issues не закрываются автоматически.

Исправления затрагивают математические операции, основной API, зависимости и внешний вычислительный модуль. Экспериментальные режимы отклоняются явно. Доверенные локальные контрольные точки отделены от непиклируемых входов сервиса.

## Соответствие задачам

| Задача | Результат | Проверка или документ |
|---|---|---|
| [#58](https://github.com/v1docq/FedCore/issues/58) | Единые метаданные, ресурсы wheel/sdist, установка вне исходного дерева | `test_core_packaging.py` |
| [#59](https://github.com/v1docq/FedCore/issues/59) | Автономные CPU-проверки и матрица Windows/Linux, Python 3.10/3.11; удалённый CI требует отдельного результата | `test_root_optional_boundary.py; .github/workflows/unit_test.yml` |
| [#60](https://github.com/v1docq/FedCore/issues/60) | Предсказания и метки собираются за один проход, включая перемешанные данные | `test_core_contracts.py` |
| [#61](https://github.com/v1docq/FedCore/issues/61) | Независимые настройки, проверка типов и комбинаций до запуска ресурсов | `test_core_contracts.py` |
| [#62](https://github.com/v1docq/FedCore/issues/62) | Восстановление архитектуры, состояния, dtype и режимов после очистки кэша | `test_core_contracts.py` |
| [#63](https://github.com/v1docq/FedCore/issues/63) | Рабочие публичные fit/predict/save/load, реальная операция LoRA и малый эволюционный поиск | `test_core_contracts.py` |
| [#64](https://github.com/v1docq/FedCore/issues/64) | Обратимая адаптация FEDOT и явное владение Dask | `test_core_contracts.py; test_root_dask.py` |
| [#65](https://github.com/v1docq/FedCore/issues/65) | Монотонный выбор первого достаточного ранга, нулевой спектр и ранг один | `test_math_layers.py` |
| [#66](https://github.com/v1docq/FedCore/issues/66) | Повторный SVD фактического обученного оператора, знаки и масштаб факторов | `test_math_layers.py` |
| [#67](https://github.com/v1docq/FedCore/issues/67) | Семантика Linear/Embedding/Conv1d/Conv2d, группы, padding, bias и стоимость представления | `test_math_layers.py; tests/unit/low_rank/test_layers.py` |
| [#68](https://github.com/v1docq/FedCore/issues/68) | Устойчивые компоненты потерь, градиенты ученика и реальные пользовательские веса обучения | `test_math_contracts.py; test_math_training.py` |
| [#69](https://github.com/v1docq/FedCore/issues/69) | Регуляризаторы пространственных, нулевых и замороженных факторов; сохранены проверки PR #57 | `test_math_layers.py; test_math_pr57_regularizers.py` |
| [#70](https://github.com/v1docq/FedCore/issues/70) | Восстановление down_proj по калибровочной матрице Грама и псевдообратной; невозможный допуск отклоняется | `test_math_contracts.py` |
| [#71](https://github.com/v1docq/FedCore/issues/71) | Единицы стоимости, направления оптимизации, ограничения и явная недоступность измерения энергии | `test_math_contracts.py; tests/unit/metrics/test_computational.py` |
| [#72](https://github.com/v1docq/FedCore/issues/72) | Единый расчёт Парето и сравнение с независимым эталоном | `test_math_contracts.py` |
| [#73](https://github.com/v1docq/FedCore/issues/73) | Настоящие prepare, forward/backward/step, convert для QAT; независимая исходная модель | `test_math_training.py; tests/unit/quantization/test_quant_utils.py` |
| [#74](https://github.com/v1docq/FedCore/issues/74) | Версионированный каталог поддержки и отказ экспериментальных режимов до изменения модели | `test_root_capabilities.py` |
| [#75](https://github.com/v1docq/FedCore/issues/75) | Строгий формат экспорта и фактическая загрузка TorchScript/ONNX | `test_export_contract.py` |
| [#76](https://github.com/v1docq/FedCore/issues/76) | FX-проверка последовательного графа, сохранение порядка, явный отказ ветвлений | `test_export_partitions.py` |
| [#77](https://github.com/v1docq/FedCore/issues/77) | Вход .fcb без pickle, ограниченные JSON/NPY, токен, пути и размеры | `test_export_security.py; test_export_service.py` |
| [#78](https://github.com/v1docq/FedCore/issues/78) | Неизменяемые роли данных, SQLite-состояния, перезапуск, отмена и владение заданиями | `test_export_runtime.py; test_export_service.py` |
| [#79](https://github.com/v1docq/FedCore/issues/79) | Отдельный вычислительный процесс, типизированный manifest IndustrialTS и односторонний адаптер tdecomp | `test_export_extensions.py; test_export_runtime.py` |
| [#80](https://github.com/v1docq/FedCore/issues/80) | Нативная LoRA-операция, независимая исходная модель, обучение адаптеров и merge/unmerge | `test_math_contracts.py; test_math_training.py; test_core_contracts.py` |
| [#81](https://github.com/v1docq/FedCore/issues/81) | Инварианты индексов, frozen, пересечения, копирование и восстановление состояния | `test_math_contracts.py` |
| [#82](https://github.com/v1docq/FedCore/issues/82) | Исправленная граница сохранённой энергии фильтров | `test_math_layers.py` |
| [#83](https://github.com/v1docq/FedCore/issues/83) | Многоклассовые и регрессионные метрики старого публичного интерфейса | `test_math_contracts.py` |
| [#84](https://github.com/v1docq/FedCore/issues/84) | План приращений шести существующих PR, зависимости и реестр 16 веток с SHA; сами старые PR не переписаны | `docs/refactoring/branch_and_pr_plan.md; branch_inventory.json` |

Имена test_*.py без полного пути относятся к `tests/refactoring`. Таблица показывает реализованное поведение и источник проверки; она не заменяет результаты удалённого CI. Для #84 подготовлен план последовательного переноса приращений: старые PR и ветки сохранены, массовое слияние не выполнено.

## Проверки и ограничения

Общий итог: **309 passed**, без пропущенных сценариев, 128.79 с. Набор включает `tests/refactoring`, старые проверки слоёв, вычислительных метрик и квантования; современный FEDOT подключён явно через два разных Python-окружения. Результат сохранён в JUnit-файле `work/refactoring/integrated-tests-final.xml`. Проверка синтаксиса JavaScript также прошла. Изменения опубликованы в [черновом PR #86](https://github.com/v1docq/FedCore/pull/86). Сборка проверяет wheel, sdist, повторную сборку из sdist и импорт установленного пакета вне исходного дерева. Объявленные зависимости `.[test,export,service]` успешно разрешены без установки; `pip check` не обнаружил конфликтов в проверочном окружении.

Локальная среда: Windows, Python 3.10.0, Torch 2.2.0+cu121 с выполнением проверок на CPU. Современный FEDOT вызывающей стороны проверяется в отдельном процессе через исходное дерево IndustrialTS; вычислительный процесс использует отдельное окружение с опубликованным FEDOT 0.7.5. Код других проектов не менялся.

GPU, большие LLM, TensorRT, аппаратное исполнение NPU и развёртывание Docker не подтверждены. Полный старый набор сетевых примеров не объявляется прошедшим. Отсутствие произвольных дополнительных зависимостей локально проверяется имитацией их отсутствия; чистые Windows/Linux-установки относятся к матрице CI. Задания внешнего модуля ограничивают время и размер входов, но не устанавливают предел всей оперативной памяти средствами ОС.

У внешнего модуля подтверждён профиль уже обученных тензорных моделей для классификации и регрессии. SSA/MSSA, CatBoost, произвольные динамические графы, несколько входов и обучение в вычислительном процессе не заявлены. Происхождение старых external-компонентов проинвентаризировано; отсутствующие лицензии и уведомления не выдуманы.

Перед сохранением выполнен редакторский проход по `edit-russian-technical-text`.

Удалённый CI выявил общий изменяемый `Task` по умолчанию в Python 3.11 и отсутствие `wheel` в наборе сборочных тестов. Добавлены независимая фабрика Task, её регрессионная проверка и явные зависимости setuptools/wheel для `test`. После исправления запускается новая матрица CI; прежние отказы не считаются успешной проверкой.
