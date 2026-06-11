Руководство пользователя FedCore
================================

Данный документ содержит пошаговую инструкцию по установке, запуску и базовому
использованию открытой библиотеки **FedCore**.

FedCore — библиотека для автоматизации сжатия, адаптации и портирования моделей
машинного обучения на основе глубоких нейронных сетей. Библиотека позволяет
применять методы прунинга, квантизации и малорангового разложения, оценивать
качество и вычислительные характеристики моделей, а также готовить модели к
запуску на различных вычислительных архитектурах.

1. Ссылки на проект
-------------------

Репозиторий открытой библиотеки:

.. code-block:: text

   https://github.com/v1docq/FedCore

Файл открытой лицензии:

.. code-block:: text

   https://github.com/v1docq/FedCore/blob/main/LICENSE

Русскоязычная инструкция по работе с библиотекой:

.. code-block:: text

   https://github.com/v1docq/FedCore/blob/main/docs/tutorial_ru.rst

Документация API:

.. code-block:: text

   https://github.com/v1docq/FedCore/tree/main/docs

Примеры использования:

.. code-block:: text

   https://github.com/v1docq/FedCore/tree/main/examples

2. Минимальные технические требования
-------------------------------------

Для базового запуска библиотеки требуется:

.. list-table::
   :header-rows: 1
   :widths: 25 38 37

   * - Компонент
     - Минимальное требование
     - Рекомендуемое значение
   * - Операционная система
     - Linux Ubuntu 20.04 / 22.04 или совместимая ОС
     - Ubuntu 22.04
   * - Python
     - 3.8+
     - 3.10
   * - Оперативная память
     - 8 ГБ
     - 16 ГБ и более
   * - Процессор
     - x86-64 CPU
     - многоядерный x86-64 CPU
   * - GPU
     - не требуется для базовых сценариев
     - NVIDIA GPU с поддержкой CUDA
   * - Git
     - требуется
     - актуальная версия
   * - pip / venv
     - требуется
     - актуальная версия
   * - Docker
     - не требуется для базового запуска
     - Docker и Docker Compose

Для GPU-сценариев дополнительно рекомендуется установить:

* драйвер NVIDIA;
* совместимую версию CUDA;
* PyTorch с поддержкой CUDA.

Для экспорта моделей под отдельные аппаратные платформы могут потребоваться
дополнительные инструменты и SDK: TensorRT, RKNN Toolkit2, ONNX Runtime,
LiteRT / TFLite, OpenVINO, TVM или иные средства, соответствующие целевой
архитектуре.

3. Установка библиотеки
-----------------------

3.1. Клонирование репозитория
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/v1docq/FedCore.git
   cd FedCore

3.2. Создание виртуального окружения
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Для Linux / macOS:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

Для Windows:

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate

3.3. Установка зависимостей
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

3.4. Установка библиотеки в режиме разработки
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

3.5. Установка зависимостей для демонстрационных примеров
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch pandas matplotlib notebook onnx

.. note::

   Для Docker-примера локально Docker-зависимости ставить не нужно, они
   устанавливаются внутри контейнера.

4. Проверка установки
---------------------

.. code-block:: bash

   python -c "import fedcore; print('FedCore installed successfully')"

Если команда завершается без ошибки, библиотека установлена корректно.

5. Базовая логика работы с FedCore
----------------------------------

Типовой сценарий работы с библиотекой включает следующие этапы:

#. подготовка исходной модели и данных;
#. выбор метода сжатия или адаптации модели;
#. настройка конфигурации эксперимента;
#. запуск процедуры сжатия / адаптации;
#. оценка качества и вычислительных характеристик;
#. сохранение или экспорт полученной модели.

Упрощённая схема работы через API:

.. code-block:: python

   from fedcore.api.main import FedCore

   fedcore = FedCore(api_config)

   fedcore.fit(train_data)

   report = fedcore.get_report(test_data)

   print(report)

Конкретный состав ``api_config``, ``train_data`` и ``test_data`` зависит от
выбранной задачи, модели и метода сжатия.

6. Пример 1. Сжатие ResNet-18 методом pruning
---------------------------------------------

Прунинг используется для удаления наименее значимых параметров модели и
уменьшения вычислительной сложности.

Путь к примеру:

.. code-block:: text

   examples/pruning_resnet18/

Запуск скрипта:

.. code-block:: bash

   cd examples/pruning_resnet18
   python run_pruning_resnet18.py

Запуск ноутбука из корня репозитория:

.. code-block:: bash

   jupyter notebook examples/pruning_resnet18/resnet18_pruning_demo.ipynb

Ожидаемый результат:

* создание демонстрационной модели ResNet-18;
* baseline-оценка модели;
* применение pruning;
* короткое дообучение после pruning;
* сравнение числа ненулевых параметров, оценочного размера и времени инференса;
* сохранение результатов в ``results/pruning_resnet18/``.

7. Пример 2. Малоранговое разложение ResNet-18
----------------------------------------------

Малоранговое разложение используется для аппроксимации матриц весов через
представление меньшего ранга. Это позволяет уменьшить количество параметров и
объём памяти, необходимый для хранения модели.

Путь к примеру:

.. code-block:: text

   examples/low_rank_resnet18/

Запуск скрипта:

.. code-block:: bash

   cd examples/low_rank_resnet18
   python run_low_rank_resnet18.py

Запуск ноутбука из корня репозитория:

.. code-block:: bash

   jupyter notebook examples/low_rank_resnet18/resnet18_low_rank_demo.ipynb

Ожидаемый результат:

* создание демонстрационной модели ResNet-18;
* baseline-оценка модели;
* SVD-разложение слоёв ``Conv2d`` и ``Linear``;
* короткое дообучение после замены слоёв;
* сравнение числа параметров, оценочного размера и времени инференса;
* сохранение результатов в ``results/low_rank_resnet18/``.

8. Пример 3. Квантизация ResNet-18
----------------------------------

Квантизация используется для перехода от стандартного FP32-представления
параметров модели к более компактным числовым форматам.

Путь к примеру:

.. code-block:: text

   examples/quantization_resnet18/

Запуск скрипта:

.. code-block:: bash

   cd examples/quantization_resnet18
   python run_quantization_resnet18.py

Запуск ноутбука из корня репозитория:

.. code-block:: bash

   jupyter notebook examples/quantization_resnet18/resnet18_quantization_demo.ipynb

Ожидаемый результат:

* создание quantization-friendly ResNet-18;
* baseline-оценка FP32-модели;
* fusion слоёв ``Conv2d + BatchNorm2d + ReLU``;
* калибровка модели;
* post-training static quantization;
* сравнение размера ``state_dict`` и времени инференса;
* сохранение результатов в ``results/quantization_resnet18/``.

9. Пример 4. Экспорт модели в ONNX
----------------------------------

Экспорт в ONNX используется для подготовки модели к запуску во внешних
фреймворках и на целевых вычислительных платформах.

Путь к примеру:

.. code-block:: text

   examples/export_onnx/

Запуск скрипта:

.. code-block:: bash

   cd examples/export_onnx
   python export_to_onnx.py

Запуск ноутбука из корня репозитория:

.. code-block:: bash

   jupyter notebook examples/export_onnx/resnet18_export_onnx_demo.ipynb

Ожидаемый результат:

* создание демонстрационной модели ResNet-18;
* сохранение PyTorch ``state_dict``;
* экспорт модели в ``.onnx``;
* проверка ONNX-модели через ``onnx.checker``;
* сохранение результатов в ``results/export_onnx/``.

10. Пример 5. Запуск модуля экспорта через Docker
-------------------------------------------------

Docker-пример демонстрирует запуск вспомогательного сервиса экспорта моделей.

Путь к примеру:

.. code-block:: text

   deployment/model_exporter/

Запуск:

.. code-block:: bash

   cd deployment/model_exporter
   docker compose up --build

После запуска сервис будет доступен по адресу:

.. code-block:: text

   http://localhost:5000

Проверка состояния:

.. code-block:: bash

   curl http://localhost:5000/health

Загрузка файла:

.. code-block:: bash

   curl -X POST "http://localhost:5000/upload" \
     -F "file=@demo_model.pt"

Демонстрационный экспорт файла:

.. code-block:: bash

   curl -X POST "http://localhost:5000/export?filename=demo_model.pt&target_format=onnx"

Просмотр файлов:

.. code-block:: bash

   curl http://localhost:5000/files

В demo-версии ``/export`` имитирует экспорт: копирует загруженный файл и меняет
расширение. В промышленной версии этот endpoint может вызывать реальный
pipeline FedCore.

11. Возможные проблемы и способы их устранения
----------------------------------------------

Ошибка: модуль ``fedcore`` не найден
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Убедитесь, что активировано виртуальное окружение и библиотека установлена в
режиме разработки:

.. code-block:: bash

   source .venv/bin/activate
   pip install -e .

Для Windows:

.. code-block:: bash

   .venv\Scripts\activate
   pip install -e .

Ошибка при ONNX-экспорте
~~~~~~~~~~~~~~~~~~~~~~~~

Установите пакет ``onnx``:

.. code-block:: bash

   pip install onnx

Ошибка при запуске Docker-примера
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Проверьте, что Docker запущен:

.. code-block:: bash

   docker --version
   docker compose version

Если порт ``5000`` занят, измените проброс порта в ``docker-compose.yml``.

12. Результаты работы
---------------------

После успешного запуска сценариев пользователь получает:

#. исходную и адаптированную / сжатую модель;
#. метрики качества модели;
#. вычислительные метрики;
#. сведения о размере модели;
#. при необходимости — экспортированный файл модели в целевом формате;
#. логи и CSV-отчёты выполнения эксперимента.

13. Направления прикладного использования
-----------------------------------------

FedCore может применяться в следующих направлениях:

#. промышленные IoT-системы с ограничениями по памяти и вычислительной мощности;
#. автономные системы навигации и детекции объектов;
#. системы компьютерного зрения на edge-устройствах;
#. мобильные и встраиваемые приложения машинного обучения;
#. системы обработки потоковых данных с низкой задержкой;
#. системы прогнозирования энергопотребления;
#. задачи портирования моделей на специализированные аппаратные архитектуры;
#. сценарии, где требуется компактная модель для передачи по ограниченным каналам связи.

14. Лицензия
------------

Доступ к использованию библиотеки предоставляется на условиях открытой лицензии,
размещённой в корне репозитория:

.. code-block:: text

   https://github.com/v1docq/FedCore/blob/main/LICENSE

Перед использованием библиотеки необходимо ознакомиться с условиями лицензии.

15. Краткий чек-лист для первого запуска
----------------------------------------

.. code-block:: bash

   git clone https://github.com/v1docq/FedCore.git
   cd FedCore
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   python -c "import fedcore; print('FedCore installed successfully')"

После этого можно переходить к запуску примеров из каталогов:

.. code-block:: text

   examples/pruning_resnet18/
   examples/low_rank_resnet18/
   examples/quantization_resnet18/
   examples/export_onnx/
   deployment/model_exporter/
