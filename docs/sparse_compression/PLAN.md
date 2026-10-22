# [SPARSE] Внедрить методы разреженности и гибридные представления на основе PR #140

<!-- fedcore-sparse-2026-10-13:sp00 -->
Основание: [PR #140](https://github.com/v1docq/FedCore/pull/140), коммит [`844114f`](https://github.com/v1docq/FedCore/commit/844114fc968c436e030b93e4ccac6f954ae76ff0). Рабочая ветка: [`fedcore_sparse`](https://github.com/v1docq/FedCore/tree/fedcore_sparse), созданная от актуальной вершины `svd_compression`.

## Текущее состояние

PR #140 даёт общую основу SVD, статистики, бюджеты, ссылки, обучение и экспорт. Набор #108–139 относится к существующей серии SVD; семи методов этой серии в нём нет.

## Что сделать

Вести карту этапов и проверенных профилей для TEDEPR, FlexiGPT, HASSLE-free, LoSparse, MoE-I², SoLA и PruneNet. Зафиксировать для каждого источник/версию, вид преобразования, данные, оптимизируемую цель, рабочее представление и границы. Статус «механизм проверен» не должен автоматически означать «экспорт поддержан» или «получено ускорение».

## Критерии готовности и минимальная демонстрация

- [ ] Таблица охватывает все семь методов и ссылается на [#142](https://github.com/v1docq/FedCore/issues/142)–[#156](https://github.com/v1docq/FedCore/issues/156); для каждого опубликованы минимум математическая проверка, ограничение реализации и команда малого воспроизводимого примера либо причина блокировки. TEDEPR остаётся кандидатом до восстановления спецификации.
- [ ] Основная серия запускается из установленного пакета вне исходного дерева.

## Зависимости

Это сводная задача, без собственного вычислительного блокера. Завершение определяется готовностью согласованных рабочих этапов. #108 и PR #140 — основание, а не требования повторной реализации.

## Границы задачи

Полные результаты статей, большие LLM/GPU и сравнение эффективности PETRA не обещаются в этой сводной задаче.

## Анализ PR #140 и принятые решения

- Планы, полные вторые моменты, численные решения, StorageCost/allocate, атомарная замена, обучение и локальные контрольные точки уже реализованы. Расширяем существующие границы; открытые #109–115 не означают, что требуется начать заново.
- [Планы и ExplicitMask](https://github.com/v1docq/FedCore/blob/844114fc968c436e030b93e4ccac6f954ae76ff0/fedcore/algorithm/low_rank/plans.py) не задают готовую маску остатка, FFN или экспертов. Пустые ветви получают явную семантику без ослабления прежних SVD-проверок.
- [_prepare_mlp](https://github.com/v1docq/FedCore/blob/844114fc968c436e030b93e4ccac6f954ae76ff0/fedcore/algorithm/low_rank/coupled_execution.py) подгоняет down; PruneNet нужен прямой срез, SoLA — точное разбиение до факторизации. Это отдельный результат [#143](https://github.com/v1docq/FedCore/issues/143).
- [ResidualLinear](https://github.com/v1docq/FedCore/blob/844114fc968c436e030b93e4ccac6f954ae76ff0/fedcore/algorithm/low_rank/structured_layers.py) и топология дают основу с поправкой. FlexiGPT сохраняет физическую общность и обучаемость основы; LoSparse получает компактный остаток с прежней внешней шириной.
- [Внешний формат](https://github.com/v1docq/FedCore/blob/844114fc968c436e030b93e4ccac6f954ae76ff0/fedcore/external_runtime/models.py) запрещает общие ссылки, а контракт v3 ограничен P2, CPU и тензорными классификацией/регрессией. Внутренняя контрольная точка, самостоятельный экспорт и внешний вызов допускаются отдельно.
- [CI исходного коммита](https://github.com/v1docq/FedCore/actions/runs/38270655436) завершился успешно. Эта проверка статическая: тесты не запускались заново, sparse-методы пока не реализованы, ускорение не измерено. PR остаётся открытым черновиком и не считается слитым в main.

## Рабочие задачи

| Issue | Результат |
|---|---|
| [#142](https://github.com/v1docq/FedCore/issues/142) | Описать гибридные структуры, допустимость и их фактический бюджет |
| [#143](https://github.com/v1docq/FedCore/issues/143) | Добавить прямой выбор и точное разбиение связанных каналов FFN |
| [#144](https://github.com/v1docq/FedCore/issues/144) | Исполнять и сохранять физический оператор N:M плюс малоранговая поправка |
| [#145](https://github.com/v1docq/FedCore/issues/145) | Исполнять малоранговую ветвь с компактным остатком выбранных выходов |
| [#146](https://github.com/v1docq/FedCore/issues/146) | Реализовать HASSLE-free с полной метрикой, SparseGPT и проверяемым масштабированием |
| [#147](https://github.com/v1docq/FedCore/issues/147) | Реализовать обучение LoSparse от точной функции до компактного остатка |
| [#148](https://github.com/v1docq/FedCore/issues/148) | Реализовать FlexiGPT с обучаемой общей основой и индивидуальными поправками позиций |
| [#149](https://github.com/v1docq/FedCore/issues/149) | Реализовать SoLA как точное разбиение FFN и сжатие marginal-ветви |
| [#150](https://github.com/v1docq/FedCore/issues/150) | Добавить физическое удаление MoE-экспертов с согласованной маршрутизацией |
| [#151](https://github.com/v1docq/FedCore/issues/151) | Реализовать оценки MoE-I² и поиск наборов с Block-wise KT-Reception Field |
| [#152](https://github.com/v1docq/FedCore/issues/152) | Сжать оставшихся экспертов MoE-I² и восстановить компактную модель |
| [#153](https://github.com/v1docq/FedCore/issues/153) | Реализовать PruneNet с корректной вероятностью действия и выбранной опубликованной целью |
| [#154](https://github.com/v1docq/FedCore/issues/154) | Восстановить спецификацию TEDEPR до реализации метода |
| [#155](https://github.com/v1docq/FedCore/issues/155) | Провести новые представления через контрольные точки, публичное исполнение и явный допуск экспорта |
| [#156](https://github.com/v1docq/FedCore/issues/156) | Подготовить общую серию малых демонстраций и проверку полезности конечных артефактов |

## Очерёдность

1. [#142](https://github.com/v1docq/FedCore/issues/142) — общий контракт, потребность в данных, стоимость и ограничения. Параллельно можно восстанавливать источник TEDEPR в [#154](https://github.com/v1docq/FedCore/issues/154).
2. [#143](https://github.com/v1docq/FedCore/issues/143), [#144](https://github.com/v1docq/FedCore/issues/144), [#145](https://github.com/v1docq/FedCore/issues/145), [#150](https://github.com/v1docq/FedCore/issues/150) — независимые физические преобразования после общего контракта.
3. HASSLE-free после [#144](https://github.com/v1docq/FedCore/issues/144), LoSparse после [#145](https://github.com/v1docq/FedCore/issues/145), SoLA/PruneNet после [#143](https://github.com/v1docq/FedCore/issues/143). FlexiGPT переиспользует готовую общую основу. MoE: [#150](https://github.com/v1docq/FedCore/issues/150) → [#151](https://github.com/v1docq/FedCore/issues/151) → автоматический вариант [#152](https://github.com/v1docq/FedCore/issues/152); первая численная проверка последнего возможна по явно заданной карте без поиска.
4. [#155](https://github.com/v1docq/FedCore/issues/155) и [#156](https://github.com/v1docq/FedCore/issues/156) добавляют допуск и демонстрации по готовым профилям. Они не требуют ждать TEDEPR или завершения всех методов для первого ограниченного результата.

## Источники методов

- [TEDEPR: аннотация, DOI 10.1109/TPAMI.2024.3437421](https://pubmed.ncbi.nlm.nih.gov/39388327/) — полного текста нет, алгоритмическая реализация не допущена.
- [FlexiGPT, arXiv:2501.14713v2](https://arxiv.org/pdf/2501.14713v2).
- [HASSLE-free, arXiv:2502.00899v1](https://arxiv.org/pdf/2502.00899v1).
- [LoSparse, arXiv:2306.11222v2](https://arxiv.org/pdf/2306.11222v2).
- [MoE-I², arXiv:2411.01016v1](https://arxiv.org/html/2411.01016v1).
- [SoLA, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33923/36078).
- [PruneNet, arXiv:2501.15296v3](https://arxiv.org/html/2501.15296v3).
