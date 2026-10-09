# Задачи архитектурного плана

Сводная задача: [#108](https://github.com/v1docq/FedCore/issues/108). Все новые задачи относятся к ветке `svd_compression`, созданной от коммита `8c0e748d6400fdb28fe17c54e2b3cfbb1a0eda08` из PR #86.

Статус: план утверждён; основа P0–P1 (#109–#115) реализована и подготовлена к проверке изменений. Возможности и ограничения перечислены в [IMPLEMENTATION.md](IMPLEMENTATION.md), результаты — в [VALIDATION.md](VALIDATION.md). Методы P2–P5 остаются в плане. Номер этапа задаёт порядок допуска возможностей. Зависимости между новыми задачами указаны явно. Ссылки на прежние задачи обозначают используемый код или смежное исследование и сами по себе не блокируют работу.

## P0

| Задача | Результат | Зависит от новых задач |
|---|---|---|
| [#109](https://github.com/v1docq/FedCore/issues/109) | [SVD][P0] Разделить цель, решатель, структуру и план преобразования | — |
| [#110](https://github.com/v1docq/FedCore/issues/110) | [SVD][P0] Восстанавливать one/two/three-layer и заменённый корневой слой | [#109](https://github.com/v1docq/FedCore/issues/109) |
| [#111](https://github.com/v1docq/FedCore/issues/111) | [SVD][P0] Сохранять повторные ссылки и общие параметры при замене и загрузке | [#109](https://github.com/v1docq/FedCore/issues/109), [#110](https://github.com/v1docq/FedCore/issues/110) |

## P1

| Задача | Результат | Зависит от новых задач |
|---|---|---|
| [#112](https://github.com/v1docq/FedCore/issues/112) | [SVD][P1] Ввести типы статистик, ресурсный предел и инвалидацию по версии графа | [#109](https://github.com/v1docq/FedCore/issues/109), [#111](https://github.com/v1docq/FedCore/issues/111) |
| [#113](https://github.com/v1docq/FedCore/issues/113) | [SVD][P1] Подключить рабочее weighted-приближение Linear и channel Conv | [#112](https://github.com/v1docq/FedCore/issues/112), [#110](https://github.com/v1docq/FedCore/issues/110), [#111](https://github.com/v1docq/FedCore/issues/111) |
| [#114](https://github.com/v1docq/FedCore/issues/114) | [SVD][P1] Распределять целые структуры по фактическому общему бюджету | [#109](https://github.com/v1docq/FedCore/issues/109), [#111](https://github.com/v1docq/FedCore/issues/111) |
| [#115](https://github.com/v1docq/FedCore/issues/115) | [SVD][P1] Провести новый план через PETRA, экспорт и внешний контракт | [#113](https://github.com/v1docq/FedCore/issues/113), [#114](https://github.com/v1docq/FedCore/issues/114) |

## P2

| Задача | Результат | Зависит от новых задач |
|---|---|---|
| [#116](https://github.com/v1docq/FedCore/issues/116) | [SVD][P2] Реализовать диагональную цель ASVD с явной abs-статистикой | [#112](https://github.com/v1docq/FedCore/issues/112), [#113](https://github.com/v1docq/FedCore/issues/113), [#114](https://github.com/v1docq/FedCore/issues/114), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#117](https://github.com/v1docq/FedCore/issues/117) | [SVD][P2] Собрать индивидуальный эмпирический Fisher и профиль FWSVD | [#112](https://github.com/v1docq/FedCore/issues/112), [#116](https://github.com/v1docq/FedCore/issues/116), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#118](https://github.com/v1docq/FedCore/issues/118) | [SVD][P2] Добавить аффинную PCA AFM и групповой вариант Bolaco | [#112](https://github.com/v1docq/FedCore/issues/112), [#115](https://github.com/v1docq/FedCore/issues/115), [#114](https://github.com/v1docq/FedCore/issues/114) |
| [#119](https://github.com/v1docq/FedCore/issues/119) | [SVD][P2] Задать центрированную метрику и аппаратный профиль FLAR-SVD | [#112](https://github.com/v1docq/FedCore/issues/112), [#113](https://github.com/v1docq/FedCore/issues/113), [#114](https://github.com/v1docq/FedCore/issues/114), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#120](https://github.com/v1docq/FedCore/issues/120) | [SVD][P2] Поддержать остаточные ветви EoRA и mixed-rank ViT | [#113](https://github.com/v1docq/FedCore/issues/113), [#112](https://github.com/v1docq/FedCore/issues/112), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#121](https://github.com/v1docq/FedCore/issues/121) | [SVD][P2] Выполнить DRONE на тонких поддержках с актуальными входами | [#113](https://github.com/v1docq/FedCore/issues/113), [#112](https://github.com/v1docq/FedCore/issues/112), [#114](https://github.com/v1docq/FedCore/issues/114), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#122](https://github.com/v1docq/FedCore/issues/122) | [SVD][P2] Подгонять левый фактор SVD-LLM v1 на текущем входе | [#113](https://github.com/v1docq/FedCore/issues/113), [#112](https://github.com/v1docq/FedCore/issues/112), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#123](https://github.com/v1docq/FedCore/issues/123) | [SVD][P2] Восстанавливать факторы SVD-LLM v5 через существующую LoRA | [#113](https://github.com/v1docq/FedCore/issues/113), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#124](https://github.com/v1docq/FedCore/issues/124) | [SVD][P2] Проверить множители и область распределения рангов SVD-LLM V2 | [#113](https://github.com/v1docq/FedCore/issues/113), [#114](https://github.com/v1docq/FedCore/issues/114), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#125](https://github.com/v1docq/FedCore/issues/125) | [SVD][P2] Реализовать общий физический базис и точную стоимость Basis Sharing | [#113](https://github.com/v1docq/FedCore/issues/113), [#114](https://github.com/v1docq/FedCore/issues/114), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#126](https://github.com/v1docq/FedCore/issues/126) | [SVD][P2] Добавить частотные группы GroupReduce для Embedding и tied head | [#116](https://github.com/v1docq/FedCore/issues/116), [#114](https://github.com/v1docq/FedCore/issues/114), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |

## P3

| Задача | Результат | Зависит от новых задач |
|---|---|---|
| [#127](https://github.com/v1docq/FedCore/issues/127) | [SVD][P3] Разделить критерии и режимы исполнения ESPACE | [#118](https://github.com/v1docq/FedCore/issues/118), [#112](https://github.com/v1docq/FedCore/issues/112), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#128](https://github.com/v1docq/FedCore/issues/128) | [SVD][P3] Добавить профиль PELA с явными feature loss и регуляризатором | [#115](https://github.com/v1docq/FedCore/issues/115), [#110](https://github.com/v1docq/FedCore/issues/110) |
| [#129](https://github.com/v1docq/FedCore/issues/129) | [SVD][P3] Добавить независимые вмешательства LASER и отдельный компактный экспорт | [#109](https://github.com/v1docq/FedCore/issues/109), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#130](https://github.com/v1docq/FedCore/issues/130) | [SVD][P3] Ввести согласованные V/O-профили FLAT и MoDeGPT | [#113](https://github.com/v1docq/FedCore/issues/113), [#114](https://github.com/v1docq/FedCore/issues/114), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#131](https://github.com/v1docq/FedCore/issues/131) | [SVD][P3] Выбирать реальные каналы MLP в MoDeGPT и FLAT | [#118](https://github.com/v1docq/FedCore/issues/118), [#115](https://github.com/v1docq/FedCore/issues/115), [#111](https://github.com/v1docq/FedCore/issues/111), [#122](https://github.com/v1docq/FedCore/issues/122) |
| [#132](https://github.com/v1docq/FedCore/issues/132) | [SVD][P3] Проверить Q/K-преобразования с RoPE, GQA и масштабом логитов | [#130](https://github.com/v1docq/FedCore/issues/130), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#133](https://github.com/v1docq/FedCore/issues/133) | [SVD][P3] Согласовать PCA embedding и все потребители в CTR AFM | [#118](https://github.com/v1docq/FedCore/issues/118), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |

## P4

| Задача | Результат | Зависит от новых задач |
|---|---|---|
| [#134](https://github.com/v1docq/FedCore/issues/134) | [SVD][P4] Реализовать ARS с различением обучаемой маски и префиксного экспорта | [#114](https://github.com/v1docq/FedCore/issues/114), [#115](https://github.com/v1docq/FedCore/issues/115), [#111](https://github.com/v1docq/FedCore/issues/111) |
| [#135](https://github.com/v1docq/FedCore/issues/135) | [SVD][P4] Реализовать Dobi-SVD с отдельными градиентом, IPCA и remapping | [#118](https://github.com/v1docq/FedCore/issues/118), [#114](https://github.com/v1docq/FedCore/issues/114), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#136](https://github.com/v1docq/FedCore/issues/136) | [SVD][P4] Реализовать состояние и физическое удаление групп RankDyna | [#114](https://github.com/v1docq/FedCore/issues/114), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |

## P5

| Задача | Результат | Зависит от новых задач |
|---|---|---|
| [#137](https://github.com/v1docq/FedCore/issues/137) | [SVD][P5] Проверить полный поворот SliceGPT до уменьшения ширины | [#118](https://github.com/v1docq/FedCore/issues/118), [#111](https://github.com/v1docq/FedCore/issues/111), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#138](https://github.com/v1docq/FedCore/issues/138) | [SVD][P5] Создавать LightFormer с полными общими группами и картой переноса | [#111](https://github.com/v1docq/FedCore/issues/111), [#110](https://github.com/v1docq/FedCore/issues/110), [#115](https://github.com/v1docq/FedCore/issues/115) |
| [#139](https://github.com/v1docq/FedCore/issues/139) | [SVD][P5] Добавить LRT как обучение факторизованной архитектуры с нуля | [#109](https://github.com/v1docq/FedCore/issues/109), [#110](https://github.com/v1docq/FedCore/issues/110), [#115](https://github.com/v1docq/FedCore/issues/115) |

## Покрытие методов

| № | Метод | Задачи |
|---|---|---|
| 1 | AFLRC / Bolaco | [#118](https://github.com/v1docq/FedCore/issues/118) |
| 2 | ARS | [#134](https://github.com/v1docq/FedCore/issues/134) |
| 3 | ASVD | [#116](https://github.com/v1docq/FedCore/issues/116) |
| 4 | Basis Sharing | [#111](https://github.com/v1docq/FedCore/issues/111), [#125](https://github.com/v1docq/FedCore/issues/125) |
| 5 | AFM / Features Are Low-Rank | [#118](https://github.com/v1docq/FedCore/issues/118) |
| 6 | Dobi-SVD | [#135](https://github.com/v1docq/FedCore/issues/135) |
| 7 | DRONE | [#121](https://github.com/v1docq/FedCore/issues/121) |
| 8 | RankDyna | [#136](https://github.com/v1docq/FedCore/issues/136) |
| 9 | EoRA | [#120](https://github.com/v1docq/FedCore/issues/120) |
| 10 | ESPACE | [#127](https://github.com/v1docq/FedCore/issues/127) |
| 11 | FLAR-SVD | [#119](https://github.com/v1docq/FedCore/issues/119) |
| 12 | FLAT-LLM | [#130](https://github.com/v1docq/FedCore/issues/130), [#131](https://github.com/v1docq/FedCore/issues/131), [#132](https://github.com/v1docq/FedCore/issues/132) |
| 13 | GroupReduce | [#111](https://github.com/v1docq/FedCore/issues/111), [#126](https://github.com/v1docq/FedCore/issues/126) |
| 14 | FWSVD | [#117](https://github.com/v1docq/FedCore/issues/117) |
| 15 | LightFormer | [#111](https://github.com/v1docq/FedCore/issues/111), [#138](https://github.com/v1docq/FedCore/issues/138) |
| 16 | LRT | [#139](https://github.com/v1docq/FedCore/issues/139) |
| 17 | Activation-aware mixed-rank ViT | [#120](https://github.com/v1docq/FedCore/issues/120) |
| 18 | MoDeGPT | [#130](https://github.com/v1docq/FedCore/issues/130), [#131](https://github.com/v1docq/FedCore/issues/131), [#132](https://github.com/v1docq/FedCore/issues/132) |
| 19 | PELA | [#128](https://github.com/v1docq/FedCore/issues/128) |
| 20 | SliceGPT | [#137](https://github.com/v1docq/FedCore/issues/137) |
| 21 | SVD-LLM | [#113](https://github.com/v1docq/FedCore/issues/113), [#122](https://github.com/v1docq/FedCore/issues/122), [#123](https://github.com/v1docq/FedCore/issues/123) |
| 22 | SVD-LLM V2 | [#124](https://github.com/v1docq/FedCore/issues/124) |
| 23 | LASER | [#129](https://github.com/v1docq/FedCore/issues/129) |
| 24 | CTR AFM | [#133](https://github.com/v1docq/FedCore/issues/133) |

## Существующие исследования и механизмы

- [#100](https://github.com/v1docq/FedCore/issues/100): weighted-SVD, равный ранг, обусловленность, сдвиг распределения и связь локальной ошибки с качеством.
- [#88](https://github.com/v1docq/FedCore/issues/88) / [#89](https://github.com/v1docq/FedCore/issues/89): роли данных, общий запуск, происхождение и исходная модель.
- [#95](https://github.com/v1docq/FedCore/issues/95) / [#96](https://github.com/v1docq/FedCore/issues/96): LM-протокол и измерение конечного артефакта.
- [#99](https://github.com/v1docq/FedCore/issues/99) / [#101](https://github.com/v1docq/FedCore/issues/101) / [#102](https://github.com/v1docq/FedCore/issues/102): цепочки, регуляризация и затраты обучения.
- [#79](https://github.com/v1docq/FedCore/issues/79) / [#103](https://github.com/v1docq/FedCore/issues/103): внешний модуль и его измерение.

Прежние issues не изменялись и не закрывались при публикации этого плана. Детальные ограничения, проверки и границы каждой новой задачи находятся в её описании.
