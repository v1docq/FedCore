"""Render the completed third packet from its immutable numerical summaries."""
from pathlib import Path
import json
import statistics
import hashlib

RAW = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Third-Package-2026-10-15')
DOCS = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Comparisons-2026-10-13/checkout/docs/svd_compression')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fmt(value, digits=3):
    return f'{value:.{digits}f}'.replace('.', ',')


def render_html(a, p, counts):
    names = {'svd_factor': 'SVD: факторы Conv2 + Linear r16', 'svd_materialized': 'SVD: плотная сборка Conv2 + Linear r16',
             'weighted_factor': 'Взвешенный SVD: факторы Conv2 + Linear r16',
             'weighted_materialized': 'Взвешенный SVD: плотная сборка Conv2 + Linear r16', 'dense': 'Плотная CE400'}
    rendering = {'counts': counts, 'names': names, 'profiles': a['profiles'],
                 'comparisons': a['comparisons'], 'graphs': a['prepared']['graphs'],
                 'source': p['source']['sha256'], 'protocol': p['protocol_sha256']}
    encoded = json.dumps(rendering, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    (RAW / 'rendering-data.json').write_bytes((encoded + '\n').encode('utf-8'))
    template = (Path(__file__).parent / 'third-template.html').read_text(encoding='utf-8')
    html = template.replace('DATA_PAYLOAD', encoded.replace('<', '\\u003c'))
    (DOCS / 'third-local-package-infographic.html').write_bytes(html.encode('utf-8'))


def main():
    a, p, run = read(RAW / 'analysis/analysis.json'), read(RAW / 'protocol.json'), read(RAW / 'run.json')
    ledger = read(RAW / 'ledger.json')
    records = {}
    for profile in run['profiles']:
        for stage, ref in profile['stages'].items():
            records[(profile['device'], profile['batch'], stage)] = read(RAW / ref['relative_path'])
    counts = {'processes': len(ledger['attempts']), 'timing_processes': len(records),
              'pair_units': sum(len(r['records']) for r in records.values())}
    for metric in ('wall_ms', 'full_path_ms', 'cuda_event_ms', 'throughput_seconds'):
        counts[metric] = sum(len(side.get(metric, [])) for r in records.values()
                            for unit in r['records'] for side in unit['sides'].values())
    render_html(a, p, counts)
    if not (RAW / 'validation.json').is_file():
        print(json.dumps({'counts': counts, 'status': 'html_rendered_final_validation_pending'}, sort_keys=True))
        return
    v = read(RAW / 'validation.json')
    def row(device, batch, subject, metric):
        return next(x for x in a['comparisons'] if (x['device'], x['batch'], x['subject'], x['metric']) ==
                    (device, batch, subject, metric))
    def ratio(r):
        return f"{fmt(r['median'])} [{fmt(r['min'])}–{fmt(r['max'])}]"
    def metric_table(subjects):
        out = ['| Вариант | Завершённый вызов p50 | Полный путь p50 | Пропускная способность |',
               '| --- | --- | --- | --- |']
        for label, subject in subjects:
            values = [ratio(row('cpu', 64, subject, m)) for m in
                      ('wall_ms', 'full_path_ms', 'throughput_seconds')]
            out.append('| ' + ' | '.join([label] + values) + ' |')
        return '\n'.join(out)
    gates = ['| Режим | Предварительный контроль | Основная серия | Допуск сравнения |',
             '| --- | --- | --- | --- |']
    for profile in a['profiles']:
        stages = profile['stages']
        status = lambda stage: ('пройден' if stages[stage]['gate']['passed'] else 'не пройден') if stage in stages else 'не запускалась'
        eligible = stages.get('main', {}).get('ranking_eligible', False)
        label = ('CPU, один поток' if profile['device'] == 'cpu' else 'RTX 4080') + f", пакет {profile['batch']}"
        gates.append(f"| {label} | {status('pilot')} | {status('main')} | {'да' if eligible else 'нет'} |")
    quality = ['| Граф | Байты тензоров | Валидация | Прежняя тестовая выборка |', '| --- | ---: | --- | --- |']
    names = {'svd_factor': 'SVD: факторы Conv2 + Linear r16', 'svd_materialized': 'SVD: плотная сборка Conv2 + Linear r16',
             'weighted_factor': 'Взвешенный SVD: факторы Conv2 + Linear r16',
             'weighted_materialized': 'Взвешенный SVD: плотная сборка Conv2 + Linear r16', 'dense': 'Плотная CE400'}
    for name in ('dense', 'svd_factor', 'svd_materialized', 'weighted_factor', 'weighted_materialized'):
        graph = a['prepared']['graphs'][name]
        q = graph['known_quality']
        quality.append(f"| {names[name]} | {graph['tensor_bytes']:,} | {q['validation']['correct']}/{q['validation']['n']} | {q['test']['correct']}/{q['test']['n']} |".replace(',', ' '))
    absolute = []
    raw = records[('cpu', 64, 'main')]
    for name in ('svd', 'weighted'):
        pairs = [r for r in raw['records'] if r['unit']['subject'] == f'{name}_factor_vs_materialized']
        absolute.append((name, *(statistics.median(statistics.median(r['sides'][side]['wall_ms']) for r in pairs) for side in ('a', 'b'))))
    gpu = next(x for x in a['profiles'] if x['device'] == 'cuda:0' and x['batch'] == 64)
    failed_gpu = [c for c in gpu['stages']['main']['gate']['checks'] if not c['passed']]
    max_parity = max(s['max_abs'] for case in a['prepared']['parity'].values() for s in case['splits'].values())
    text = f'''# Третий локальный пакет: упрощение Conv и контроль измерений GPU

Завершён 15.10.2026, [PR #140](https://github.com/v1docq/FedCore/pull/140). [Интерактивная инфографика](third-local-package-infographic.html), [исходные данные и пересчёт](results/third-local-2026-10-15/README.md).

На CPU с одним потоком и пакетом 64 замена двух факторов Conv2 одной плотной свёрткой с тем же восстановленным весом уменьшила время полной сжатой CNN примерно на 8–9%. Хранение увеличилось на 4096 байт, с **78 312 до 82 408 байт**. Предварительный и итоговый контроли этого режима прошли. Для GPU воспроизводимость пока не подтверждена: предварительный контроль пакета 64 прошёл, а итоговый — нет. Его результаты остаются диагностическими.

## Что сравнивалось

Две сохранённые DigitsCNN — `conv2_svd_linear16` и `conv2_weighted_linear16` — уже прошли 400 восстановительных обновлений CE в первом пакете. Первый Conv и Linear r16 здесь сохраняются; второй Conv либо исполняется двумя факторами, либо заменяется стандартной свёрткой с весом, собранным из этих же факторов. Отдельный эталон — плотная CNN после 400 обновлений продолжения обучения. Новых обучений, данных и кластерных запусков нет.

Факторизованный Conv2 выполняет пространственную свёртку 16→8, затем точечную 8→16. Плотная сборка выполняет одну свёртку 16→16 с тем же математическим оператором и смещением. В вычислениях FP32 порядок операций меняется, поэтому проверяется численное совпадение с `rtol=1e-4`, `atol=1e-5`, а не побитовое равенство выходов.

| Представление всей модели | Хранение тензоров, байт | MAC на изображение |
| --- | ---: | ---: |
| Плотная CNN после CE400 | 274 920 | 222 848 |
| Conv2 r8 и Linear r16 | 78 312 | 109 184 |
| Плотная сборка Conv2 и Linear r16 | 82 408 | 174 720 |

После сборки сохраняется **29,97%** хранения плотной CNN. Число MAC возрастает, но удаляется одна операция Conv. Наблюдаемое время подтверждает пользу такого представления только в указанном режиме CPU; оно не объясняет причинный вклад каждого оператора.

## Эквивалентность и качество

SHA256 плотного веса совпал с SHA256 произведения текущих факторов в обоих вариантах. Смещение и все остальные тензоры сохранены. Экспортированные TorchScript-графы проверены после загрузки. Максимальное различие выходов на прежних валидационной и тестовой выборках — **{max_parity:.8g}**; решения `argmax` совпали для всех объектов.

{chr(10).join(quality)}

Это проверка сохранности уже полученного результата на данных разработки. Одна исходная модель (seed 42) и ранее использованная тестовая выборка не дают новой независимой оценки обобщения.

## Протокол и допуск

Четыре режима проверяются независимо: CPU с одним потоком и RTX 4080, каждый с пакетами 1 и 64. Все графы — FP32/TorchScript; TF32 и `cudnn.benchmark` отключены, `cudnn.deterministic=True`. Десять объектов TorchScript в каждом процессе представляют пять графов и по одной независимо загруженной копии каждого для контроля одинаковых весов. Каждая копия прогревается не менее 50 вызовов и 0,25 с, с пределом 10 000 вызовов.

До основных сравнений выполняются восемь контрольных блоков каждого графа — четыре AB и четыре BA. Для **каждого** обязательного показателя **каждого** графа заранее заданы три условия: медиана отношения 0,95–1,05; не менее 7/8 отношений в пределах 0,85–1,15; отношение медиан AB и BA в пределах 0,90–1,10. Это эксплуатационные допуски, а не тест статистической значимости.

Основная серия запускается только после допуска своего режима. В ней 20 блоков, каждый содержит два сравнения факторов с плотной сборкой, четыре сравнения с плотной CNN и пять контролей одинакового графа. Итоговые границы медианы и эффекта порядка те же; в широкий диапазон должны попасть не менее 18/20 контрольных отношений. Неполные блоки не заменяются. Сбой контроля исключает ранжирование всего режима.

{chr(10).join(gates)}

Все **{counts['processes']} процессов** завершились с кодом 0 без нарушения ресурсных лимитов: один процесс подготовки и {counts['timing_processes']} процессов измерений. Статус успешного процесса отделён от допуска по устойчивости. Сохранены **{counts['pair_units']} полных парных единиц**; основных серий было две, по 220 единиц каждая. CPU1 и GPU1 остановились после своих 40 предварительных контролей. GPU64 сохранил основную серию целиком; {len(failed_gpu)}/20 проверок «граф × показатель» не прошли.

Для каждой стороны сохраняются 10 вызовов на показатель в предварительной серии и 20 в основной, плюс блок из 50 вызовов для пропускной способности. Архив содержит {counts['wall_ms']} времён завершённого вызова, {counts['full_path_ms']} времён полного пути, {counts['cuda_event_ms']} интервалов CUDA Events и {counts['throughput_seconds']} блоков пропускной способности. Профилировщик во время измерений не используется.

## Результат CPU: та же функция, другое представление Conv2

Ниже отношение `T_факторы / T_плотная_сборка`; больше 1 означает меньшее время плотной сборки. Для пропускной способности это то же отношение длительностей равных блоков по 50 вызовов. Скобки — наблюдавшийся диапазон 20 парных блоков, без доверительного интервала.

{metric_table([('SVD', 'svd_factor_vs_materialized'), ('Взвешенный SVD', 'weighted_factor_vs_materialized')])}

Медианы блочных p50 завершённого вызова: SVD — **{fmt(absolute[0][1], 4)}→{fmt(absolute[0][2], 4)} мс**, взвешенный SVD — **{fmt(absolute[1][1], 4)}→{fmt(absolute[1][2], 4)} мс**. Это времена пакета из 64 изображений, а не одного изображения. Парные отношения рассчитываются до агрегирования; отношение двух указанных медиан может немного отличаться от медианы отношений.

## Сравнение с плотной CNN

Здесь отношение `T_плотная_CE400 / T_кандидат`. Сопоставляются разные обученные функции с одинаковым бюджетом восстановления; это не контроль математической эквивалентности.

{metric_table([('SVD: факторы Conv2 + Linear r16', 'dense_vs_svd_factor'), ('SVD: плотная сборка Conv2 + Linear r16', 'dense_vs_svd_materialized'), ('Взвешенный SVD: факторы Conv2 + Linear r16', 'dense_vs_weighted_factor'), ('Взвешенный SVD: плотная сборка Conv2 + Linear r16', 'dense_vs_weighted_materialized')])}

В этом локальном режиме плотная сборка Conv2 с сохранённым Linear r16 даёт отношение времени около **1,18×** к плотной CE400 при 29,97% её хранения. Различие между обычным и взвешенным SVD здесь не устанавливает преимущество одного алгоритма: обе серии используют по одной сохранённой модели, без новых обучающих повторов.

## Что установлено для GPU

Перенос сравнения в один прогретый процесс и чередование AB/BA оказались недостаточны для итогового допуска этой серии. При пакете 1 основные сравнения не запускались. При пакете 64 прошёл короткий контроль, но более длинная серия выявила нарушения его допусков. Поэтому численные отношения GPU в данных и инфографике помечены как диагностические; ускорение и ранжирование не заявляются.

Завершённый вызов, полный путь с передачей входа и выхода, CUDA Events и пропускная способность измеряются раздельно. CUDA Events могут включать промежутки между запусками с CPU и не являются чистым временем ядер. Частоты, P-state, мощность и температура сохранены в журналах ресурсов; они описывают наблюдения, но не доказывают причину нестабильности. Поведение драйвера, влияние наблюдателя и фоновых приложений отдельно не идентифицированы.

## Ресурсы, проверка и воспроизведение

Перед запуском процесса и после его завершения проверяются три последовательных тихих наблюдения: CPU≤25% и свободная RAM≥6 GiB; для процессов GPU дополнительно GPU≤10% и свободная GPU-память≥2 GiB. Активность собственного процесса во время работы не классифицируется как внешний шум. Контролируются собственное дерево процессов с пределом RSS 12 GiB, доступная RAM, CUDA-память и время. Все процессы завершены до итоговых тестов.

Научный учёт — **{ledger['charged_seconds']:.3f} с** из 7200 с, включая резерв 300 с на предварительные проверки. Из него подготовка — {ledger['compute_by_stage_seconds']['prepare']:.3f} с, предварительные процессы — {ledger['compute_by_stage_seconds']['pilot']:.3f} с, основные — {ledger['compute_by_stage_seconds']['main']:.3f} с. Ожидание наблюдений — {ledger['admission_wait_seconds']:.3f} с, включая {ledger['extra_noise_wait_seconds']:.3f} с дополнительного ожидания; обязательная пауза после выхода процессов — {ledger['mandatory_drain_seconds']:.1f} с. Итоговый учёт с проверками публикуется отдельно и не изменяет исходный научный журнал.

Итоговый прогон прежнего автономного набора: **{v['counts']['passed']} passed, {v['counts']['skipped']} skipped**, ошибок и падений нет. Все 104 новые проверки входят в общий набор: 90 зафиксированы до измерений, ещё 14 проверяют отдельную программу переноса результатов. [Журнал проверки](results/third-local-2026-10-15/validation.json) и [независимая проверка](results/third-local-2026-10-15/independent-review.json) входят в публикацию. До запуска зафиксированы 264 научных исходных файла.

Первый общий прогон в песочнице дал 1754 passed, 2 failed, 3 skipped: два прежних теста Dask не смогли запустить локальный цикл событий. Оба сбоя воспроизвелись отдельно. Минимальная программа без FedCore локализовала ожидание в `socketpair/accept`; при выполнении той же программы вне песочницы запуск занял около 0,016 с. Итоговый общий прогон поэтому выполнен вне песочницы с тем же кодом и прежними ограничениями ресурсов. [Первый исход](results/third-local-2026-10-15/validation-attempt-01.json) и [диагностика Dask](results/third-local-2026-10-15/qa/dask-repro-01/diagnostic-summary.json) сохранены; настройки системы и зависимости не менялись.

- SHA256 научных исходников Git-LF: `{p['source']['sha256']}`.
- SHA256 протокола: `{p['protocol_sha256']}`.
- [Паспорт публикации](results/third-local-2026-10-15/publication-manifest.json) связывает опубликованные байты с локальными оригиналами.

[Итоговый учёт](results/third-local-2026-10-15/final-ledger.json) включает оба полных прогона и отдельный резерв на диагностику и публикацию: **1371,553 с из 7200 с**. Первоначальный научный журнал сохранён без изменения.

Встроенные данные HTML сверены с исходным анализом, синтаксис JavaScript проверен без выполнения браузерного кода. Визуальная проверка не проведена: встроенный браузер отклонил открытие `file://`. Это ограничение отмечено в [журнале проверки HTML](results/third-local-2026-10-15/html-validation.json).

Численный пересчёт не требует PyTorch. На Windows исходный анализ запускается так:

```powershell
python -I -S scripts/analyze_svd_third_package.py --root docs/svd_compression/results/third-local-2026-10-15 --destination third-recalculated
```

Индекс исходного Windows-запуска хранит обратные косые черты в относительных путях. Отдельный [переносимый пересчёт](../../scripts/replay_svd_third_package_portable.py) создаёт временное представление индекса с нормализованными разделителями и вызывает тот же зафиксированный анализ. Исходные JSON, протокол, веса и численные данные при этом не меняются. Преобразование путей имеет собственный журнал; вспомогательная программа добавлена после измерений и не входит в научный снимок.

```text
python -I -S scripts/replay_svd_third_package_portable.py --root docs/svd_compression/results/third-local-2026-10-15 --destination third-portable-recalculated
```

## Следующий шаг

Для локального CPU64 разумно оставить плотный Conv2 и сжатый Linear r16 как рабочий вариант этой CNN. Решение для CPU1 и GPU откладывается до воспроизводимого контроля. На A100/Linux следующий ограниченный пилот должен сначала проверить устойчивость одинаковых графов и получение реальных событий CUDA-ядер; затем можно переходить к обученной ResNet18/CIFAR10 с несколькими начальными состояниями обучения и равным хранением всей модели. Повторное обучение и широкая матрица точностей не устраняют выявленную проблему измерительного тракта сами по себе.
'''
    (DOCS / 'THIRD_LOCAL_PACKAGE.md').write_bytes(text.encode('utf-8'))
    rendering = {'counts': counts, 'names': names, 'profiles': a['profiles'],
                 'comparisons': a['comparisons'], 'graphs': a['prepared']['graphs'],
                 'source': p['source']['sha256'], 'protocol': p['protocol_sha256']}
    (RAW / 'rendering-data.json').write_bytes((json.dumps(rendering, ensure_ascii=False,
          sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps({'counts': counts, 'validation': v['counts'],
                      'report_bytes': len(text.encode('utf-8'))}, sort_keys=True))


if __name__ == '__main__':
    main()
