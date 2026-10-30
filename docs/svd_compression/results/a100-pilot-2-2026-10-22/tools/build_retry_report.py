"""Build an offline report of the authorized repeat from recounted evidence."""
import argparse
import json
from html import escape
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('raw',type=Path)
parser.add_argument('analysis',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def number(value,n=2):return f'{value:.{n}f}'.replace('.',',')
a=read(args.analysis/'analysis.json')
r=read(args.raw/'report.json') if a['worker_report_present'] else read(args.analysis/'derived-interruption-outcome.json')
receipt=read(args.raw/'launch-receipt-pilot-2-2026-10-22.json')
attempts=read(args.analysis/'attempts.json')
assert a['verification_consistent'],'Do not publish inconsistent calculations'
guard=a['repeat_guard']
names={'dense_original':'Исходная плотная M0','dense_recovery':'Плотная + 200 обновлений',
    'fixed_prune':'Каналы 75%','fixed_svd':'SVD 50%','fixed_hybrid':'Каналы → SVD',
    'random_finalist':'Финалист случайного поиска','evolution_finalist':'Финалист эволюции'}
def roles(row):return ' / '.join(names.get(role,role) for role in row['roles'])
compact=[row for row in a['final_rows'] if not any(role.startswith('dense') for role in row['roles'])]
accepted=sum(row['accepted'] for row in compact)
lead=(f"Повтор завершил протокол: {accepted} из {len(compact)} уникальных компактных моделей выполнили совместный критерий хранения и качества. "
      'Одна исходная модель и одна пара поисков не подтверждают статистического превосходства эволюции.'
      if a['status']=='completed' else f"Повтор остановлен: {a['stop_reason'] or a['failure'] or a['status']}. Ниже приведены только фактически завершённые этапы.")
if not r.get('initial_aa',{}).get('accepted'):
    latency_note='Начальный допуск A/A не пройден или не завершён. Вывод об ускорении не допускается.'
else:
    latency_note=(f"Начальный допуск A/A пройден. Допущенных строк конечных парных сравнений: {len(a['latency_rows'])}." if r.get('latency') else 'Начальный допуск A/A пройден; конечные измерения восстановленных моделей не начинались. Вывода об ускорении нет.')
genotypes={row['solution_id']:row['genotype'] for records in r.get('all_short_attempts',{}).values() for row in records}
payload={'analysis':a,'report':r,'attempts':attempts,'receipt':receipt,'genotypes':genotypes,
    'lead':lead,'latency_note':latency_note}
template=(Path(__file__).parent/'a100-retry-report.template.html').read_text(encoding='utf-8')
if not a['worker_report_present']:
    template=template.replace('Пик PyTorch reserved','Сохранённый максимум reserved').replace('raw/report.json','raw/pilot-2-supervision.json')
    template=template.replace('Общий предел — 240 минут,', 'Итогового отчёта работника нет; полный пик CUDA до остановки неизвестен. Общий предел — 240 минут,')
args.output.mkdir(parents=True,exist_ok=False)
(args.output/'a100-pilot-2-report.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')),encoding='utf-8')

lines=['# CV-пилот A100: разрешённая повторная попытка','',
    '[PR #140](https://github.com/v1docq/FedCore/pull/140) · ResNet18 / CIFAR-10 · 22.10.2026','',lead,'',
    '## Протокол и полный расход','',
    'Исходный код, wheel, численный протокол и среда сохранены. С разрешения пользователя начат один повтор с обучением с нуля в новом каталоге `pilot-2`. Первая попытка остановлена при внешней конкуренции после 58/80 эпох; её журналы и веса не заменены. Точное продолжение с частичных весов не выполнялось.','',
    '| Показатель | Значение |','| --- | ---: |',
    f"| Первая попытка, занятый слот | {number(receipt['prior_charged_seconds']/60)} мин |",
    f"| Повтор, внешний полный слот | {number(guard['elapsed_seconds']/60)} мин |",
    f"| Общий учтённый расход | {number(guard['total_charged_seconds']/60)} / 240 мин |",
    f"| Предельная длительность повтора | {number(receipt['repeat_slot_seconds']/60)} мин |",
    f"| Статус повтора | `{a['status']}` |",
    f"| Полные эпохи исходного обучения повтора | {len(a['epochs'])}/80 |",
    f"| Пик PyTorch allocated / reserved | {number(a['peak_allocated_bytes']/1024**3)} / {number(a['peak_reserved_bytes']/1024**3)} GiB |",
    f"| Максимум RSS по внешним наблюдениям | {number(a['peak_rss_bytes']/1024**3)} GiB |",'',
    'Время включает построение, передачу данных, сохранение и проверки во время GPU-слота. Счётчики PyTorch не включают всю память драйвера, внешняя RSS-телеметрия дискретна. Подготовка окружения и CPU-проверки относятся к инженерной подготовке, отдельно от занятого GPU-слота. Дополнительных автоматических попыток нет.','',
    ('CUDA-память взята из финального отчёта работника.' if a['worker_report_present'] else 'Итогового отчёта работника нет. Показанные максимумы CUDA ограничены сохранёнными сообщениями; полный пик и хвост обновлений до принудительной остановки неизвестны.'),'',
    'Разделение официальной обучающей части CIFAR-10: 44 000 изображений для SGD, 1000 для калибровки и 5000 для поиска. Официальный тест из 10 000 изображений открывается после фиксации выбранных файлов. Восемь BasicBlock, семь допустимых рецептов каждого: плотный, физическое сокращение каналов, SVD и последовательность сокращения и SVD. FP32, TF32 выключен, одна A100 40 ГБ, четыре CPU-потока.','']
baseline_path=args.raw/'baseline.json'
if baseline_path.is_file():
    baseline=read(baseline_path)
    lines+=['## Исходная модель','',f"80 эпох SGD; top-1 на поисковой выборке — {number(100*baseline['validation']['top1'])}%, CE — {number(baseline['validation']['cross_entropy'],4)}. Физические тензоры всей M0: {baseline['tensor_bytes']} байт ({number(baseline['tensor_bytes']/1048576)} MiB).",'']
short=r.get('all_short_attempts',{})
if short:
    candidates=[]
    if short.get('random'):candidates.append(('Плотная после калибровки BN',short['random'][0]))
    for label in ('fixed_prune','fixed_svd','fixed_hybrid'):
        if short.get(label):candidates.append((names[label],short[label][0]))
    best={}
    for run,label in (('random','Лучший короткий R'),('evolution','Лучший короткий E')):
        row=next((row for row in short.get(run,[]) if row['solution_id']==a['selected_finalists'].get(run)),None)
        if row is not None:best[run]=row;candidates.append((label,row))
    merged={}
    for label,row in candidates:
        entry=merged.setdefault(row['solution_id'],{'row':row,'labels':[]})
        entry['labels'].append(label)
    lines+=['## Что показали выполненные короткие оценки','',
        '| Вариант | Физические байты | Доля M0 | CE поисковой выборки | Top-1 поисковой выборки |',
        '| --- | ---: | ---: | ---: | ---: |']
    for entry in merged.values():
        row=entry['row']
        lines.append(f"| {' / '.join(entry['labels'])} | {row['tensor_bytes']} | {number(100*row['tensor_bytes']/a['source_tensor_bytes'])}% | {number(row['cross_entropy'],6)} | {number(100*row['top1'])}% |")
    lines+=['','Все эти оценки выполнены без SGD после одинаковой калибровки BatchNorm на 1000 изображениях. У плотного короткого контроля эта процедура тоже меняет статистики BN: его нельзя смешивать с исходной M0 до калибровки. Таблица относится к поисковой выборке и не подтверждает конечный допуск качества.','']
    if {'random','evolution'}<=set(best):
        first,second=best['random'],best['evolution']
        lines+=[f"В этой трассе E дал CE {number(second['cross_entropy'],6)} против {number(first['cross_entropy'],6)} у R; разность top-1 — {number(100*(second['top1']-first['top1']))} п.п., разность хранения E−R — {second['tensor_bytes']-first['tensor_bytes']} байт. Это описательное сравнение двух выбранных коротких оценок, без независимого теста и статистического подтверждения.",'']
    lines+=['Работник не успел выбрать финалистов и начать восстановление. Показанные лучшие допустимые варианты пересчитаны после остановки по сохранённым записям; исходный результат не дописывался. Общие четыре начальных рецепта совпали по составу и численным метрикам, но эволюционный исполнитель оценил их в обратном порядке. Стоимость отдельно учтена в каждом поиске.','']
lines+=['## Случайный и эволюционный поиск','',
    'Общие цели — поисковая CE и физические тензорные байты всей модели. До 12 уникальных коротких попыток, до 48 предложений и до 30 минут на поиск; одинаковый префикс из четырёх графов оценивается независимо. Короткая оценка не содержит SGD и использует одинаковую калибровку BatchNorm. Потолок 80% M0 применяется при продвижении, а не при генерации.','',
    '| Поток | Учтённые попытки | Успешные своевременные | Доля M0 у лучшего допустимого | CE лучшего допустимого | Учтённое время оценок, с |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for run in a['attempt_counts']:
    rows=[row for row in attempts if row['run']==run]
    good=[row for row in rows if row['status']=='succeeded' and row['timely']]
    finalist=a['selected_finalists'].get(run)
    best=next((row for row in good if row['solution_id']==finalist),None)
    ratio='—' if best is None else number(100*best['tensor_bytes']/a['source_tensor_bytes'])+'%'
    ce='—' if best is None else number(best['cross_entropy'],4)
    lines.append(f"| {run} | {len(rows)} | {len(good)} | {ratio} | {ce} | {number(sum(row['wall_seconds'] for row in rows))} |")
lines+=['', 'Сумма времени оценок не включает всю организационную работу поиска; общий расход учитывается внешним слотом. Итоговая точность после восстановления отвечает на другой вопрос, чем короткая поисковая CE. Совпадение финалистов с префиксом или фиксированным контролем показано совмещёнными ролями, без искусственного увеличения числа независимых моделей.','',
    '## Качество после одинакового восстановления','',
    'По плану до шести уникальных графов должны получить по 200 обновлений SGD по одной ленте данных. В этой попытке восстановление не началось. Критерий: не более 80% тензорных байтов M0 и не более 2 п.п. потери top-1 относительно плотной модели после таких же 200 обновлений. Исходная плотная M0 и плотная после восстановления — разные контрольные точки.','',
    '| Роли модели | Тензоры, MiB | Доля M0 | Файл, MiB | Top-1 теста | Δ к плотной после восстановления, п.п. | Совместный критерий |',
    '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
for row in a['final_rows']:
    verdict='контроль' if any(role.startswith('dense') for role in row['roles']) else 'выполнен' if row['accepted'] else 'не выполнен'
    lines.append(f"| {roles(row)} | {number(row['tensor_bytes']/1048576)} | {number(100*row['ratio'])}% | {number(row['file_bytes']/1048576)} | {number(100*row['top1'])}% | {number(row['delta_pp_vs_dense_recovery'])} | {verdict} |")
if not a['final_rows']:lines+=['','Конечные оценки отсутствуют; пропуски не заменены нулями.']
lines+=['','## Измерения времени GPU','',latency_note,'',
    '| Модель | Размер пакета | Измеритель | Плотная, мс | Вариант, мс | Медиана B/A | Наблюдаемое A/B | Диапазон пар B/A |',
    '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
by_id={row['artifact_id']:row for row in a['final_rows']}
for row in a['latency_rows']:
    clock='CUDA Events' if row['clock']=='cuda_event' else 'синхронизированное время'
    lines.append(f"| {roles(by_id[row['artifact_id']])} | {row['batch']} | {clock} | {number(row['reference_ms'],3)} | {number(row['candidate_ms'],3)} | {number(row['median_paired_B_over_A'],3)} | {number(row['observed_speedup'],3)}× | {number(row['min_pair_ratio'],3)}–{number(row['max_pair_ratio'],3)} |")
lines+=['', 'План предусматривает 20 пар AB/BA на режим с отдельным начальным и конечным A/A. A/B — наблюдаемое отношение времени в данном режиме, диапазоны пар не являются доверительными интервалами. Уменьшение хранения само по себе не доказывает ускорения.','',
    '## Проверка и ограничения','',
    f"Независимый пересчёт: {a['checks']} согласованных проверок, {a['prediction_recounts']['files']} файлов предсказаний и {a['prediction_recounts']['rows']} строк. Использованы NumPy и стандартная библиотека, без исполнения модели. Фактическое покрытие перечислено в checks.json; незавершённые стадии не считаются проверенными. Для имеющихся записей сверены физические байты с Torch ZIP-хранилищами, идентификаторы, сроки и счётчики попыток; отбор, восстановление и конечные критерии сверяются только при наличии соответствующих свидетельств.", '',
    'Поисковая выборка участвует в отборе и не является независимым подтверждением качества. Одна исходная модель и одна пара поисков не показывают устойчивого превосходства метода. Меню ограничено внутренними каналами BasicBlock; произвольные архитектуры и полная композиция Sparse-методов не проверены. Настоящие CUDA sparse/INT8-ядра не использовались. Непрошедший A/A запрещает ранжирование по скорости, сохраняя отдельный вывод о качестве и хранении.','',
    '## Происхождение','',f"Код: `{receipt['source_commit']}`.",f"Wheel: `{receipt['wheel_sha256']}`.",f"Численный протокол: `{receipt['protocol_sha256']}`.", '',
    '[HTML с данными и графиками](a100-pilot-2-report.html) · [первая прерванная попытка](A100_PILOT_1.md) · [исходный итог](results/a100-pilot-2-2026-10-22/raw/report.json) · [пересчёт](results/a100-pilot-2-2026-10-22/analysis/analysis.json).','']
if a['failure']:
    lines+=['## Причина ошибки и подготовленное исправление','',
        f"Исходный запуск сохраняет статус `failed`: `{a['failure']}`. Журнал эволюции подтверждает завершение 12 оценок и остановку по лимиту оценок; сохранение итогового словаря оборвалось на списке типизированных CVGenotype. Частично записанный `search/evolution.json.tmp` сохранён; полноценный `search/evolution.json` не создавался задним числом.", '',
        'Локальное исправление преобразует генотипы в их явный переносимый формат только при записи двух отчётов. Типизированный результат общего поиска и механизм fingerprints не меняются. Четыре новые регрессии покрывают обе записи, отсутствие мутаций, отказ для некорректных победителей и пустой архив. Целевая CPU-серия: 199 passed, 2 skipped. Исправление не применялось к завершившемуся GPU-запуску.', '',
        'Выполненный расход двух попыток — 55,66 минуты из 240; остаток — 184,34 минуты. Новый GPU-запуск не разрешён автоматически. Следующее предложение — отдельное дозавершение из сохранённой полной M0: проверить все хеши, воспроизвести выбор из готовых 27 оценок, выполнить пять уникальных восстановлений по 200 обновлений, зафиксировать шесть конечных артефактов и лишь затем открыть независимый тест и измерения времени. Повторное исходное обучение и повтор поисков для этого не нужны. Продолжение требует отдельного разрешения и проверки свободной A100.','']
if not a['worker_report_present']:
    lines=[line.replace('raw/report.json','raw/pilot-2-supervision.json').replace('Пик PyTorch allocated / reserved','Сохранённые максимумы PyTorch allocated / reserved') for line in lines]
if not a['selection_reported_by_worker']:
    lines+=['Пересчитанный лучший кандидат частичного потока не считается финалистом, выбранным работником.','']
(args.output/'A100_PILOT_2.md').write_text('\n'.join(lines),encoding='utf-8')
if short:
    table_rows=[]
    for entry in merged.values():
        row=entry['row']
        table_rows.append('<tr><td>'+escape(' / '.join(entry['labels']))+'</td><td>'+number(100*row['tensor_bytes']/a['source_tensor_bytes'])+'%</td><td>'+number(row['cross_entropy'],6)+'</td><td>'+number(100*row['top1'])+'%</td></tr>')
    card='<section class="card"><h2>Выполненные короткие оценки</h2><p>Исходная M0 до калибровки BN: 95,06%. Ниже — отдельные оценки после одинаковой калибровки BN, без SGD, на поисковой выборке.</p><div class="scroll"><table><thead><tr><th>Вариант</th><th>Хранение M0</th><th>CE</th><th>Top-1</th></tr></thead><tbody>'+''.join(table_rows)+'</tbody></table></div><p>Все 27 оценок завершены. Восстановление и независимый тест не начинались. Лучшие R/E пересчитаны после остановки; это предварительные наблюдения одной пары поисков.</p></section>'
    path=args.output/'a100-pilot-2-report.html'
    html=path.read_text(encoding='utf-8')
    html=html.replace('<section class="card"><h2>Что предусматривал план</h2>',card+'<section class="card"><h2>Что предусматривал план</h2>',1)
    path.write_text(html,encoding='utf-8')
print(json.dumps({'status':a['status'],'checks':a['checks'],'compact_artifacts':len(compact),'accepted':accepted}))
