"""Build an offline technical report from independently recounted evidence."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('raw',type=Path)
parser.add_argument('analysis',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
args.output.mkdir(exist_ok=False,parents=True)
def load(path):return json.loads(path.read_text(encoding='utf-8'))
a=load(args.analysis/'analysis.json')
r=load(args.raw/'report.json') if a['worker_report_present'] else load(args.analysis/'derived-interruption-outcome.json')
attempts=load(args.analysis/'attempts.json')
receipt=load(args.raw/'launch-receipt-2026-10-22.json')
roles={'dense_original':'Плотная M0','dense_recovery':'Плотная + 200 обновлений','fixed_prune':'Каналы 75%',
       'fixed_svd':'SVD 50%','fixed_hybrid':'Каналы → SVD','random_finalist':'Финалист Random','evolution_finalist':'Финалист Evolution'}
def names(row):return ' / '.join(roles.get(value,value) for value in row['roles'])
def number(value,digits=2):return f'{value:.{digits}f}'.replace('.',',')
compact=[row for row in a['final_rows'] if not any(value.startswith('dense') for value in row['roles'])]
accepted=sum(row['accepted'] for row in compact)
if a['status']=='completed':
    lead=f"Пилот завершил протокол. Критерии хранения и качества выполнили {accepted} из {len(compact)} уникальных компактных артефактов. Это одна исходная модель и одна пара поисков, без статистического подтверждения превосходства эволюции."
else:
    lead=f"Пилот прерван при появлении второго GPU-процесса. Завершены {len(a['epochs'])} из 80 эпох исходного обучения; допуск исходной модели, поиск графов сжатия, восстановление и измерения скорости не выполнялись."
if not r.get('initial_aa'):
    latency_note='Измерения задержки не выполнялись: пилот не дошёл до допуска.'
elif not r['initial_aa'].get('accepted'):
    latency_note='Начальный A/A не пройден. Выводы об ускорении запрещены; качество и физические байты рассматриваются отдельно.'
else:
    latency_note=f"Начальный A/A пройден. В таблице только допущенные сравнения; наблюдения для не прошедших конечный A/A графов не ранжируются по скорости. Строк: {len(a['latency_rows'])}."
genotypes={record['solution_id']:record['genotype'] for records in r.get('all_short_attempts',{}).values() for record in records}
payload={'analysis':a,'report':r,'attempts':attempts,'receipt':receipt,'lead':lead,'latency_note':latency_note,'genotypes':genotypes}
template=(Path(__file__).parent/'a100-pilot-report.template.html').read_text(encoding='utf-8')
embedded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
(args.output/'a100-pilot-1-report.html').write_text(template.replace('__DATA__',embedded),encoding='utf-8')
lines=['# Первый CV-пилот на A100: прерванное обучение ResNet18 / CIFAR-10','',
 'Дата: 22.10.2026. [PR #140](https://github.com/v1docq/FedCore/pull/140).',
 '',lead,'','## Условия и сохранённые границы','',
 'Исходная модель обучается 80 полных эпох на 44 000 изображениях. 1000 изображений отведены для калибровки, 5000 — для поиска. Официальные 10 000 test открываются только после фиксации выбранных моделей. Меню каждого из восьми BasicBlock содержит семь допустимых рецептов: плотный, сокращение каналов, SVD и согласованную последовательность сокращения и SVD.',
 '', 'Цели короткой оценки — CE и физические тензорные байты всей модели. Потолок продвижения равен 80% M0. Восстановление — 200 одинаковых обновлений SGD из M0 по общей ленте данных. Итоговый допуск качества — не более 2 п.п. потери к плотной модели после тех же 200 обновлений.',
 '', 'Одна A100 40 ГБ, FP32, TF32 выключен, четыре CPU-потока. Ограничения: 240 минут общего слота, 36 GiB PyTorch VRAM, 16 GiB RSS. Код, команда и окружение не менялись; автоматических повторов нет.',
 '', '## Фактический расход','',
 '| Показатель | Значение |','| --- | ---: |',
 f"| Статус | `{a['status']}` |",f"| Учтённое время слота | {number(a['elapsed_seconds']/60)} мин |",
 f"| Полное время надзора | {number(a['supervised_seconds']/60)} мин |",f"| Завершённые эпохи M0 | {len(a['epochs'])}/80 |",
 f"| Максимум PyTorch allocated в сохранённых сообщениях | {number(a['peak_allocated_bytes']/1024**3)} GiB |",
 f"| Максимум PyTorch reserved в сохранённых сообщениях | {number(a['peak_reserved_bytes']/1024**3)} GiB |",
 f"| Пик RSS работника | {number(a['peak_rss_bytes']/1024**3)} GiB |",'',
 'Показатели PyTorch не включают всю служебную память CUDA-драйвера. Пик RSS получен внешним наблюдением процесса; снимки NVIDIA не выдаются за непрерывно измеренный пик всей карты.']
if not a['worker_report_present']:
    lines+=['','## Обстоятельства остановки и сохранённые веса','',
        'Запуск начат 22.10.2026 в 16:28:50 МСК после освобождения карты и проверки отсутствия вычислительных GPU-процессов. Около 16:51:50 МСК надзор обнаружил два процесса вместо одного и завершил только собственную группу. Причина — `external_gpu_contention`, код работника — `-9`; ошибок телеметрии нет.', '',
        'Работник не успел создать `report.json`. Этот файл не восстановлен задним числом: производный итог хранится в `analysis/derived-interruption-outcome.json`. Снимок NVIDIA в 16:55:52 МСК показывает PID `2778980`, 578 MiB и загрузку GPU 79%; владелец из контейнера неизвестен. PID в момент остановки не записан, поэтому поздний снимок не устанавливает личность процесса, вызвавшего остановку.', '',
        f"Завершённые эпохи содержат {a['completed_epoch_updates']} обновления. Последнее сообщение на 59-й эпохе содержит {a['last_observed_updates']} обновления; между ним и SIGKILL могли выполниться дополнительные шаги. Точный конечный счётчик неизвестен. Максимумы CUDA взяты из сохранённых накопительных счётчиков и не доказывают полный пик до завершения процесса или потребности последующих этапов.", '',
        'Обучающая CE снизилась с 2,51348 до 0,07396 на 58-й эпохе. Проверка на поисковой выборке и независимом тесте не проводилась. Снижение обучающей ошибки не подтверждает прохождение допуска top-1 ≥90%.', '',
        '| Снимок модели | Размер файла, байт | Тензорные хранилища, байт | SHA-256 |',
        '| --- | ---: | ---: | --- |']
    inventory=load(args.raw/'remote-file-inventory.json')
    for epoch in (20,40):
        name=f'partial/baseline_epoch_{epoch:03d}.pt'
        proof=inventory['files'][name]
        lines.append(f"| `{name}` | {proof['bytes']} | {proof['torch_zip_storage_bytes']} | `{proof['sha256']}` |")
    lines+=['', 'Веса остаются на A100 в `/root/fedcore_sparse_cv_20261021/results/pilot-1/`. Снимки содержат состояние модели и архитектуру, но не состояние SGD с momentum, планировщика скорости обучения и генераторов случайных чисел. Точное продолжение с 40-й эпохи не поддерживается. Снимка после 58-й эпохи нет.']
admission=args.raw/'baseline_training_admission.json'
if admission.is_file():
    proof=load(admission)
    lines+=['','## Допуск исходной модели','',
        f"Top-1 на 5000 поисковых изображениях: **{number(100*proof['validation']['top1'])}%**. Допуск исходного обучения: `{proof['accepted']}`."]
lines+=['','## Поиск и восстановление','',
    '| Поток | Учтённые попытки | Статусы |','| --- | ---: | --- |']
for run,counts in a['attempt_counts'].items():
    lines.append(f"| `{run}` | {sum(counts.values())} | {', '.join(key+': '+str(value) for key,value in counts.items())} |")
lines+=['',('Оба поиска имеют одинаковые пределы и независимо оценивают общий префикс из четырёх графов. Число фактических оценок указано выше.' if a['attempt_counts'] else 'Поиск, короткие контроли и восстановление не запускались. Ни одного кандидата не оценено.'),'',
    '| Роль | Тензоры, MiB | Доля M0 | Файл, MiB | Top-1 test | Δ к dense-recovery, п.п. | Критерий |',
    '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
for row in a['final_rows']:
    verdict='контроль' if any(role.startswith('dense') for role in row['roles']) else 'выполнен' if row['accepted'] else 'не выполнен'
    lines.append(f"| {names(row)} | {number(row['tensor_bytes']/1048576)} | {number(100*row['ratio'])}% | {number(row['file_bytes']/1048576)} | {number(100*row['top1'])}% | {number(row['delta_pp_vs_dense_recovery'])} | {verdict} |")
if not a['final_rows']:lines+=['','Итоговые test-оценки не выполнялись; отсутствующие результаты не заменены нулями.']
lines+=['','## Измерения GPU','',latency_note,'',
    '| Артефакт | Batch | Часы | Dense, мс | Вариант, мс | Медиана B/A | A/B | Диапазон пар B/A |',
    '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
mapping={row['artifact_id']:row for row in a['final_rows']}
for row in a['latency_rows']:
    clock='CUDA Events' if row['clock']=='cuda_event' else 'синхронизированное время'
    lines.append(f"| {names(mapping[row['artifact_id']])} | {row['batch']} | {clock} | {number(row['reference_ms'],3)} | {number(row['candidate_ms'],3)} | {number(row['median_paired_B_over_A'],3)} | {number(row['observed_speedup'],3)}× | {number(row['min_pair_ratio'],3)}–{number(row['max_pair_ratio'],3)} |")
lines+=['',('Диапазоны — наблюдавшиеся 20 пар, а не доверительные интервалы. Сравнения с тем же плотным артефактом являются контролем, а не эффектом сжатия.' if a['latency_rows'] else 'По плану предполагались 20 пар AB/BA с отдельным допуском A/A. В этой попытке ни одна пара не измерена.'),'',
 '## Независимая проверка','',
 f"Пересчёт согласован: `{a['verification_consistent']}`. Выполнено {a['checks']} проверок; пересчитаны {a['prediction_recounts']['files']} файлов предсказаний и {a['prediction_recounts']['rows']} строк. Проверка использует NumPy и стандартную библиотеку без импорта FedCore/PyTorch и без повторного исполнения моделей.",
 '', ('Проверены разделение данных, паспорт протокола и пакета, бюджеты, канонические идентификаторы, сроки, выбор финалистов, общая лента восстановления, CE/top-1 по строкам предсказаний и конечные критерии.' if a['worker_report_present'] else 'Проверены контрольные суммы скопированных файлов, паспорт кода и протокола, разделение данных без пересечений, последовательность эпох и обновлений, внешний предел времени, причина остановки и отсутствие последующих этапов. Файлы предсказаний не созданы. Отбор, восстановление, конечные критерии качества и A/A в этом запуске не проверены. Нулевое число наблюдений не подтверждает эти свойства.'), '',
 'Физические байты независимо рассчитаны по формам полного CifarResNet18: 44 734 408 байт. Они совпадают с суммой размеров хранилищ внутри Torch ZIP двух частичных снимков. Проверка размеров не заменяет допуск качества и численную проверку загрузки. Бинарные веса локально не загружались; они остаются на A100.','',
 '## Ограничения и следующий шаг','',
 'Сравнение методов ещё не выполнено. Выводов об эффективности Sparse/SVD, преимуществе эволюции, качестве сжатия и ускорении нет. По плану поисковая выборка служит для отбора, независимый тест открывается после фиксации артефактов. Текущий профиль использует веса Conv и физическое сокращение каналов; CUDA sparse/INT8-ядра не заявляются.',
 '', f"Из общего лимита 240 минут учтено {number(a['supervised_seconds']/60)} минуты. Арифметический остаток — {number((14400-a['supervised_seconds'])/60)} минуты ({number((14400-a['supervised_seconds'])/3600)} GPU-часа). Остаток не разрешает автоматический повтор. Для новой попытки нужны отдельное разрешение пользователя и свободное окно без автоматического запуска посторонних GPU-задач. Предыдущий расход сохраняется в общем бюджете; полное повторное обучение или добавление точного возобновления следует выбрать до новых вычислений."]
if a['failure'] or a['stop_reason']:
    lines+=['','## Причина остановки','',f"`{a['stop_reason'] or a['failure']}`"]
lines+=['','## Происхождение','',f"Код: `{receipt['source_commit']}`.",f"Протокол: `{receipt['protocol_sha256']}`.",
    f"Wheel: `{receipt['wheel_sha256']}`.",'',
    '[Автономный HTML](a100-pilot-1-report.html) · [исходный итог надзора](results/a100-pilot-1-2026-10-22/raw/pilot-1-supervision.json) · [пересчёт](results/a100-pilot-1-2026-10-22/analysis/analysis.json).','']
(args.output/'A100_PILOT_1.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'output':str(args.output),'status':a['status'],'compact_accepted':accepted,'compact_artifacts':len(compact)}))
