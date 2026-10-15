import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

// Rebuilds the report snapshot from the independently checked, portable analysis.
// Arguments: analysis.json, full summary.json, protocol.json, app/src/data.json.
const [analysisFile, summaryFile, protocolFile, outputFile] = process.argv.slice(2);
if (!outputFile) throw new Error("Expected analysis, summary, protocol, output paths");
const read = file => JSON.parse(fs.readFileSync(file, "utf8"));
const analysis = read(analysisFile), full = read(summaryFile), protocol = read(protocolFile);
const audit = read(path.join(path.dirname(summaryFile),"final-audit.json"));
const previous = read(outputFile);
const base = "https://github.com/v1docq/FedCore/blob/svd_compression/docs/svd_compression/results/local-2026-10-13/";
const names = {svd:"SVD",weighted_svd:"Weighted-SVD",asvd:"ASVD",afm:"AFM",baseline:"Исходная модель"};
const cohortId = row => row.cohort === "joint_channel_conv_linear" ? "cnn-chain" : `${row.architecture}-linear`;
const cohortName = row => row.architecture === "mlp" ? "MLP · Linear" : row.cohort === "joint_channel_conv_linear" ? "CNN · Conv + Linear" : "CNN · Linear";
const rankOrder = row => row.cohort === "joint_channel_conv_linear" ? ["features.0","features.2","classifier.0"] : Object.keys(row.ranks ?? {});
const ranks = row => rankOrder(row).map(operator=>row.ranks?.[operator]).join(" / ");
const fullById = new Map(full.rows.map(row => [`${row.architecture}-s${row.seed}-${row.case_id}`,row]));
function transformLayers(transform) {
  return [...(transform?.weighted_transform?.layers ?? []), ...(transform?.activation_transform?.layers ?? []),
    ...(transform?.steps ?? []).flatMap(transformLayers)];
}
const normalize = row => ({...row, scenario: row.architecture === "mlp" ? "BreastCancerBigMLP" : "DigitsCNN",
  method_id: row.method, method: names[row.method], cohort_id: cohortId(row), cohort: cohortName(row), ranks: ranks(row),
  rank_paths:rankOrder(row).join(" → "),rank_mapping:JSON.stringify(row.ranks),
  delta_pp: row.accuracy_delta_pp ?? null, gate_passed: row.baseline_gate_passed ? "Пройден" : "Не пройден",
  construction_seconds: row.compression_seconds ?? null,
  calibration_nmse: row.cohort === "common_linear" ? fullById.get(row.row_id)?.isolated_layer_errors?.calibration?.[Object.keys(row.ranks)[0]]?.relative_mse ?? null : null,
  model_hash: fullById.get(row.row_id)?.model_hash ?? null,
  artifact_sha256: fullById.get(row.row_id)?.timing?.artifact?.sha256 ?? null});
const all = analysis.rows.map(normalize);
const candidates = all.filter(row => row.method_id !== "baseline"), baselines = all.filter(row => row.method_id === "baseline");
for (const row of candidates) {
  const peers=[...candidates.filter(peer=>peer.seed===row.seed&&peer.cohort_id===row.cohort_id),
    ...baselines.filter(peer=>peer.seed===row.seed&&peer.architecture===row.architecture)];
  row.pareto_within_seed=!peers.some(peer=>peer.tensor_bytes<=row.tensor_bytes && peer.test_accuracy>=row.test_accuracy &&
    (peer.tensor_bytes<row.tensor_bytes||peer.test_accuracy>row.test_accuracy));
}
const paired = analysis.paired_summaries.map(row => ({
  architecture: row.architecture, cohort_id: cohortId(row), cohort: cohortName(row), method: names[row.method], method_id:row.method,
  level:row.level, ranks: candidates.find(c => c.architecture === row.architecture && c.level === row.level && c.method_id === row.method && c.cohort_id === cohortId(row))?.ranks,
  paired_seeds:row.succeeded_seeds.length, mean_delta_pp:row.accuracy_delta_pp.mean,
  min_delta_pp:row.accuracy_delta_pp.min, max_delta_pp:row.accuracy_delta_pp.max, sample_sd_pp:row.accuracy_delta_pp.sample_sd,
  tensor_bytes:row.tensor_bytes.length === 1 ? row.tensor_bytes[0] : null,
  tensor_saved_percent:row.tensor_saved_percent.mean, analytical_mac_fraction:row.analytical_mac_fraction.mean,
  latency_p50_mean_ms:row.latency_p50_ms.mean, latency_p50_min_ms:row.latency_p50_ms.min, latency_p50_max_ms:row.latency_p50_ms.max,
  latency_p95_mean_ms:row.latency_p95_ms.mean, compression_mean_seconds:row.compression_seconds.mean,
  candidate_total_mean_seconds:row.total_candidate_seconds.mean,
  cross_entropy_delta:row.cross_entropy_delta.mean, macro_f1_delta:row.macro_f1_delta.mean,
  observed_latency_ratio:row.observed_latency_ratio.mean,
}));
const layers = analysis.layers.map(row => {
  const raw=fullById.get(row.row_id), layer=transformLayers(raw?.transform).find(item=>item.path===row.operator);
  return {...row,cohort_id:cohortId(row),cohort:cohortName(row),method:names[row.method],path:row.operator,
    ranks:raw ? ranks(raw) : "", input_dim:raw?.budget?.targets?.[row.operator]?.matrix_shape?.[1] ?? layer?.shape?.slice(1).reduce((a,b)=>a*b,1) ?? null,
    observations:layer?.statistics?.[0]?.count ?? null, gram_rank:layer?.numerics?.[0]?.moment_rank ?? null,
    actual_rank:raw?.actual_ranks?.[row.operator]?.stored_rank ?? null,
    calibration_nmse:row.role === "calibration" ? row.relative_mse : null,
    propagated_input_nmse:raw?.propagated_layer_errors?.[row.role]?.[row.operator]?.input_error?.relative_mse ?? null,
    propagated_output_nmse:raw?.propagated_layer_errors?.[row.role]?.[row.operator]?.output_error?.relative_mse ?? null};
});
const timing = all.map(row => {
  const t = fullById.get(row.row_id)?.timing;
  return {row_id:row.row_id,cohort:row.cohort,method:row.method,seed:row.seed,ranks:row.ranks,
    batch:64,threads:1,warmup:20,samples:row.timing_samples,
    p50_ms:row.latency_p50_ms,p95_ms:row.latency_p95_ms,load_seconds:row.load_seconds,
    system_cpu_percent_before:t?.system_cpu_percent_before,system_cpu_percent_after:t?.system_cpu_percent_after,
    runtime:t?.profile?.runtime,runtime_version:t?.profile?.runtime_version};
});
const metrics = [
  {label:"Полный объём тензоров",definition:"Все уникальные физические хранилища параметров и буферов модели, включая неизменённые операторы и смещения.",componentIds:["svd-baselines","svd-costs",...['mlp-linear','cnn-linear','cnn-chain'].map(id=>`svd-quality-${id}`)],variable:"tensor_bytes"},
  {label:"Изменение точности, п.п.",definition:"100 × (точность на test кандидата − точность на test исходной модели того же seed).",variable:"delta_pp",formula:"100 * (test_accuracy - paired_baseline_test_accuracy)"},
  {label:"Локальная NMSE",definition:"MSE выхода изменённого слоя на зафиксированном входе исходной модели, делённая на средний квадрат исходного выхода. Для цепочки не суммируется.",variable:"calibration_nmse"},
  {label:"Аналитические MAC",definition:"Умножения с накоплением на один объект по формам операторов; без смещений, активаций и расходов памяти. Это не измеренное время.",variable:"analytical_macs"},
];
function source(file, detail, definitions=[]) {return {label:"Локальная обученная серия FedCore · 13.10.2026",classification:"observed",provider:"FedCore local experiment",
  files:[`docs/svd_compression/results/local-2026-10-13/${file}`],links:[{label:"Проверенные исходные данные",url:base+file}],
  metricDefinitions:definitions,evidenceFlow:[{title:"Измерение",detail:"Шесть исходных моделей, 90 заранее заданных кандидатов, три seed обучения. Все кандидаты сохранены; восстановительного обучения нет."},
    {title:"Проверка",detail:"Независимый scripts/analyze_svd_comparison.py повторно рассчитывает парные разности и сводки из сохранённых JSON. SHA256 analysis.json: "+crypto.createHash("sha256").update(fs.readFileSync(analysisFile)).digest("hex")},
    {title:"Преобразование для отчёта",detail}],
  caveats:["Три seed — описательная серия, SD не является доверительным интервалом.","Равный ранг совпадает с равным объёмом тензоров выбранных конфигураций.","Сырые 200 аппаратных повторов каждой сессии доступны в summary.json; HTML содержит их p50/p95, а не дубликаты массивов."]};}
const queries = {
  protocol:{rows:Object.entries(protocol.architectures).map(([architecture,value])=>({architecture,epochs:value.baseline_epochs,validation_gate:value.minimum_validation_accuracy,seeds:"41, 42, 43",split:"60 / 15 / 10 / 15",recovery_updates:0})),source:source("protocol.json","Фиксированные настройки серии, без численных исходов.")},
  candidates:{rows:candidates,payloadColumns:["rank_mapping"],source:source("analysis.json","rows: исключён baseline; ранги приведены в порядке операторов features.0 → features.2 → classifier.0 для цепочки; исходное ranks отображение сохранено в rank_mapping. calibration_nmse присоединена из isolated_layer_errors по row_id. Исходные logit, F1, entropy, стоимость и хеши сохранены.",metrics)},
  baselines:{rows:baselines,source:source("analysis.json","rows: method=baseline; одна строка на architecture × seed.",metrics.slice(0,1))},
  paired:{rows:paired,source:source("paired.csv","paired_summaries: значения mean/min/max/sample_sd присоединены к исходному вектору рангов. Единица анализа — парный seed обучения, n=3.",metrics.filter(m=>m.variable!=="calibration_nmse"))},
  layers:{rows:layers,source:source("layers.csv","Слой × role × кандидат. Число наблюдений и ранг Gram добавлены из диагностик преобразования; propagated errors добавлены из фактически изменённого графа. Null означает отсутствие применимой диагностики.",metrics.filter(m=>m.variable==="calibration_nmse"))},
  timing:{rows:timing,source:source("summary.json","96 свежих процессов: сохраняются p50/p95 и условия, без массивов сырых повторов. Использовано timing.metrics, не однократная проверка export.metrics.")},
  outcomes:{rows:[{planned_candidates:90,succeeded_candidates:analysis.counts.succeeded_candidates,failed_candidates:90-analysis.counts.succeeded_candidates,passed_baselines:baselines.filter(b=>b.baseline_gate_passed).length,total_baselines:6,timing_sessions:analysis.counts.timing_sessions}],source:source("analysis.json","counts + число истинных baseline_gate_passed. Все запланированные конфигурации учтены.")},
  audit:{rows:[{status:audit.status,cumulative_worker_wall_seconds:audit.resources.cumulative_worker_wall_seconds,
    maximum_sampled_tree_sum_rss_bytes:audit.resources.maximum_sampled_tree_sum_rss_bytes,minimum_available_ram_bytes:audit.resources.minimum_available_ram_bytes,
    v1_v2_strict_numerical_parity:audit.v1_v2_strict_numerical_parity,all_source_baselines_unchanged:audit.all_source_baselines_unchanged,
    all_checkpoint_export_parity_passed:audit.all_checkpoint_export_parity_passed,resource_violation_count:audit.resources.resource_violations.length,
    source_sha256:audit.git_lf_source_sha256,background_cpu_min:audit.background_cpu_percent_before_after.minimum,
    background_cpu_median:audit.background_cpu_percent_before_after.median,background_cpu_max:audit.background_cpu_percent_before_after.maximum}],
    source:source("final-audit.json","Численные итоги окончательного аудита; без системных идентификаторов процессов и локальных путей. RSS дерева — максимум наблюдавшихся снимков, не точный пик выделенной памяти.")},
};
const summary = "## Главные результаты\n\n- **Linear CNN, ранг 8:** AFM даёт среднее изменение точности на test **0,00 ± 0,37 п.п.**, обычный SVD — **−17,16 ± 9,63 п.п.** при одинаковых **47 592 байтах** всех тензоров модели. После ± указан разброс трёх парных seed, а не доверительный интервал.\n- **Linear CNN, ранг 16:** обычный SVD сохраняет наблюдавшуюся точность всех трёх исходных моделей при **82 408 байтах** вместо **274 920 байт**. Статистическая цель нужна не при любой степени сжатия.\n- **MLP:** точность почти не различает методы. **Conv + Linear:** агрессивные ранги резко ухудшают качество; Weighted-SVD при средних и больших рангах уступает SVD.\n\nВсе шесть исходных моделей прошли пороги качества. Завершены **90 из 90 кандидатов**, без восстановительного обучения. Эта малая серия не устанавливает универсальное превосходство метода.";
const cohortNotes = {
  "cnn-linear":"AFM при ранге 8 сохраняет среднее наблюдавшееся качество, но отдельные изменения имеют разные знаки: −0,37 / 0 / +0,37 п.п. Weighted-SVD в среднем теряет 0,74 п.п., ASVD — 4,20 п.п. При ранге 16 обычный SVD уже сохраняет точность всех трёх исходных моделей.\n\nДля этих float32-конфигураций с двумя факторами и bias **равный ранг означает равный полный объём тензоров**. Это один контроль, а не независимая вторая серия. Размер файлов сравнивается отдельно.",
  "mlp-linear":"Test содержит 86 объектов. Один правильный ответ меняет точность одного seed примерно на 1,16 п.п. Средний прирост Weighted-SVD +0,39 п.п. — один дополнительный правильный ответ в одном из трёх повторов, что не устанавливает преимущество метода. Остальные методы сохраняют наблюдавшуюся точность.\n\nРавный ранг здесь совпадает с равным полным объёмом тензоров; отдельного независимого сравнения бюджетов нет.",
  "cnn-chain":"Вектор рангов в графиках и таблицах следует порядку **`features.0 → features.2 → classifier.0`**. При (1, 4, 8) средняя потеря составляет 57,04 п.п. у SVD и 47,78 п.п. у Weighted-SVD. При (2, 8, 16) Weighted-SVD теряет 12,72 п.п. против 3,33 п.п. у SVD; при (3, 12, 32) — 1,60 против 0,49 п.п. Это прямой отрицательный результат для универсального преимущества взвешенной цели.\n\nВыбраны одинаковые операторы и ранги; объём тензоров совпадает. Калибровка следующего слоя собирается на текущем изменённом графе.",
};
const localErrorNotes="Локальная ошибка вычислена на **зафиксированных входах исходной модели**. Ошибка распространяющегося сигнала вычислена отдельно на изменённом графе; для цепочки показатели каждого слоя доступны в таблице ниже.\n\nПолные вторые моменты Linear вырождены по построению: **128 входных признаков / 57 объектов** у MLP и **1024 / 180** у CNN. `ridge=0`, `support_only` ограничивает цель наблюдаемым подпространством. Влияние ограниченной поддержки, последовательных изменений и нелинейностей эта серия причинно не разделяет.";
previous.title="При сильном сжатии CNN цель метода меняет качество";
previous.status="reviewed";
previous.buildStatus="updating";
previous.generatedAt=new Date().toISOString();
previous.report={asOf:"2026-10-13",summary,cohortNotes,localErrorNotes,
  outcomeNotes:"## Все конфигурации завершены; отрицательные результаты сохранены\n\n90 из 90 кандидатов успешно построены и проверены после повторной загрузки. Успешное выполнение не означает полезного сжатия: агрессивная цепочка Conv + Linear резко теряет качество, а взвешенная цель не выигрывает во всех режимах.",
  nextStep:"## Следующий этап — заранее закреплённая независимая проверка\n\nПроверить AFM при сильном сжатии Linear и обычный SVD при умеренном; для Weighted-SVD — отдельно менять размер calibration, регуляризацию и политику nullspace. Для Conv нужен более осторожный бюджет и самостоятельная проверка восстановления.\n\nКандидат для развёртывания выбирается по validation и допустимой потере качества. Ретроспективный максимум этой test-таблицы не является правилом выбора. Ещё не проверены крупные CNN/LLM, сдвиг распределения, другие dtype, batch=1 и GPU."};
previous.queries=queries;
fs.writeFileSync(outputFile,JSON.stringify(previous,null,2)+"\n");
console.log(JSON.stringify({output:path.basename(outputFile),candidates:candidates.length,baselines:baselines.length,layers:layers.length,paired:paired.length,bytes:fs.statSync(outputFile).size}));
