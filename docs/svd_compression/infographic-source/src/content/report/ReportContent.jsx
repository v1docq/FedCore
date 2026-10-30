import React, { useState } from "react";
import { DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import { ReportDisclosure } from "../shared/ReportDisclosure.jsx";
import { copy } from "./report-copy.js";

const COHORTS = [
  { id: "mlp-linear", label: "MLP · один Linear", scenario: "BreastCancerBigMLP" },
  { id: "cnn-linear", label: "CNN · один Linear", scenario: "DigitsCNN" },
  { id: "cnn-chain", label: "CNN · Conv + Linear", scenario: "DigitsCNN" },
];
const pct = value => Number.isFinite(value) ? `${(100 * value).toFixed(2)}%` : "—";
const fixed = (value, n = 2) => {
  if (!Number.isFinite(value)) return "—";
  const result=value.toFixed(n);
  return Number(result) === 0 ? (0).toFixed(n) : result;
};
const number = value => Number.isFinite(value) ? value.toLocaleString("ru-RU") : "—";
const col = (field, label, renderCell) => ({ field, label, ...(renderCell ? { renderCell } : {}) });
const columns = [col("cohort", "Сравнение"), col("method", "Метод"), col("seed", "Seed"), col("ranks", "Ранги"),
  col("test_accuracy", "Test", pct), col("delta_pp", "Δ, п.п.", value => fixed(value, 3)),
  col("tensor_bytes", "Тензоры, байт", number), col("artifact_bytes", "Файл, байт", number),
  col("latency_p50_ms", "p50, мс", value => fixed(value, 4)), col("latency_p95_ms", "p95, мс", value => fixed(value, 4)),
  col("construction_seconds", "Преобразование, с", value => fixed(value, 4)), col("total_candidate_seconds", "Весь путь, с", value => fixed(value, 4)),
  col("pareto_within_seed", "Парето, свой seed", value => value ? "Да" : "Нет")];
const pairedColumns = [col("cohort", "Сравнение"), col("method", "Метод"), col("ranks", "Ранги"),
  col("paired_seeds", "Пар seed"), col("mean_delta_pp", "Среднее Δ, п.п.", value => fixed(value, 3)),
  col("min_delta_pp", "Минимум, п.п.", value => fixed(value, 3)), col("max_delta_pp", "Максимум, п.п.", value => fixed(value, 3)),
  col("sample_sd_pp", "SD, п.п.", value => fixed(value, 3)), col("tensor_bytes", "Тензоры, байт", number)];
function Prose({ id, value, ...props }) { return <RichNarrative id={`svd:${id}`} value={value} label="Редактировать пояснение" {...props} />; }
function TableEvidence({ id, title, queryId, rows, columns: tableColumns, ...props }) {
  return <DataComponent id={id} title={title} queryId={queryId} kind="table" sourceRows={rows} displayRows={rows} {...props}>
    <div data-reviewed-rows><DataTable rows={rows} columns={tableColumns} caption={title} compactNumbers={false} /></div>
  </DataComponent>;
}
export function ReportContent() {
  const { snapshot, appTitle, canEdit, mode, setAppTitle } = useDataApp();
  const [cohortId, setCohortId] = useState("cnn-linear");
  const rows = id => snapshot.queries?.[id]?.rows ?? [];
  const candidates = rows("candidates"), baselines = rows("baselines"), paired = rows("paired");
  const selected = candidates.filter(row => row.cohort_id === cohortId);
  const cohort = COHORTS.find(item => item.id === cohortId);
  const cohortBaselines = baselines.filter(row => row.scenario === cohort.scenario);
  const cohortPairs = paired.filter(row => row.cohort_id === cohortId);
  const points = [...selected, ...cohortBaselines].map(row => ({ ...row, name: `${row.method} · ${row.ranks ? `r=${row.ranks} · ` : ""}seed ${row.seed}`,
    modelKiB: row.tensor_bytes / 1024, "Точность test (%)": 100 * row.test_accuracy }));
  const workRows = selected.filter(row => row.method_id === "svd" && row.seed === 41).map(row => ({...row,
    rank_label:`Ранги ${row.ranks}`, "Тензоры (%)":100*row.tensor_retained_fraction, "Аналитические MAC (%)":100*row.analytical_mac_fraction}));
  const hasResults = candidates.length > 0;
  return <article className="report-content svd-report" aria-label="Сравнение низкорангового сжатия FedCore">
    <header className="report-hero">
      <Prose id="eyebrow" value="FEDCORE · ЛОКАЛЬНОЕ СРАВНЕНИЕ · 13 ОКТЯБРЯ 2026" className="svd-eyebrow" />
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? event => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
      <Prose id="deck" value={copy.deck} className="report-deck" />
    </header>
    <ReportSection id="svd-summary" title={hasResults ? "Главные результаты" : "Статус данных"} queryId={hasResults ? "candidates" : "protocol"}
      {...(hasResults ? { queryIds: ["candidates", "baselines", "paired"], sourceRowsByQuery: { candidates, baselines, paired } } : { sourceRows: rows("protocol") })} showHeading={false}>
      <Prose id="summary" value={snapshot.report?.summary ?? copy.pending} className="report-summary-lead" />
    </ReportSection>
    {hasResults && <section className="svd-section">
      <Prose id="baseline-heading" value="## Сначала проверяем качество исходных моделей\n\nКаждый кандидат сравнивается с исходной моделью своего seed. Порог допуска применяется к validation до сжатия; test используется для итоговой оценки." />
      <TableEvidence id="svd-baselines" title="Обученные исходные модели" queryId="baselines" rows={baselines}
        columns={[col("scenario", "Модель"), col("seed", "Seed"), col("validation_accuracy", "Validation", pct), col("test_accuracy", "Test", pct),
          col("gate_passed", "Допуск"), col("tensor_bytes", "Тензоры, байт", number), col("artifact_bytes", "Файл, байт", number)]} />
    </section>}
    {hasResults && <section className="svd-section">
      <Prose id="tradeoff-heading" value={copy.tradeoff} />
      <div className="svd-cohort-tabs" role="group" aria-label="Выбрать сравнение">
        {COHORTS.map(item => <button key={item.id} type="button" aria-pressed={cohortId === item.id} onClick={() => setCohortId(item.id)}>{item.label}</button>)}
      </div>
      <EvidenceChart id={`svd-quality-${cohortId}`} title={`${cohort.label}: качество и полный объём весов`} queryId="candidates"
        queryIds={["candidates", "baselines"]} sourceRowsByQuery={{ candidates: selected, baselines: cohortBaselines }} sourceRows={selected} rows={points}
        spec={{ type: "scatter", x: "modelKiB", y: "Точность test (%)", series: "method", xLabel: "Тензоры всей модели, КиБ", yLabel: "Точность test, %", stackable: false, valueDecimals: 3, startAtZero:false }} height={370} />
      <ReportSection id={`svd-cohort-notes-${cohortId}`} title="Как читать сравнение" queryId="candidates" sourceRows={selected} showHeading={false}>
        <Prose id={`cohort-notes-${cohortId}`} value={snapshot.report?.cohortNotes?.[cohortId] ?? copy.rankBudget} />
      </ReportSection>
      <EvidenceChart id={`svd-delta-${cohortId}`} title="Парное изменение точности на test: среднее трёх seed" queryId="paired"
        sourceRows={cohortPairs} rows={cohortPairs.map(row=>({...row,name:`${row.method} · ${row.ranks}`}))}
        spec={{type:"horizontalBar",x:"name",y:"mean_delta_pp",xLabel:"Изменение, п.п.",stackable:false,valueDecimals:3}}
        height={Math.max(300,cohortPairs.length*36)} />
      <TableEvidence id={`svd-paired-${cohortId}`} title="Парное изменение качества по трём seed" queryId="paired" rows={cohortPairs} columns={pairedColumns} />
      <Prose id="spread" value={copy.spread} />
      <Prose id="compute-heading" value={copy.compute} />
      <EvidenceChart id={`svd-compute-${cohortId}`} title={`${cohort.label}: доля исходного хранения и аналитических операций`} queryId="candidates"
        sourceRows={selected.filter(row=>row.method_id==="svd"&&row.seed===41)} rows={workRows}
        spec={{type:"horizontalBar",x:"rank_label",y:"Тензоры (%)",fields:["Тензоры (%)","Аналитические MAC (%)"],stackable:false,valueDecimals:1}}
        height={310} />
      <Prose id="local-heading" value={copy.localError} />
      {cohortId !== "cnn-chain" && <EvidenceChart id={`svd-local-error-${cohortId}`} title={`${cohort.label}: локальная ошибка и изменение качества`} queryId="candidates"
        sourceRows={selected} rows={selected.filter(row => Number.isFinite(row.calibration_nmse)).map(row => ({ ...row, name: `${row.method} · r=${row.ranks} · seed ${row.seed}` }))}
        spec={{ type: "scatter", x: "calibration_nmse", y: "delta_pp", series: "method", xLabel: "NMSE выхода слоя на calibration", yLabel: "Изменение точности на test, п.п.", stackable: false, valueDecimals: 4 }} height={330} />}
      <ReportSection id="svd-local-interpretation" title="Что означает локальная ошибка" queryId="candidates" sourceRows={selected} showHeading={false}>
        <Prose id="local-interpretation" value={snapshot.report?.localErrorNotes ?? copy.localErrorLimit} />
      </ReportSection>
    </section>}
    {hasResults && <section className="svd-section">
      <Prose id="cost-heading" value={copy.costs} />
      <TableEvidence id="svd-costs" title="Наблюдавшиеся задержки при фоновой нагрузке и стоимость построения" queryId="candidates" rows={candidates} columns={columns} />
      <ReportSection id="svd-timing-notes" title="Условия измерения времени" queryId="timing" sourceRows={rows("timing")} showHeading={false}>
        <Prose id="timing" value={snapshot.report?.timingNotes ?? copy.timing} />
        <ReportDisclosure id="svd-timing-details" label="Показать условия и ограничения измерения">
          <Prose id="timing-details" value={copy.timingDetails} />
          <div data-reviewed-rows><DataTable rows={rows("timing")} caption="Условия аппаратных измерений" compactNumbers={false} /></div>
        </ReportDisclosure>
      </ReportSection>
    </section>}
    <ReportSection id="svd-methods" title="Протокол и границы сравнения" queryId="protocol" sourceRows={rows("protocol")} showHeading={false}>
      <Prose id="scope" value={copy.scope} />
      <ReportDisclosure id="svd-protocol-disclosure" label="Методы, разбиение данных и ранги">
        <Prose id="methods" value={copy.methods} />
        <Prose id="objectives" value={copy.objectives} />
        <div data-reviewed-rows><DataTable rows={rows("protocol")} caption="Зафиксированные параметры серии" compactNumbers={false} searchable={false}
          columns={[col("architecture","Архитектура"),col("epochs","Эпох"),col("validation_gate","Порог на validation"),
            col("seeds","Seed"),col("split","Доли train / validation / calibration / test"),col("recovery_updates","Шагов восстановления")]} /></div>
      </ReportDisclosure>
    </ReportSection>
    {rows("layers").length > 0 && <TableEvidence id="svd-statistics" title="Ранг калибровочных статистик и фактические формы" queryId="layers" rows={rows("layers")}
      columns={[col("cohort", "Сравнение"), col("method", "Метод"), col("seed", "Seed"), col("ranks", "Ранги"), col("path", "Слой"),
        col("role", "Роль данных"), col("input_dim", "Вход"), col("observations", "Наблюдений"), col("gram_rank", "Ранг Gram"), col("actual_rank", "Факт. ранг"),
        col("relative_mse", "Локальная NMSE", value => fixed(value, 6)), col("propagated_input_nmse", "Сдвиг входа", value => fixed(value, 6)), col("propagated_output_nmse", "Сдвиг выхода", value => fixed(value, 6))]} />}
    {rows("outcomes").length > 0 && <ReportSection id="svd-outcomes" title="Отказы и отрицательные результаты" queryId="outcomes"
      queryIds={["outcomes","audit"]} sourceRowsByQuery={{outcomes:rows("outcomes"),audit:rows("audit")}} showHeading={false}>
      <Prose id="outcomes" value={snapshot.report?.outcomeNotes ?? "## Отказы и отрицательные результаты\n\nВсе отклонённые и неудачные конфигурации остаются в таблице. Отсутствие результата не заменяется нулевым значением."} />
      <div data-reviewed-rows><DataTable rows={rows("outcomes")} caption="Исход каждой конфигурации" compactNumbers={false}
        columns={[col("planned_candidates","Запланировано"),col("succeeded_candidates","Завершено"),col("failed_candidates","Ошибок выполнения"),
          col("passed_baselines","Исходных моделей с допуском"),col("total_baselines","Всего исходных моделей"),col("timing_sessions","Аппаратных сессий")]} /></div>
      <ReportDisclosure id="svd-reliability" label="Повтор серии и проверка измерительного тракта">
        <Prose id="reliability" value="Первая серия завершила все конфигурации, но внешний монитор Windows видел процесс запуска виртуального окружения вместо дочернего Python. Её значения RSS не используются как память вычислительного процесса. После исправления серия полностью повторена: **все 96 хешей моделей, значения качества, ранги и локальные/распространяющиеся ошибки совпали**.\n\nСуммарно учтено **593,73 с** вычислительной работы обеих серий и аппаратных сессий. Максимальная наблюдавшаяся сумма RSS дерева — **755 200 000 байт** (около 720,2 МиБ); минимально доступная RAM — **12 190 498 816 байт**. Нарушений утверждённых ограничений не зарегистрировано. RSS измеряется снимками, может повторно учитывать общие страницы и не является точным пиком выделенной памяти." />
      </ReportDisclosure>
    </ReportSection>}
    {hasResults && <ReportSection id="svd-next-step" title="Что следует из сравнения" queryId="candidates" sourceRows={candidates} showHeading={false}>
      <Prose id="next-step" value={snapshot.report?.nextStep ?? copy.nextStep} />
    </ReportSection>}
    {hasResults && <ReportSection id="svd-raw-evidence" title="Проверить исходные числа" queryId="candidates" sourceRows={candidates} showHeading={false}>
      <Prose id="raw-heading" value="## Проверить исходные числа\n\nТаблица содержит все измеренные конфигурации; меню каждого графика открывает его данные и происхождение. Значения, единицы и формулы сохранены внутри этого HTML и доступны без подключения к сети." />
      <ReportDisclosure id="svd-raw-disclosure" label="Все измеренные кандидаты"><div data-reviewed-rows><DataTable rows={candidates}
        columns={Object.keys(candidates[0] ?? {}).map(field=>col(field,field,value=>typeof value === "number" ? String(value) : typeof value === "boolean" ? String(value) : value ?? "—"))}
        caption="Все исходные значения кандидатов без округления" compactNumbers={false} /></div></ReportDisclosure>
    </ReportSection>}
  </article>;
}
