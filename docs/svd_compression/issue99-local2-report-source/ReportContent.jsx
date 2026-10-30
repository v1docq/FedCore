import React from "react";
import {
  DataComponent, DataTable, EvidenceChart, Filters, ReportSection,
  RichNarrative, SectionHeader, useDataApp, useSectionFilters,
} from "../../data-app-public.jsx";
import { findings } from "./findings.js";
import "./report.css";

const modeColors = ["#176b91", "#c47a19", "#5c49a8"];
const modeLabels = {
  R: "R · готовые планы", T: "T · случайный обход", E: "E · эволюционный поиск",
};
const modeOrder = { R: 0, T: 1, E: 2 };
const orderModes = (rows) => [...rows].sort((a, b) =>
  (modeOrder[a.mode] ?? 3) - (modeOrder[b.mode] ?? 3));
const oneContext = (rows, fields) => rows.length === 0 ||
  new Set(rows.map((row) => JSON.stringify(fields.map((field) => row[field])))).size === 1;

function formatCell(value, field) {
  if (value == null || (typeof value === "number" && !Number.isFinite(value))) return "—";
  if (typeof value === "boolean") return value ? "Да" : "Нет";
  if (typeof value !== "number") return String(value);
  if (/accuracy|storage_fraction/.test(field)) {
    return `${(value * 100).toLocaleString("ru-RU", { maximumFractionDigits: 4 })}%`;
  }
  const digits = /hv|hypervolume|CE|cross_entropy|quality|delta|oracle_gap/.test(field) ? 8
    : /ratio|seconds|macro_F1/.test(field) ? 6
      : /minutes|MiB|KiB|actual|limit/.test(field) ? 3 : 0;
  return value.toLocaleString("ru-RU", { maximumFractionDigits: digits });
}

function tableColumns(fields) {
  return fields.map(([field, label]) => ({ field, label,
    ...(/^delta/.test(field) ? { deltaTone: "neutral" } : {}),
    renderCell: (value) => formatCell(value, field),
  }));
}

function Prose({ section }) {
  const { reviewedRows, visible } = useDataApp();
  if (!section || !visible(section.id)) return null;
  const ids = section.queries;
  return <ReportSection id={section.id} queryId={ids[0]} queryIds={ids}
    title={section.title} showHeading={false}
    sourceRowsByQuery={Object.fromEntries(ids.map((queryId) => [queryId, reviewedRows(queryId)]))}>
    <RichNarrative id={`${section.id}:body`} value={`## ${section.title}\n\n${section.body}`} />
  </ReportSection>;
}

function EvidenceTable({ id, queryId, title, rows, fields, scoped, sourceRows, searchable = false }) {
  const { visible } = useDataApp();
  if (!visible(id)) return null;
  return <DataComponent id={id} queryId={queryId} title={title} kind="table"
    {...scoped} displayRows={rows} sourceRows={sourceRows ?? scoped?.sourceRows ?? rows}>
    <div className="local2-table-scroll" tabIndex={0} aria-label={`Прокручиваемая таблица: ${title}`}
      style={{ "--local2-table-min-width": `${Math.max(640, fields.length * 132)}px` }}>
      <DataTable rows={rows} columns={tableColumns(fields)} compactNumbers={false}
        searchable={searchable} label={title} />
    </div>
  </DataComponent>;
}

function ComparisonChart({ id, queryId, title, scoped, contextFields, spec, prompt, height = 340 }) {
  const { visible } = useDataApp();
  if (!visible(id)) return null;
  const single = oneContext(scoped.displayRows, contextFields);
  const rows = single ? orderModes(scoped.displayRows) : [];
  return <EvidenceChart id={id} queryId={queryId} title={title} variant="card"
    {...scoped} displayRows={rows} rows={rows} height={height} spec={spec}
    renderPlot={(plot) => single ? plot : null}>
    {!single && <RichNarrative id={`${id}:scope-help`} className="local2-scope-help" value={prompt} />}
  </EvidenceChart>;
}

function SearchSummary() {
  const { reviewedRows, visible } = useDataApp();
  const rows = reviewedRows("search_summary", ["model_seed", "budget", "mode"]);
  return <div className="local2-group">
    <div className="local2-budget-charts">
      {["attempts", "time"].map((budget) => {
        const id = `summary-${budget}`;
        const sourceRows = rows.filter((row) => row.budget === budget);
        return visible(id) && <EvidenceChart key={id} id={id} queryId="search_summary"
          title={budget === "attempts" ? "Медиана HV при бюджете до 24 оценок" : "Медиана HV при бюджете 30 секунд"}
          variant="card" rows={orderModes(sourceRows)} sourceRows={sourceRows} height={320}
          spec={{ type: "bar", x: "model_seed", y: "median_hv", series: "mode",
            colors: modeColors, legend: { labels: modeLabels }, stackable: false,
            xLabel: "Разбиение", yLabel: "Нормированный HV", valueDecimals: 8, startAtZero: true }} />;
      })}
    </div>
    <EvidenceTable id="summary-table" queryId="search_summary"
      title="Полные повторы, медианы и диапазоны трёх состояний поиска" rows={rows}
      fields={[["model_seed", "Разбиение"], ["budget_label", "Бюджет"], ["mode", "Режим"],
        ["complete_repeats", "Полных повторов из 3"], ["median_hv", "Медиана HV"],
        ["min_hv", "Минимум HV"], ["max_hv", "Максимум HV"],
        ["median_accepted_hv", "Медиана HV с допуском качества"], ["median_attempts", "Медиана оценок в срок"]]} />
  </div>;
}

function SearchDetail() {
  const queryIds = ["curves", "search_runs", "paired", "common_bytes", "plateau"];
  const scope = useSectionFilters([
    { id: "model", label: "Разбиение", field: "model_seed", defaultValue: "19901", queryIds },
    { id: "budget", label: "Бюджет", field: "budget", defaultValue: "time", queryIds },
    { id: "search", label: "Состояние поиска", field: "search_seed", defaultValue: "11", queryIds },
  ], {}, { id: "local2-search-detail" });
  const curves = scope.componentProps("curves", ["model_seed", "budget", "search_seed", "mode", "elapsed_parent_seconds"]);
  const attemptCurves = scope.componentProps("curves", ["model_seed", "budget", "search_seed", "mode", "attempt"]);
  const runs = scope.componentProps("search_runs", ["model_seed", "budget", "search_seed", "mode"]);
  const paired = scope.componentProps("paired", ["model_seed", "budget", "search_seed", "comparison"]);
  const common = scope.componentProps("common_bytes", ["model_seed", "budget", "search_seed", "tensor_bytes"]);
  const plateau = scope.componentProps("plateau", ["model_seed", "budget", "search_seed", "mode"]);
  return <section className="local2-group" aria-label="Подробности поискового сравнения">
    <SectionHeader id="local2-search-detail-heading" title="Один повтор поиска"
      filters={<Filters {...scope.filterProps} />} />
    <ComparisonChart id="search-curve" queryId="curves" scoped={curves}
      title="Накопленный HV по времени от старта дочернего процесса"
      contextFields={["model_seed", "budget", "search_seed"]}
      prompt="Выберите по одному разбиению, бюджету и состоянию поиска, чтобы сравнить траектории R, T и E. Таблицы ниже сохраняют весь выбранный набор строк."
      spec={{ type: "line", x: "elapsed_parent_seconds", y: "normalized_hypervolume", series: "mode",
        colors: modeColors, legend: { labels: modeLabels }, stackable: false,
        xLabel: "Секунды от старта процесса", yLabel: "Нормированный HV", valueDecimals: 8 }} />
    <ComparisonChart id="search-attempt-curve" queryId="curves" scoped={attemptCurves}
      title="Накопленный HV по числу оценок в срок"
      contextFields={["model_seed", "budget", "search_seed"]}
      prompt="Выберите по одному разбиению, бюджету и состоянию поиска, чтобы сравнить траектории R, T и E по числу оценок."
      spec={{ type: "line", x: "attempt", y: "normalized_hypervolume", series: "mode",
        colors: modeColors, legend: { labels: modeLabels }, stackable: false,
        xLabel: "Завершённые оценки в срок", yLabel: "Нормированный HV", valueDecimals: 8 }} />
    <EvidenceTable id="search-detail-table" queryId="search_runs" scoped={runs} rows={runs.displayRows}
      title="Попытки, завершённые оценки и полнота исполнения" searchable
      fields={[["model_seed", "Разбиение"], ["budget_label", "Бюджет"], ["search_seed", "Состояние поиска"],
        ["mode", "Режим"], ["on_time", "Оценок в срок"], ["attempts", "Начато попыток"],
        ["completed_attempts", "Завершено попыток"], ["incomplete_attempts", "Не завершено"], ["late", "Поздних оценок"],
        ["main_complete", "Основное исполнение полное"], ["normalized_hypervolume", "HV"],
        ["accepted_normalized_hypervolume", "HV с допуском качества"], ["oracle_gap", "Отставание от полного эталона"],
        ["accepted_oracle_gap", "Отставание от эталона с допуском"],
        ["created_proposals", "Создано предложений"], ["cache_hits", "Попаданий в кэш"],
        ["parent_seconds", "Полное время, с"], ["status", "Статус"], ["stop_reason", "Причина остановки"]]} />
    <EvidenceTable id="paired-detail" queryId="paired" scoped={paired} rows={paired.displayRows}
      title="Парные различия: второй режим минус первый"
      fields={[["model_seed", "Разбиение"], ["budget_label", "Бюджет"], ["search_seed", "Состояние поиска"],
        ["comparison", "Сравнение"], ["complete", "Пара полная"], ["delta_hv", "ΔHV"],
        ["delta_accepted_hv", "ΔHV с допуском качества"]]} />
    <EvidenceTable id="common-bytes" queryId="common_bytes" scoped={common} rows={common.displayRows}
      title="R и T при точно одинаковом хранении тензоров"
      fields={[["model_seed", "Разбиение"], ["budget_label", "Бюджет"], ["search_seed", "Состояние поиска"],
        ["tensor_bytes", "Байты тензоров"], ["main_complete", "Оба исполнения полные"],
        ["R_attempts", "Оценок R в срок"], ["T_attempts", "Оценок T в срок"],
        ["CE_R", "Лучший CE режима R"], ["CE_T", "Лучший CE режима T"], ["delta_CE_T_minus_R", "CE(T) − CE(R)"]]} />
    <EvidenceTable id="plateau-table" queryId="plateau" scoped={plateau} rows={plateau.displayRows}
      title="Ретроспективное правило плато: без реальной остановки вычислений"
      fields={[["mode", "Режим"], ["status", "Исход диагностики"], ["stop_after_finite", "Условная остановка после оценки"],
        ["later_finite_evaluations", "Оценок после неё"], ["missed_hypervolume", "Пропущенный ненормированный HV"],
        ["actual_time_saved", "Измеренная экономия времени"]]} />
  </section>;
}

function MatrixRatios() {
  const { visible } = useDataApp();
  const scope = useSectionFilters([
    { id: "size", label: "Размер матрицы", field: "n", defaultValue: "1024", queryIds: ["matrix_summary"] },
  ], {}, { id: "local2-matrix-ratios" });
  const scoped = scope.componentProps("matrix_summary", ["n", "invalid_label", "arm_label"]);
  const single = oneContext(scoped.displayRows, ["n"]);
  return <section className="local2-group" aria-label="Отношения времени по доле отказов">
    <SectionHeader id="local2-matrix-ratios-heading" title="Четыре доли недопустимых запросов"
      filters={<Filters {...scope.filterProps} />} />
    {visible("matrix-ratios") && <EvidenceChart id="matrix-ratios" queryId="matrix_summary"
      title="Медиана отношения времени A/B и C/B" variant="card" {...scoped}
      displayRows={single ? scoped.displayRows : []} rows={single ? scoped.displayRows : []}
      renderPlot={(plot) => single ? plot : null} height={340}
      spec={{ type: "bar", x: "invalid_label", y: "median_ratio", series: "arm_label",
        colors: [modeColors[0], modeColors[2]], stackable: false, startAtZero: true,
        xLabel: "Доля недопустимых запросов", yLabel: "Время режима / время B", valueDecimals: 6 }}>
      {!single && <RichNarrative id="matrix-ratios:scope-help" className="local2-scope-help"
        value="Выберите один размер матрицы, чтобы сравнить отношения времени при четырёх долях недопустимых запросов." />}
    </EvidenceChart>}
  </section>;
}

function MatrixDetail() {
  const queryIds = ["matrix_summary", "matrix_streams", "matrix_pairs", "matrix_controls"];
  const scope = useSectionFilters([
    { id: "size", label: "Размер матрицы", field: "n", defaultValue: "1024", queryIds },
    { id: "invalid", label: "Доля недопустимых запросов", field: "invalid_label", defaultValue: "50%", queryIds },
  ], {}, { id: "local2-matrix-detail" });
  const summary = scope.componentProps("matrix_summary", ["n", "invalid_label", "arm_label"]);
  const streams = scope.componentProps("matrix_streams", ["n", "invalid_label", "group", "stream_id", "arm"]);
  const pairs = scope.componentProps("matrix_pairs", ["n", "invalid_label", "arm_label", "block"]);
  const controls = scope.componentProps("matrix_controls", ["n", "invalid_label", "pair"]);
  return <section className="local2-group" aria-label="Проверки и исходные матричные потоки">
    <SectionHeader id="local2-matrix-detail-heading" title="Контроли выбранного матричного условия"
      filters={<Filters {...scope.filterProps} />} />
    <EvidenceTable id="matrix-summary-table" queryId="matrix_summary" scoped={summary} rows={summary.displayRows}
      title="Полнота блоков и все условия инженерного выигрыша"
      fields={[["size_label", "Размер"], ["invalid_label", "Недопустимых"], ["arm_label", "Сравнение с B"],
        ["pairs", "Полных блоков из 6"], ["median_ratio", "Медиана отношения"],
        ["min_ratio", "Минимум отношения"], ["max_ratio", "Максимум отношения"],
        ["median_absolute_seconds", "Медиана разности времени, с"], ["bb_pass", "B/B пройден"],
        ["order_ratio", "Отношение до/после B"], ["order_pass", "Порядок пройден"],
        ["median_b_seconds", "Медиана времени B, с"], ["duration_pass", "Длительность B пройдена"],
        ["resource_eligible", "Ресурсный надзор подтверждён"],
        ["eligible", "Все контроли пройдены"], ["observed_win", "Критерий выигрыша выполнен"]]} />
    <EvidenceTable id="matrix-pairs-table" queryId="matrix_pairs" scoped={pairs} rows={pairs.displayRows}
      title="Парные отношения A/B и C/B в полных блоках ABC"
      fields={[["size_label", "Размер"], ["invalid_label", "Недопустимых"], ["arm_label", "Сравнение с B"],
        ["block", "Блок"], ["a_seconds", "Время A или C, с"], ["b_seconds", "Время B, с"],
        ["ratio", "Отношение времени к B"], ["difference_seconds", "Разность времени с B, с"],
        ["before_b", "Режим выполнен до B"]]} />
    <EvidenceTable id="matrix-controls-table" queryId="matrix_controls" scoped={controls} rows={controls.displayRows}
      title="Четыре контрольные пары B/B"
      fields={[["size_label", "Размер"], ["invalid_label", "Недопустимых"], ["pair", "Пара"],
        ["B1_seconds", "Время B1, с"], ["B2_seconds", "Время B2, с"], ["ratio", "Время B1 / время B2"]]} />
    <EvidenceTable id="matrix-stream-table" queryId="matrix_streams" scoped={streams} rows={streams.displayRows}
      title="Исходные потоки по группам timed, control и warmup" searchable
      fields={[["size_label", "Размер"], ["invalid_label", "Недопустимых"], ["stream_id", "Поток"],
        ["group", "Группа"], ["block", "Блок"], ["pair", "Пара B/B"], ["label", "Позиция B/B"],
        ["tape_seed", "Лента запросов"], ["arm_label", "Режим"], ["primary_seconds", "Сумма времени вызовов, с"],
        ["stream_seconds", "Полное время потока, с"], ["request_count", "Запросов"], ["observed_svd_starts", "Начато SVD"],
        ["resource_status", "Статус ресурсного надзора"], ["resource_eligible", "Ресурсный надзор подтверждён"]]} />
  </section>;
}

function FrontierDetail() {
  const queryIds = ["frontiers"];
  const scope = useSectionFilters([
    { id: "model", label: "Разбиение", field: "model_seed", defaultValue: "19901", queryIds },
    { id: "budget", label: "Бюджет", field: "budget", defaultValue: "time", queryIds },
    { id: "search", label: "Состояние поиска", field: "search_seed", defaultValue: "11", queryIds },
  ], {}, { id: "local2-frontier-detail" });
  const scoped = scope.componentProps("frontiers", ["model_seed", "budget", "search_seed", "mode", "solution_id", "KiB", "CE"]);
  return <section className="local2-group" aria-label="Наблюдавшиеся границы Парето">
    <SectionHeader id="local2-frontier-heading" title="Наблюдавшиеся решения на границе Парето"
      filters={<Filters {...scope.filterProps} />} />
    <ComparisonChart id="frontier-chart" queryId="frontiers" title="Хранение тензоров и CE на проверочной выборке"
      scoped={scoped} contextFields={["model_seed", "budget", "search_seed"]} height={360}
      prompt="Выберите по одному разбиению, бюджету и состоянию поиска, чтобы показать сопоставимые границы Парето."
      spec={{ type: "scatter", x: "KiB", y: "CE", series: "mode", colors: modeColors,
        legend: { labels: modeLabels }, stackable: false,
        xLabel: "Тензоры модели, KiB", yLabel: "CE на проверочной выборке", valueDecimals: 8 }} />
  </section>;
}

function ResourceUsage() {
  const { reviewedRows } = useDataApp();
  const sourceRows = reviewedRows("resources");
  const rows = sourceRows.flatMap((row) => [
    { metric: "Полное время, мин", actual: row.elapsed_minutes, limit: row.wall_limit_minutes },
    { metric: "Попытки MLP", actual: row.mlp_ledger_count, limit: row.mlp_limit },
    { metric: "Матричные запросы", actual: row.matrix_ledger_count, limit: row.matrix_limit },
    { metric: "Начатые шаги обучения", actual: row.training_updates_started, limit: row.training_update_limit },
    { metric: "Завершённые шаги обучения", actual: row.training_updates_completed, limit: row.training_update_limit },
    { metric: "Максимальный наблюдавшийся RSS дерева процесса, MiB", actual: row.max_tree_rss_MiB,
      limit: row.tree_rss_limit_GiB == null ? null : row.tree_rss_limit_GiB * 1024 },
    { metric: "Начатые вызовы SVD", actual: row.observed_svd_starts, limit: null },
    { metric: "Завершённые вызовы SVD", actual: row.observed_svd_completions, limit: null },
  ]);
  return <EvidenceTable id="resource-table" queryId="resources" title="Накопленные расходы и утверждённые пределы"
    rows={rows} sourceRows={sourceRows} fields={[["metric", "Счётчик"], ["actual", "Фактический расход"], ["limit", "Предел"]]} />;
}

export function ReportContent() {
  const { appTitle, setAppTitle, canEdit, mode, reviewedRows } = useDataApp();
  const section = (name) => <Prose section={findings[name]} />;
  const editing = canEdit && mode === "edit";
  return <article className="report-content local2-report" aria-label="Отчёт второго локального CPU-пакета FedCore #99">
    <header className="report-hero local2-hero">
      <h1 data-data-app-title contentEditable={editing} suppressContentEditableWarning
        aria-label={editing ? "Изменить заголовок отчёта" : undefined}
        onBlur={editing ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={editing ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="local2:deck" className="report-deck"
        value="FedCore #99 · второй локальный CPU-пакет · PR #140 · 21 октября 2026" />
    </header>
    {section("summary")}
    {section("data")}
    <EvidenceTable id="baseline-table" queryId="baselines" title="Исходные плотные MLP на трёх новых разбиениях"
      rows={reviewedRows("baselines", ["model_seed"])}
      fields={[["model_seed", "Разбиение"], ["validation_CE", "CE на проверочной выборке"],
        ["validation_accuracy", "Доля верных ответов, проверка"], ["test_CE", "CE на тестовой выборке"],
        ["test_accuracy", "Доля верных ответов, тест"], ["training_steps", "Шагов обучения"]]} />
    {section("search")}
    <SearchSummary />
    <SearchDetail />
    <EvidenceTable id="oracle-table" queryId="oracles" title="Полный каталог из 440 решений и его полнота"
      rows={reviewedRows("oracles", ["model_seed"])}
      fields={[["model_seed", "Разбиение"], ["complete", "Каталог полный"], ["attempts", "Начато попыток"],
        ["completed_attempts", "Завершено попыток"], ["incomplete_attempts", "Не завершено"],
        ["on_time", "Оценок в срок"], ["finite", "Оценок с конечной метрикой"], ["accepted", "В допуске качества"],
        ["normalized_hypervolume", "HV полного эталона"], ["accepted_normalized_hypervolume", "HV эталона с допуском"],
        ["partial_archive_hypervolume", "HV наблюдавшегося архива"]]} />
    {section("matrix")}
    <MatrixRatios />
    <MatrixDetail />
    {section("quality")}
    <FrontierDetail />
    <EvidenceTable id="test-table" queryId="test_selected" title="Модели, выбранные на проверочной выборке: результаты на тестовой"
      rows={reviewedRows("test_selected", ["model_seed", "job", "byte_cap", "solution_id"])} searchable
      fields={[["model_seed", "Разбиение"], ["job", "Источник выбора"], ["byte_cap", "Предел, байты"],
        ["solution_id", "Решение"], ["tensor_bytes", "Тензоры, байты"], ["storage_fraction", "Доля хранения M0"],
        ["validation_cross_entropy", "CE на проверочной выборке"], ["test_CE", "CE на тестовой выборке"],
        ["test_accuracy", "Доля верных ответов, тест"], ["test_macro_F1", "Macro-F1 на тестовой выборке"]]} />
    {section("resources")}
    <ResourceUsage />
    <EvidenceTable id="job-table" queryId="jobs" title="Дочерние процессы и причины остановки"
      rows={reviewedRows("jobs", ["job"])} searchable
      fields={[["job", "Процесс"], ["kind", "Задача"], ["status", "Статус"], ["elapsed_seconds", "Полное время, с"],
        ["peak_rss_bytes", "Пиковый RSS дерева, байты"], ["stop", "Причина остановки"], ["error", "Ошибка"]]} />
    {section("methods")}
    {section("next")}
  </article>;
}
