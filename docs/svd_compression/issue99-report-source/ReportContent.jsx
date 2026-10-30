import React from "react";
import { DataComponent, DataTable, EvidenceChart, Filters, ReportSection,
  RichNarrative, useDataApp, useSectionFilters } from "../../data-app-public.jsx";
import { findings } from "./findings.js";
import "./report.css";

const columns = (fields) => fields.map(([field, label]) => ({ field, label,
  renderCell: (value) => typeof value === "number" ? value.toLocaleString("ru-RU", {
    maximumFractionDigits: /hv|hypervolume|CE|quality|delta/.test(field) ? 8 : 4,
    minimumFractionDigits: /hv|hypervolume|CE|quality|delta/.test(field) ? 6 : 0,
  }) : value == null ? "—" : String(value),
}));
const colors = ["#176b91", "#c47a19", "#5c49a8"];

function Prose({ section }) {
  const { reviewedRows, visible } = useDataApp();
  if (!visible(section.id)) return null;
  const ids = section.queries;
  return <ReportSection id={section.id} queryId={ids[0]} queryIds={ids}
    sourceRowsByQuery={Object.fromEntries(ids.map((q) => [q, reviewedRows(q)]))}
    title={section.title} showHeading={false}>
    <RichNarrative id={`${section.id}:body`} value={`## ${section.title}\n\n${section.body}`} />
  </ReportSection>;
}

function Table({ id, queryId, title, rows, fields, scoped }) {
  const { visible } = useDataApp();
  if (!visible(id)) return null;
  return <DataComponent id={id} queryId={queryId} title={title} kind="table"
    {...scoped} displayRows={rows} sourceRows={scoped?.sourceRows || rows}>
    <DataTable rows={rows} columns={columns(fields)} compactNumbers={false}
      searchable={rows.length > 12} label={title} />
  </DataComponent>;
}

function SearchDetail() {
  const { visible } = useDataApp();
  const scope = useSectionFilters([
    { id: "model", label: "Обученная модель", field: "model_seed", defaultValue: "41", queryIds: ["curves", "search_runs", "paired", "common_bytes"] },
    { id: "budget", label: "Бюджет", field: "budget", defaultValue: "time", queryIds: ["curves", "search_runs", "paired", "common_bytes"] },
    { id: "search", label: "Состояние поиска", field: "search_seed", defaultValue: "11", queryIds: ["curves", "search_runs", "paired", "common_bytes"] },
  ], {}, { id: "issue99-search-detail" });
  const curves = scope.componentProps("curves", ["model_seed", "budget", "search_seed", "mode", "elapsed_parent_seconds"]);
  const runs = scope.componentProps("search_runs", ["model_seed", "budget", "search_seed", "mode"]);
  const paired = scope.componentProps("paired", ["model_seed", "budget", "search_seed", "comparison"]);
  const common = scope.componentProps("common_bytes", ["model_seed", "budget", "search_seed", "tensor_bytes"]);
  return <section className="issue99-detail">
    <RichNarrative id="search-detail:heading" value="### Как строится архив одного сравнения\n\nВыберите модель, бюджет и состояние поиска. Время отсчитывается от старта дочернего процесса и включает загрузку. Поздние результаты сохранены в исходных данных, но не улучшают архив в пределах срока." />
    <Filters {...scope.filterProps} />
    {visible("search-curve") && <EvidenceChart id="search-curve" title="Нормированный гиперобъём по времени от старта процесса"
      queryId="curves" {...curves} rows={curves.displayRows} height={340}
      spec={{ type: "line", x: "elapsed_parent_seconds", y: "normalized_hypervolume", series: "mode",
        xLabel: "Секунды", yLabel: "HV / 2,1", stackable: false, colors, valueDecimals: 6 }} />}
    <Table id="search-detail-table" queryId="search_runs" title="Полные оценки, предложения и расходы"
      scoped={runs} rows={runs.displayRows} fields={[["mode", "Режим"], ["finite_within_deadline", "Оценок в срок"],
        ["created_proposals", "Создано предложений"], ["cache_hits", "Повторных результатов"],
        ["normalized_hypervolume", "HV / 2,1"], ["stop_reason", "Остановка"]]} />
    <Table id="paired-detail" queryId="paired" title="Парные различия гиперобъёма: второй режим минус первый"
      scoped={paired} rows={paired.displayRows} fields={[["comparison", "Пара"],
        ["delta_hypervolume_second_minus_first", "ΔHV"],
        ["delta_accepted_hypervolume_second_minus_first", "ΔHV с допуском качества"]]} />
    <Table id="common-bytes" queryId="common_bytes" title="R и T при точно одинаковом фактическом хранении"
      scoped={common} rows={common.displayRows} fields={[["tensor_bytes", "Байты"], ["CE_R", "Лучший CE режима R"],
        ["CE_T", "Лучший CE режима T"], ["delta_CE_T_minus_R", "CE(T) − CE(R)"]]} />
  </section>;
}

function OrderDetail() {
  const scope = useSectionFilters([{ id: "model", label: "Обученная модель", field: "model_seed",
    defaultValue: "41", queryIds: ["order"] }], {}, { id: "issue99-order-detail" });
  const scoped = scope.componentProps("order", ["model_seed", "pair"]);
  return <section className="issue99-detail">
    <Filters {...scope.filterProps} />
    <EvidenceChart id="order-delta" title="Изменение validation CE при перестановке операций"
      queryId="order" {...scoped} rows={scoped.displayRows} height={330}
      spec={{ type: "horizontalBar", x: "pair", y: "delta_CE", stackable: false, valueDecimals: 7,
        yLabel: "CE второго порядка − CE первого порядка", startAtZero: true }} />
    <Table id="order-table" queryId="order" title="Состав пар и одинаковое плотное хранение"
      scoped={scoped} rows={scoped.displayRows} fields={[["pair", "Условие"], ["first_word", "Первый порядок"],
        ["second_word", "Второй порядок"], ["delta_CE", "ΔCE"], ["equal_dense_bytes", "Одинаковые 49 416 байт"]]} />
  </section>;
}

export function ReportContent() {
  const { appTitle, setAppTitle, canEdit, mode, reviewedRows, visible } = useDataApp();
  const summaries = reviewedRows("search_summary");
  const early = reviewedRows("early_late");
  const cache = reviewedRows("cache");
  const section = (name) => <Prose section={findings[name]} />;
  return <article className="report-content issue99-report" aria-label="Отчёт первого локального пакета FedCore #99">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
      <RichNarrative id="issue99:deck" className="report-deck" value="PR #140 · первый локальный пакет issue #99 · 20 октября 2026" />
    </header>
    {section("summary")}
    {section("search")}
    {["attempts", "time"].map((budget) => {
      const rows = summaries.filter((row) => row.budget === budget);
      const id = `summary-${budget}`;
      return visible(id) && <EvidenceChart key={id} id={id} queryId="search_summary"
        title={budget === "attempts" ? "Медиана трёх поисковых повторов при 12 полных оценках" : "Медиана трёх поисковых повторов при лимите 30 секунд"}
        rows={rows} sourceRows={rows} height={310}
        spec={{ type: "bar", x: "model_seed", y: "median_hv", series: "mode", stackable: false,
          colors, yLabel: "HV / 2,1", valueDecimals: 6, startAtZero: true }} />;
    })}
    <Table id="summary-table" queryId="search_summary" title="Медианы и диапазоны по трём поисковым состояниям"
      rows={summaries} fields={[["model_seed", "Модель"], ["budget", "Бюджет"], ["mode", "Режим"],
        ["median_hv", "Медиана HV / 2,1"], ["min_hv", "Минимум"], ["max_hv", "Максимум"],
        ["median_evaluations", "Медиана оценок"], ["median_parent_seconds", "Медиана полного времени, с"]]} />
    <SearchDetail />
    {section("early")}
    <EvidenceChart id="early-late-time" title="Время одинакового потока из 163 запросов, без загрузки процесса"
      queryId="early_late" rows={early} sourceRows={early} height={300}
      spec={{ type: "bar", x: "stream_seed", y: "A_stream_seconds", fields: ["A_stream_seconds", "B_stream_seconds"],
        legend: { labels: { A_stream_seconds: "Ранняя проверка", B_stream_seconds: "Поздняя проверка" } },
        yLabel: "Секунды", stackable: false, startAtZero: true, valueDecimals: 4 }} />
    <Table id="early-late-table" queryId="early_late" title="Все шесть пар раннего и позднего отказа"
      rows={early} fields={[["stream_seed", "Поток"], ["valid", "Результаты совпали"], ["A_stream_seconds", "Рано, с"],
        ["B_stream_seconds", "Поздно, с"], ["late_over_early", "Поздно / рано"],
        ["A_decompositions_started", "Разложений рано"], ["B_decompositions_started", "Разложений поздно"]]} />
    {section("cache")}
    <EvidenceChart id="cache-time" title="Время 48 запросов при одинаковом порядке и результатах"
      queryId="cache" rows={cache} sourceRows={cache} height={340}
      spec={{ type: "bar", x: "pair_label", y: "A_stream_seconds", fields: ["A_stream_seconds", "B_stream_seconds"],
        legend: { labels: { A_stream_seconds: "Кэш включён", B_stream_seconds: "Кэш выключен" } },
        yLabel: "Секунды", stackable: false, startAtZero: true, valueDecimals: 4 }} />
    <Table id="cache-table" queryId="cache" title="Полные вычисления и подтверждённые попадания в кэш"
      rows={cache} fields={[["pair_label", "Модель / поток"], ["valid", "Результаты совпали"],
        ["A_full_evaluations", "Оценок с кэшем"], ["B_full_evaluations", "Без кэша"],
        ["A_cache_hits", "Попаданий"], ["enabled_over_disabled", "Время с кэшем / без"]]} />
    {section("order")}
    <OrderDetail />
    {section("quality")}
    <Table id="final-tests" queryId="test_selected" title="Выбранные на validation модели: исторический test"
      rows={reviewedRows("test_selected")} fields={[["model_seed", "Модель"], ["job", "Источник выбора"],
        ["byte_cap", "Предел байтов"], ["tensor_bytes", "Фактические байты"],
        ["test_CE", "Test CE"], ["test_accuracy", "Test accuracy"]]} />
    <Table id="baseline-tests" queryId="baseline_test" title="Исторический test исходных плотных MLP"
      rows={reviewedRows("baseline_test")} fields={[["model_seed", "Модель"], ["cross_entropy", "CE"],
        ["accuracy", "Accuracy"], ["macro_f1", "Macro-F1"]]} />
    {section("resources")}
    <Table id="execution-lineage" queryId="resources" title="Все исполнения и накопленный бюджет"
      rows={reviewedRows("resources")} fields={[["source_build", "Коммит"], ["status", "Статус"],
        ["mlp_attempts", "Накопленные попытки"], ["cumulative_elapsed_seconds", "Накопленное время, с"]]} />
    {section("methods")}
    {section("next")}
  </article>;
}
