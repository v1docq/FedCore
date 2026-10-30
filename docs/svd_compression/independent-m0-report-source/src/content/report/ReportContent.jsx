import React from "react";
import { DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const progressSpec = { type: "line", x: "minutes", y: "updates", startAtZero: true,
  xLabel: "Минуты от начала серии", yLabel: "Подтверждённые обновления SGD", valueDecimals: 0 };
const gpuSpec = { type: "line", x: "minutes", y: "gpu_rows", startAtZero: true,
  xLabel: "Минуты от начала серии", yLabel: "Сырые строки NVML", valueDecimals: 0 };
const columns = [
  { field: "stage", label: "Стадия" }, { field: "done", label: "Выполнено" },
  { field: "planned", label: "Предел плана" }, { field: "unit", label: "Единица" },
  { field: "status", label: "Состояние" },
];

export function ReportContent() {
  const { reviewedRows, appTitle, setAppTitle, canEdit, mode, visible } = useDataApp();
  const summary = reviewedRows("summary");
  const progress = reviewedRows("progress");
  const telemetry = reviewedRows("telemetry");
  const stages = reviewedRows("stages");
  return <article className="report-content" aria-label="Отчёт о прерванной серии A100">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent?.trim() || appTitle) : undefined}>{appTitle}</h1>
      <RichNarrative id="report:byline" value="PR #140 · A100 40 ГБ · 30 октября 2026 · Единственная попытка, без повторов" className="report-deck" />
    </header>

    {visible("opening") && <ReportSection id="opening" title="Основной результат ещё не получен" queryId="summary"
      queryIds={["summary", "telemetry", "progress"]} sourceRowsByQuery={{summary, telemetry, progress}} showHeading={false}>
      <RichNarrative id="opening:body" value={`## Основной результат ещё не получен

**Сравнить Random и Evo пока нельзя.** Серия прервалась во время первой исходной ResNet18: завершены 11 из 80 эпох, подтверждены 3521 обновление SGD. Ни один из 18 поисков не начался. Точность, разность CE и ускорение не измерены — это отсутствующие результаты, а не нулевой эффект.

Через **130,53 с** NVML показал второй GPU-процесс. Защита ошибочно отбрасывала PID хоста, недоступные внутри контейнера. Основной агент остановил только собственную серию через **237,49 с**; живых процессов серии после очистки нет. Исключительное использование GPU в этой попытке не подтверждено.

Материалы сохранены через **429,33 с от старта**, внутри восьмичасового предела. Новый эксперимент не запускался. Протокол и исполнитель реализованы, но экспериментальная приёмка остаётся незавершённой.`} />
    </ReportSection>}

    {visible("stages") && <ReportSection id="stages" title="Что удалось выполнить" queryId="stages" sourceRows={stages}>
      <DataTable rows={stages} columns={columns} searchable={false} caption="Фактические счётчики и заранее утверждённые пределы" />
      <RichNarrative id="stages:note" value="Пределы не являются новыми результатами или разрешением на дополнительные запуски. Для основного сравнения нужны все девять пар на трёх независимых M0; готовых M0 в этой попытке нет." />
    </ReportSection>}

    <section className="report-section">
      {visible("incident") && <ReportSection id="incident" title="Почему наблюдатель не остановил серию" queryId="telemetry" sourceRows={telemetry} showHeading={false}>
        <RichNarrative id="incident:body" value={`## Две сырые строки GPU превратились в «пустой GPU»

NVML возвращал PID хоста, а \`psutil\` работал в пространстве PID контейнера. В прежней функции \`classify_gpu_rows\` исключение \`NoSuchProcess\` вело к \`continue\`. Сначала отбрасывался наш процесс, затем и второй. По журналу **173 наблюдения с двумя строками** не сформировали причину остановки. Задержка между первым таким наблюдением и ручной остановкой составила **106,95 с**.

График показывает число исходных строк NVML без ошибочной фильтрации. Две строки — прямое нарушение принятого ограничения; сами номера PID не используются как цели сигналов.`} />
      </ReportSection>}
      {visible("gpu-contexts") && <EvidenceChart id="gpu-contexts" title="NVML наблюдал второй процесс с 2,18 минуты" queryId="telemetry"
        spec={gpuSpec} rows={telemetry} sourceRows={telemetry} height={300} />}
    </section>

    <section className="report-section">
      {visible("run-progress") && <EvidenceChart id="run-progress" title="M1 остановлена после 3521 завершённого шага SGD" queryId="progress"
        spec={progressSpec} rows={progress} sourceRows={progress} height={300} />}
      <RichNarrative id="run-progress:scope" value="Показан последний подтверждённый шаг в каждой целой секунде; полные записи сохранены. Выполнено 449984 предъявления обучающих изображений с повторениями. Эта кривая отражает расход и ход обучения, но не качество на отложенных данных. M1 не сохранялась как готовая исходная модель и не будет использована для подмены нового независимого обучения." />
    </section>

    {visible("methodology") && <ReportSection id="methodology" title="Проверки кода не заменяют проверку среды" queryId="summary" sourceRows={summary} showHeading={false}>
      <RichNarrative id="methodology:body" value={`## Проверки кода не заменяют проверку среды

До запуска прошли **395 CPU-тестов на Windows**; 20 Linux-проверок были пропущены. Отдельно прошли 53 проверки CV-фасада. Затем **все 468 проверок из 18 модулей прошли на Linux из установленного wheel**, с отключённой CUDA. Прежние проверки внешней остановки использовали локальные PID и не покрыли расхождение пространств PID. Поэтому успешные тесты не означают, что надзор на A100 был исправен.

Новая защита сохраняет сырые GPU-строки: более одной немедленно прекращает работу. Одна строка без внешне подтверждённой связи с локальным PID и временем его создания также не даёт допуска. Отказ необратим; исчезновение строки в следующем снимке не разрешает продолжение. Очистка различает живые процессы, zombie и неизвестное состояние.

На текущем SSH-доступе виден контейнер. **Подтверждённый наблюдатель на стороне хоста ещё не подключён.** Это оставшаяся инфраструктурная задача перед повтором; флаг в JSON и числовое совпадение PID её не решают.`} />
    </ReportSection>}

    {visible("reproduction") && <ReportSection id="reproduction" title="Доказательства сохранены без исправления истории" queryId="summary" sourceRows={summary} showHeading={false}>
      <RichNarrative id="reproduction:body" value={`## Доказательства сохранены без исправления истории

После остановки проверены **1762 прежних файла**, 325 файлов нового установленного пакета и wheel. Исходный контроллер и все его хеши сохранены. Его \`report.json\` остался со значением \`started\`, потому что процесс был остановлен извне; итоговый статус \`interrupted_infrastructure\` установлен отдельно по журналам и квитанции остановки.

Внешняя защита учла 237,49 с. Обёртка экспорта была остановлена вместе с серией, поэтому сохранение доказательств выполнено отдельно один раз: ещё 191,84 с до закрытия архива. Оно уложилось в исходное окно и 14-минутный предел завершения, но не выдаётся за работу под уже завершившимся наблюдателем.

Исходная ревизия — [a23bc9e](https://github.com/v1docq/FedCore/commit/a23bc9ecf43d05e468ecf40979a5425873772b18). Полные журналы, манифесты и автономный \`replay.py\` находятся рядом с отчётом в \`results/independent-m0-a100-2026-10-30\`. Повтор расчёта отчёта не загружает модели, не импортирует Torch и не запускает обучение.

**Кластерная работа остановлена владельцем 30.10 в 15:23 МСК.** Квитанция подтверждает отсутствие наших живых процессов и GPU-процессов. Дальнейшая подготовка доступа и вычисления остановлены; новые действия на кластере до явного возобновления не выполняются.

После возможного возобновления следующий численный пакет потребует отдельного разрешения на одну попытку, свободной A100 и проверенного контроля пространств PID. Три M0 следует обучать заново по прежним семенам и численным пределам; расширение числа попыток, подмена семян и автоматические повторы не разрешены. [PR #140](https://github.com/v1docq/FedCore/pull/140) остаётся открытым черновиком.`} />
    </ReportSection>}
  </article>;
}
