# Исходники автономной инфографики

Готовый файл: `../local-comparison-infographic.html`. Он содержит приложение, оформление и проверенные численные данные; для просмотра достаточно открыть HTML в браузере. Подключение к сети не требуется. Внешние ссылки ведут к исходным материалам в репозитории.

В этой папке сохранены редактируемые компоненты отчёта, снимок данных `data.json`, адаптер `report-adapter.mjs` и сведения о сборке. Полный проект со штатной средой Data App остаётся в локальном каталоге рабочей серии. Защищённые компоненты, редактор, просмотр источников и среда исполнения не изменялись.

## Данные и границы вывода

Снимок содержит 90 кандидатов, шесть исходных моделей, 30 парных сводок, 252 строки диагностики слоёв и условия 96 аппаратных сессий. Полные массивы по 200 повторов каждой сессии находятся в `../results/local-2026-10-13/summary.json`; в HTML сохранены p50/p95 и условия, без повторного встраивания этих массивов.

Основные выводы относятся к качеству и хранению выбранных малых моделей. Три seed дают описательный разброс, а не доказательство статистического превосходства. Измерения времени получены при фоновой нагрузке CPU и не обосновывают ранжирование методов по скорости. Аналитические MAC не являются измеренным временем.

`report-adapter.mjs` принимает четыре аргумента: проверенный `analysis.json`, полный `summary.json`, `protocol.json` и путь к `src/data.json` приложения. Рядом с полным `summary.json` должен находиться `final-audit.json`. Адаптер сохраняет идентификатор приложения, связывает измерения по `row_id`, добавляет исходные ранги и локальные ошибки, формирует компактные аппаратные строки и отметки Парето внутри каждой группы и seed. Научные исходные данные он не изменяет.

## Восстановление проекта и сборка

Требуются уже установленный плагин Data Analytics версии 1.0.11 и комплектный Node.js Codex. Загрузка пакетов и установка зависимостей не нужны. Команды выполняются из корня репозитория в PowerShell; `$app` должен указывать на новый каталог.

```powershell
$nodeExe = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
$dataPlugin = Join-Path $env:USERPROFILE '.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11'
$archive = Join-Path (Get-Location) 'docs/svd_compression/infographic-source'
$app = Join-Path (Get-Location) 'local-svd-report'

& $nodeExe "$dataPlugin/scripts/prepare-data-app.mjs" --surface report --output $app --snapshot "$archive/data.json"
Copy-Item -LiteralPath "$archive/data.json" -Destination "$app/src/data.json"
Copy-Item -Path "$archive/src/content/report/*" -Destination "$app/src/content/report/"

$env:CODEX_SESSION_ID = ''
$env:CODEX_THREAD_ID = ''
& $nodeExe "$dataPlugin/scripts/data-app.mjs" build --project-dir $app --separate-data
& $nodeExe "$dataPlugin/scripts/data-app.mjs" export-offline --project-dir $app --output '.data-app-offline/exports/local-comparison-infographic.html'
```

Повторное копирование `data.json` после подготовки восстанавливает исходный стабильный идентификатор приложения. Пустые переменные сеанса при первой сборке исключают локальную маршрутизацию к чату из переносимого HTML. Готовый автономный файл находится в `$app/.data-app-offline/exports/`; его можно скопировать в нужный каталог.

Для HTTP-предпросмотра следует запускать сервер из родительского каталога приложения с `--directory dist`. В Windows сервер с рабочим каталогом `dist` удерживает каталог и мешает атомарной замене результата при следующей сборке.
