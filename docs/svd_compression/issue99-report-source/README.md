# Исходники отчёта issue #99

`export.py` читает готовые журналы через `build_analysis`, сохраняет таблицы, исходные события и ограниченный снимок для HTML. Модели и данные он не обучает и не оценивает. `verify_run.py` — независимая программа проверки стандартной библиотекой Python; для полного аудита нужны исходные каталоги и бинарные артефакты по записанным путям.

`ReportContent.jsx`, `findings.js` и `report.css` задают русское содержимое отчёта. Численные строки находятся в `../results/issue-99-local-1-2026-10-20/reviewed-snapshot.json`. Идентификатор артефакта сохранён в снимке. `write-markdown.mjs` использует те же выводы для текстового отчёта.

Сборка требует установленного Data plugin версии 1.0.11. Использованный runtime имеет SHA-256 `9e3ede84b28aded3c7379b6e6a5611f0d95b9977eb0f781e279ceefabfcbd27e`; компилятор — `ec65601079c80bef7210f2c9760ff7f2db6adef28516ffff280d4c86f91d4078`. Загрузки пакетов из сети не нужны.

```powershell
node <data-plugin>/scripts/prepare-data-app.mjs --surface report --output <новый-проект> --snapshot <reviewed-snapshot.json>
# Прочитать AGENTS.md нового проекта. Скопировать три файла содержимого в src/content/report/.
node <data-plugin>/scripts/data-app.mjs build --project-dir <новый-проект> --separate-data
node <data-plugin>/scripts/data-app.mjs export-offline --project-dir <новый-проект> --output <новый-проект>/.data-app-offline/exports/issue-99-local-1-report.html
node docs/svd_compression/issue99-report-source/write-markdown.mjs
```

Переносить содержимое в действующий проект следует без изменения его идентификатора и сохранённых пользовательских правок. Итоговый HTML можно скопировать из каталога экспорта в `docs/svd_compression/issue-99-local-1-report.html`. Готовый HTML открывается самостоятельно и не требует Data plugin или локального сервера.
