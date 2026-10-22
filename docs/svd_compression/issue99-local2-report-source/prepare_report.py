"""Prepare reviewed report rows from the frozen analysis; never execute models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


COMPONENTS = {
    "resources": ["local2-summary", "local2-resources", "resource-table"],
    "baselines": ["local2-data", "baseline-table"],
    "search_summary": ["local2-search", "summary-attempts", "summary-time", "summary-table"],
    "search_runs": ["search-detail-table", "local2-methods"],
    "curves": ["search-curve", "search-attempt-curve"],
    "paired": ["paired-detail", "local2-search"],
    "common_bytes": ["common-bytes", "local2-search"],
    "oracles": ["oracle-table", "local2-search"],
    "frontiers": ["frontier-chart", "local2-quality"],
    "test_selected": ["test-table", "local2-quality"],
    "matrix_summary": ["local2-summary", "local2-matrix", "matrix-ratios", "matrix-summary-table", "local2-next"],
    "matrix_streams": ["matrix-stream-table", "matrix-times"],
    "matrix_pairs": ["matrix-pairs-table", "local2-matrix"],
    "matrix_controls": ["matrix-controls-table", "local2-matrix"],
    "plateau": ["plateau-table", "local2-methods"],
    "jobs": ["job-table", "local2-resources"],
}


def flat_rows(rows):
    return [{key: value for key, value in row.items()
             if not isinstance(value, (dict, list, tuple))} for row in rows]


def prepare(archive):
    analysis = json.loads((archive / "analysis.json").read_text(encoding="utf-8"))
    resources = analysis["resources"]
    tables = {name: flat_rows(analysis[name]) for name in COMPONENTS
              if name not in {"resources", "matrix_pairs", "matrix_controls", "plateau"}}
    tables["plateau"] = [dict(job=row["job"], model_seed=row["model_seed"], budget=row["budget"],
        mode=row["mode"], search_seed=row["search_seed"], **row["virtual_stop"])
        for row in analysis["search_runs"]]
    streams = analysis["matrix_streams"]
    timed = {(row["n"], row["invalid_count"], row["block"], row["arm"]): row
             for row in streams if row["group"] == "timed" and row["request_count"] == 8}
    controls = {(row["n"], row["invalid_count"], row["pair"], row["label"]): row
                for row in streams if row["group"] == "control"}
    tables["matrix_pairs"], tables["matrix_controls"] = [], []
    for n, invalid, block in sorted({key[:3] for key in timed}):
        if not all((n, invalid, block, arm) in timed for arm in "ABC"):
            continue
        b = timed[n, invalid, block, "B"]
        for arm in "AC":
            a = timed[n, invalid, block, arm]
            tables["matrix_pairs"].append(dict(n=n, invalid_count=invalid, arm=arm, block=block,
                a_seconds=a["primary_seconds"], b_seconds=b["primary_seconds"],
                ratio=a["primary_seconds"] / b["primary_seconds"],
                difference_seconds=a["primary_seconds"] - b["primary_seconds"],
                before_b=a["started_at"] < b["started_at"],
                resource_eligible=a["resource_eligible"] and b["resource_eligible"]))
    for n, invalid, pair in sorted({key[:3] for key in controls}):
        if not all((n, invalid, pair, label) in controls for label in ("B1", "B2")):
            continue
        b1, b2 = (controls[n, invalid, pair, label]["primary_seconds"] for label in ("B1", "B2"))
        tables["matrix_controls"].append(dict(n=n, invalid_count=invalid, pair=pair,
            B1_seconds=b1, B2_seconds=b2, ratio=b1 / b2,
            resource_eligible=all(controls[n, invalid, pair, label]["resource_eligible"] for label in ("B1", "B2"))))
    tables["resources"] = flat_rows([resources])
    for row in tables["resources"]:
        row["elapsed_minutes"] = row["elapsed_seconds"] / 60
        row["max_tree_rss_MiB"] = row["max_tree_rss_bytes"] / 1024**2 if row["max_tree_rss_bytes"] is not None else None
        row.update(wall_limit_minutes=120, mlp_limit=10300, matrix_limit=4350,
                   training_update_limit=2040, tree_rss_limit_GiB=8,
                   matrix_request_limit_MiB=1024)
    for name, rows in tables.items():
        for row in rows:
            if row.get("model_seed") is not None:
                row["model_seed"] = str(row["model_seed"])
            if row.get("budget") is not None:
                row["budget_label"] = {"attempts": "До 24 оценок", "time": "30 секунд", "oracle": "Полный каталог"}[row["budget"]]
            if row.get("mode") in ("R", "T", "E"):
                row["mode_label"] = {"R": "R · готовые планы", "T": "T · случайный обход", "E": "E · эволюционный поиск"}[row["mode"]]
            if row.get("invalid_fraction") is not None:
                row["invalid_percent"] = row["invalid_fraction"] * 100
            if row.get("invalid_count") is not None:
                row["invalid_label"] = f'{row["invalid_count"] * 12.5:g}%'
            if row.get("arm") is not None:
                row["arm_label"] = {"A": "A · рабочая проверка", "B": "B · поздний отказ", "C": "C · независимый контракт"}.get(row["arm"], row["arm"])
            if row.get("n") is not None:
                row["size_label"] = f'{row["n"]} × {row["n"]}'
    now = datetime.now(timezone.utc).isoformat()
    revision = analysis["code_revision"]
    definitions = {
        "resources": ["Расходы включают импорт, обучение, остановленные и завершённые попытки; пределы не перераспределяются между научными условиями."],
        "baselines": ["Три новых разбиения одних 1797 изображений Digits. В каждом 1078/269/180/270 изображений для обучения/validation/калибровки/test. Разбиения пересекаются; инициализация 101 общая."],
        "search_summary": ["Медиана и диапазон трёх поисковых состояний 11/22/33 внутри одного разбиения. В сводку входят только полностью допустимые поисковые исполнения."],
        "search_runs": ["Попытка учитывается до вычисления; незавершённые попытки расходуют бюджет. В архив поиска входят лишь оценки, завершённые до срока. R исчерпывает каталог из 15 решений; T и E используют каталог 440."],
        "curves": ["Накопленный архив конечных оценок, завершённых в срок. Одна ось — секунды от запуска дочернего процесса, включая импорт; другая — число завершённых оценок в срок (attempt). Начатые, но незавершённые попытки входят в расход, однако не добавляют точек кривой."],
        "paired": ["Парное различие нормированного гиперобъёма: второй режим минус первый, для одного разбиения, состояния поиска и бюджета. Неполная пара имеет null."],
        "common_bytes": ["Лучший CE на validation при точно 15 656 или 42 280 байтах. Это наблюдавшиеся решения; отсутствие оценки не означает равенство или ноль."],
        "oracles": ["Полный эталон означает успешную проверку всех 440 уникальных конечных решений. Частичный архив не считается полным эталоном."],
        "frontiers": ["Недоминируемые наблюдавшиеся решения по CE и полному числу байтов тензоров модели. Это validation, а не test."],
        "test_selected": ["Выбор минимального validation CE под тремя пределами хранения 20 671/41 342/58 568 байт зафиксирован хешем до чтения test. Повторные логические выборы не являются независимыми моделями."],
        "matrix_summary": ["Отношение времени A/B или C/B рассчитано в шести полных блоках ABC, без импорта и проверок после вызова. Выигрыш: медиана ≤0,9 и ≥5 из 6 отношений <1, при прошедших B/B, порядке и длительности B ≥50 мс."],
        "matrix_streams": ["Один поток — восемь запросов; primary_seconds — сумма времени вызовов. Группы timed/control/warmup разделены. Контроль B/B имеет одинаковое исполнение с разными позициями."],
        "matrix_pairs": ["Все отношения A/B и C/B из полных блоков ABC. Числитель и знаменатель — primary_seconds потоков одного размера, доли отказов и блока. before_b означает исполнение A или C раньше B."],
        "matrix_controls": ["Все четыре контроля B/B на условие. Отношение равно B1_seconds/B2_seconds; позиции B1/B2 сбалансированы. Отсутствующая пара не добавляется искусственно."],
        "plateau": ["Ретроспективная диагностика завершённой траектории: после первых 12 конечных оценок шесть подряд приростов ненормированного HV не больше 1e-6. Она не останавливала вычисления; actual_time_saved остаётся null. R неприменим: в каталоге только 15 решений."],
        "jobs": ["Статус и время каждого дочернего процесса взяты из родительского журнала; RSS относится к дереву одного процесса."],
    }
    hv_queries = {"search_summary", "search_runs", "curves", "paired", "oracles"}
    queries = {}
    for name, rows in tables.items():
        metric_definitions = [{"label": name, "definition": value,
                               "componentIds": COMPONENTS[name]}
                              for value in definitions[name]]
        if name in hv_queries:
            metric_definitions.append({"label": "Нормированный гиперобъём",
                "definition": "Минимизируются CE/ln(10) и tensor_bytes/68904. Опорная точка (2; 1,05); площадь делится на 2,1. Больше — лучше. Отдельно учитывается допуск CE≤M0+0,05 и accuracy≥M0−0,02.",
                "componentIds": COMPONENTS[name]})
        queries[name] = {"rows": rows,
            "source": {"label": "FedCore #99 · второй локальный CPU-пакет",
                "files": ["analysis.json", name + ".csv", "raw-evidence.zip"],
                "executedAt": now,
                "evidenceFlow": [
                    {"title": "Протокол и исходники", "detail": "protocol.json, PLAN_ISSUE_99_LOCAL_2.md; commit " + revision + "; поправка памяти 1 GiB утверждена 21.10.2026."},
                    {"title": "Измерения", "detail": "Единственное последовательное исполнение local2-run-1; исходные журналы в raw-evidence.zip; артефакты перечислены с SHA-256 в artifact-manifest.json."},
                    {"title": "Пересчёт", "detail": "issue99-local2-report-source/export.py → analysis.json; prepare_report.py → плоские строки и CSV. Модели повторно не запускались."}],
                "metricDefinitions": metric_definitions,
                "caveats": ["Новые разбиения ранее использованного Digits не дают независимой внешней проверки обобщения.", "Одно CPU-окружение, два математических потока; синтетические квадратные матрицы float64. На GPU и другие структуры результат не переносится автоматически."]},
            "methods": [{"language": "python", "code": "python docs/svd_compression/issue99-local2-report-source/export.py --run RUN --output ARCHIVE\npython docs/svd_compression/issue99-local2-report-source/prepare_report.py --archive ARCHIVE"}]}
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (archive / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    snapshot = {"surface": "report", "title": "FedCore: поиск сжатия на новых разбиениях и цена раннего отказа",
        "generatedAt": now, "buildStatus": "creating", "status": "observed",
        "report": {"asOf": "2026-10-21"}, "filters": [], "queries": queries}
    previous = archive / "reviewed-snapshot.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("id"):
            snapshot["id"] = old["id"]
    previous.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in tables.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    prepare(parser.parse_args().archive)
