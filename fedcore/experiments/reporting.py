"""Tables and plots derived exclusively from unrounded, identified run records."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .protocol import ProtocolError


def relative_change(baseline, candidate):
    if baseline is None or candidate is None:
        return {"status": "unavailable", "percent": None, "reason": "measurement unavailable"}
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for value in (baseline, candidate)):
        return {"status": "invalid", "percent": None, "reason": "nonfinite/invalid measurement"}
    if baseline == 0:
        return {"status": "defined" if candidate == 0 else "undefined", "percent": 0.0 if candidate == 0 else None,
                "reason": None if candidate == 0 else "zero baseline"}
    return {"status": "defined", "percent": (candidate / baseline - 1) * 100, "reason": None}


def audit_historical_percent(baseline, candidate, published_percent, *, tolerance=0.1):
    result = relative_change(baseline, candidate)
    calculated = result["percent"]
    result.update(source="published rounded values; raw measurements not reconstructed",
                  units="unknown unless documented by the historical measurement log",
                  published_percent=published_percent,
                  mismatch=calculated is not None and abs(calculated - published_percent) > tolerance,
                  sign_mismatch=calculated is not None and calculated * published_percent < 0)
    return result


def table_rows(manifest):
    records = manifest.get("candidates", [])
    ids = [record["candidate_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ProtocolError("Duplicate configuration IDs cannot form an unambiguous table")
    baseline = next((record for record in records if record["configuration"]["method"] == "baseline" and record["status"] == "succeeded"), None)
    baseline_metrics = baseline["measurement"]["metrics"] if baseline else {}
    selected = set(manifest.get("selection", {}).get("archive_ids", []))
    rows = []
    for record in records:
        metrics = record.get("measurement", {}).get("metrics", {})
        validation = record.get("validation", {})
        test = record.get("test", {})
        row = {"candidate_id": record["candidate_id"], "method": record["configuration"]["method"],
               "configuration": json.dumps(record["configuration"], sort_keys=True),
               "status": record["status"], "reason": record.get("reason"),
               "validation_metric": validation.get("metric"), "validation_value": validation.get("value"),
               "validation_macro_f1": validation.get("macro_f1"), "test_value": test.get("value"),
               "test_status": "measured_after_selection" if test else "not_selected_or_unavailable",
               "test_macro_f1": test.get("macro_f1"), "quality_direction": validation.get("direction"),
               "selected_in_archive": record["candidate_id"] in selected,
               "wall_seconds": record.get("wall_seconds"), "cache_hit": record.get("cache_hit", False),
               "artifact_sha256": record.get("measurement", {}).get("artifact", {}).get("sha256"),
               "device": record.get("measurement", {}).get("profile", {}).get("device"),
               "runtime": record.get("measurement", {}).get("profile", {}).get("runtime"),
               "latency_unit": "ms/batch", "throughput_unit": "samples/s", "size_unit": "bytes",
               "repeats": record.get("measurement", {}).get("profile", {}).get("repeats"),
               "interval_kind": "p50/p95 empirical call quantiles; no confidence interval"}
        for name in ("latency_p50_ms", "latency_p95_ms", "throughput", "file_bytes", "tensor_state_bytes",
                     "cpu_rss_before_bytes", "cpu_rss_after_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            row[name] = metrics.get(name)
            if name in ("latency_p50_ms", "throughput", "file_bytes"):
                change = relative_change(baseline_metrics.get(name), metrics.get(name))
                row[f"{name}_relative_change_percent"] = change["percent"]
                row[f"{name}_relative_change_status"] = change["status"]
        base_quality = baseline.get("validation", {}).get("value") if baseline else None
        row["validation_absolute_change"] = validation["value"] - base_quality if validation.get("value") is not None and base_quality is not None else None
        change = relative_change(base_quality, validation.get("value"))
        row["validation_relative_change_percent"] = change["percent"]
        row["validation_relative_change_status"] = change["status"]
        rows.append(row)
    return rows


def write_reports(manifest, output_dir):
    """Generate recoverable CSV/JSON plus a figure of the actual archive."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = table_rows(manifest)
    (directory / "table.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    if rows:
        with (directory / "table.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = ["# Measured PETRA pilot", "", f"Run status: {manifest['status']}.",
              "Selection uses validation only. Final test scores are recorded after the archive freezes.",
              "All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.",
              "Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.",
              "p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.",
              "", "| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        report.append("| " + " | ".join(str(row.get(key)) if row.get(key) is not None else "unavailable" for key in
                      ("candidate_id", "method", "status", "validation_value", "test_value", "file_bytes", "latency_p50_ms")) + " |")
    report += ["", manifest.get("empirical_claim", "No empirical superiority claim."), "",
               "The manifest, prediction files and raw timing repetitions are the source of this table."]
    (directory / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        feasible = set(manifest.get("selection", {}).get("feasible_ids", ()))
        selected = set(manifest.get("selection", {}).get("archive_ids", ()))
        for index, record in enumerate(manifest.get("candidates", ())):
            if record["candidate_id"] in feasible:
                point = manifest["selection"]["objectives"][record["candidate_id"]]
                axes[0].scatter(*point, color="tab:orange" if record["candidate_id"] in selected else "tab:blue")
                axes[0].annotate(record["candidate_id"][:6], point, fontsize=7,
                                 xytext=(5, 5 + 9 * (index % 3)), textcoords="offset points")
        axes[0].set_xlabel("Validation loss / frozen quality scale")
        axes[0].set_ylabel(f"{manifest['protocol']['selection_cost']} / frozen cost scale")
        axes[0].set_title("Evaluated feasible candidates")
        trace = [item for item in manifest.get("search", {}).get("trace", ()) if item.get("event") == "evaluated"]
        if trace:
            axes[1].step([item["elapsed_seconds"] for item in trace], [item["hypervolume"] for item in trace], where="post")
            axes[1].set_xlabel("Full charged search wall time, s")
            axes[1].set_ylabel("Archive hypervolume at frozen reference")
            axes[1].set_title("Measured progress; pilot only")
        else:
            for index, row in enumerate(rows):
                if row.get("file_bytes") is not None and row.get("latency_p50_ms") is not None:
                    point = (row["file_bytes"] / 1024, row["latency_p50_ms"])
                    axes[1].scatter(*point)
                    axes[1].annotate(row["method"], point, fontsize=7,
                                     xytext=(5, 5 + 9 * (index % 3)), textcoords="offset points")
            axes[1].set_xlabel("Artifact file size, KiB")
            axes[1].set_ylabel("Measured latency p50, ms/batch")
            axes[1].set_title("Loaded artifacts; one pilot run")
        for axis in axes:
            axis.margins(x=.18, y=.25)
        figure.tight_layout()
        figure.savefig(directory / "archive.png", dpi=160)
        plt.close(figure)
    except ImportError:
        (directory / "figure_status.json").write_text(json.dumps({"status": "unsupported", "reason": "matplotlib unavailable"}), encoding="utf-8")
    return {"table_json": str(directory / "table.json"), "table_csv": str(directory / "table.csv"),
            "report": str(directory / "report.md")}
