"""Read verified measured records; no estimated or manually entered results."""
from __future__ import annotations

import argparse
from pathlib import Path


def read_results(path: Path):
    from fedcore.experiments.runner import load_run
    manifest = load_run(path)
    rows = []
    for candidate in manifest["candidates"]:
        measurement, quality = candidate.get("measurement", {}).get("metrics", {}), candidate.get("validation", {})
        rows.append({"candidate_id": candidate["candidate_id"], "method": candidate["configuration"]["method"],
                     "status": candidate["status"], "validation_metric": quality.get("metric"),
                     "validation_value": quality.get("value"), "test": candidate.get("test"),
                     "file_bytes": measurement.get("file_bytes"), "latency_p50_ms": measurement.get("latency_p50_ms"),
                     "reason": candidate.get("reason")})
    return manifest, rows


def main(argv=None):
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    manifest, rows = read_results(args.run_directory)
    print("Run status:", manifest["status"])
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
