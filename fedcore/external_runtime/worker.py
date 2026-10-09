"""Private subprocess entry point. Receives request.json and writes result.json."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from .contracts import CompressionRequest, ContractError, plan_request
from .execution import execute_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    root = Path(args.job_dir).resolve()
    try:
        path = root / "request.json"
        if path.stat().st_size > 65536:
            raise ContractError("size_limit", "Request JSON exceeds 64 KiB")
        request = CompressionRequest.parse(json.loads(path.read_text(encoding="utf-8")))
        result = execute_plan(plan_request(request), root)
        code = 0
    except Exception as error:
        detail = error.to_dict() if hasattr(error, "to_dict") else {"code": "execution_failed", "message": str(error), "path": "worker"}
        result = {"version": 1, "status": "failed", "error": detail}
        code = 1
        for path in root.glob("compressed.*"):
            path.unlink(missing_ok=True)
    temporary = root / ".result.json.tmp"
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, root / "result.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
