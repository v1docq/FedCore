"""Explicit entry point; --execute is required for any ML/CUDA operation."""
from pathlib import Path
import sys
import time

started = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcore.experiments.cv_sparse_pilot import main

if __name__ == "__main__":
    raise SystemExit(main(execute_started=started))
