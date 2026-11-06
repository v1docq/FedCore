"""Prepare the frozen Q-only single-boundary ONNX diagnostic on CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedcore.experiments.heterogeneous_qonly_placement import main


if __name__ == "__main__":
    raise SystemExit(main())
