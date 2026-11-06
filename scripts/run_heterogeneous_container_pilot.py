"""Explicit one-shot container pilot; see heterogeneous_container for its limits."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedcore.experiments.heterogeneous_container import main


if __name__ == "__main__":
    raise SystemExit(main())
