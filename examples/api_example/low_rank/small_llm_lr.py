"""Compatibility entrypoint; the current scenario is recorded in its run manifest.

Run from the repository root with python -m examples.petra.run.
The historical filename is not evidence of the actual model architecture.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.petra.run import main

if __name__ == "__main__":
    raise SystemExit(main(default_scenario='lm', default_methods=['baseline', 'train', 'svd']))
