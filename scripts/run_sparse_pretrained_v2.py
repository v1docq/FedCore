"""Pinned independent-article BERT v2 series entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedcore.experiments.sparse_pretrained_v2 import main

if __name__ == "__main__":
    main()
