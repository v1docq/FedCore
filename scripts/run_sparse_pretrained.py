"""Local pinned BERT/WikiText sparse experiment entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedcore.experiments.sparse_pretrained import main

if __name__ == "__main__":
    main()
