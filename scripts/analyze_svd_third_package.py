"""Portable stdlib-only CLI for third-packet raw JSON and CSV reconstruction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcore.tools.svd_third_analysis import *  # noqa: F403


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)  # noqa: F405
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.destination)  # noqa: F405
