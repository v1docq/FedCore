"""One shared experiment entry point; scenario CLI lives in examples.petra.run."""
from __future__ import annotations


def main(argv=None):
    from examples.petra.run import main as scenario_main
    return scenario_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
