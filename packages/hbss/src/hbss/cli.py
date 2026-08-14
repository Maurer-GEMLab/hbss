"""Command-line entry point.

Target for Phase 0::

    hbss run configs/toy.yaml

which resolves the config, runs the model, and writes a run directory
containing outputs and a complete ``run.json`` provenance ledger. That command
working end to end is the Phase 0 exit criterion.

Status: skeleton.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hbss", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run inference from a config file")
    run.add_argument("config", help="path to a YAML run configuration")
    run.add_argument("--out", default="runs/", help="output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"hbss {args.command}: not implemented yet (Phase 0)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
