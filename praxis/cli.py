"""Command-line entrypoint: `praxis run --source arxiv --topic "..."`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxis",
        description="Turn research into hardware-calibrated engineering blueprints.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    run = sub.add_parser(
        "run", help="Run the Scout -> Analyst -> Architect -> Coder pipeline."
    )
    run.add_argument("--source", choices=["arxiv", "github", "hn"], default="arxiv")
    run.add_argument("--topic", required=True, help="Topic to scout for candidates.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            from praxis.pipeline import run_pipeline

            result = run_pipeline(source=args.source, topic=args.topic)
            print(result)
            return 0
        except NotImplementedError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
