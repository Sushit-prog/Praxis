"""Command-line entrypoint: `praxis run`, `praxis status`, `praxis show`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxis",
        description="Turn research into hardware-calibrated engineering blueprints.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    sub = parser.add_subparsers(dest="command", required=False)

    run = sub.add_parser("run", help="Run the Scout -> Analyst -> Architect -> Coder pipeline.")
    run.add_argument("--source", choices=["arxiv", "github", "hn"], default="arxiv")
    run.add_argument("--topic", required=True, help="Topic to scout for candidates.")
    run.add_argument("--limit", type=int, default=20, help="Max candidates to scout (default: 20).")

    sub.add_parser("status", help="Show candidate counts by status from the database.")

    show = sub.add_parser("show", help="Print the blueprint markdown for a candidate.")
    show.add_argument("candidate_id", type=int, help="Candidate id to show.")

    return parser


def _cmd_run(args) -> int:
    from praxis.pipeline import format_summary, run

    result = run(source=args.source, topic=args.topic, limit=args.limit)
    print(format_summary(result))
    return 0


def _cmd_status(args) -> int:
    from praxis.db import status_counts

    counts = status_counts()
    if not counts:
        print("No candidates in the database.")
        return 0
    print("Candidate counts by status:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    return 0


def _cmd_show(args) -> int:
    from praxis.db import Candidate, get_session, latest_blueprint

    session = get_session()
    try:
        candidate = session.get(Candidate, args.candidate_id)
    finally:
        session.close()

    if candidate is None:
        print(f"error: no candidate with id {args.candidate_id}", file=sys.stderr)
        return 1

    blueprint = latest_blueprint(args.candidate_id)
    if blueprint is None:
        print(f"error: candidate {args.candidate_id} has no blueprint", file=sys.stderr)
        return 1

    print(f"# {candidate.title}\n{candidate.url}\n")
    print(blueprint.blueprint_md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.command == "run":
        try:
            return _cmd_run(args)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            logging.error("praxis run failed: %s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "show":
        return _cmd_show(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
