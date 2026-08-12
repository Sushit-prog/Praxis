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
    run.add_argument(
        "--resume",
        action="store_true",
        help="Also process candidates left in status new/failed by earlier runs.",
    )

    sub.add_parser("status", help="Show candidate counts by status from the database.")

    usage_parser = sub.add_parser(
        "usage", help="Show LLM token usage and estimated spend from the ledger."
    )
    usage_parser.add_argument(
        "--days", type=int, default=30, help="Recent window in days (default: 30)."
    )

    show = sub.add_parser("show", help="Print the blueprint markdown for a candidate.")
    show.add_argument("candidate_id", type=int, help="Candidate id to show.")

    eval_parser = sub.add_parser(
        "eval", help="Run the golden-set evaluation harness against the Analyst and Architect."
    )
    eval_parser.add_argument(
        "--golden", help="Path to the golden-set JSON (default: bundled golden_candidates.json)."
    )
    eval_parser.add_argument(
        "--threshold", type=int, help="Feasibility threshold override for the Analyst."
    )

    review = sub.add_parser(
        "review", help="Review borderline candidates (human-in-the-loop gate)."
    )
    review_sub = review.add_subparsers(dest="review_action")
    approve_parser = review_sub.add_parser(
        "approve", help="Approve a borderline candidate and build it."
    )
    approve_parser.add_argument("candidate_id", type=int, help="Candidate id to approve.")
    reject_parser = review_sub.add_parser(
        "reject", help="Reject a borderline candidate; it will not be built."
    )
    reject_parser.add_argument("candidate_id", type=int, help="Candidate id to reject.")

    return parser


def _cmd_run(args) -> int:
    from praxis.pipeline import format_summary, run

    result = run(source=args.source, topic=args.topic, limit=args.limit, resume=args.resume)
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


def _cmd_eval(args) -> int:
    import gc
    import os
    import tempfile
    from pathlib import Path

    from praxis.config import load_config
    from praxis.db import init_db
    from praxis.eval import default_golden_path, format_report, load_golden_set, run_eval

    golden_path = Path(args.golden) if args.golden else default_golden_path()
    if not golden_path.exists():
        print(f"error: golden set not found at {golden_path}; pass --golden", file=sys.stderr)
        return 1
    fixtures = load_golden_set(golden_path)
    # Evaluate against a throwaway DB so eval candidates never pollute the real ledger.
    # Agent sessions create their own engines; gc.collect() breaks the engine/pool
    # reference cycles so the SQLite file is released before the temp dir is removed
    # (ignore_cleanup_errors covers platforms where the OS still holds the file).
    previous_url = os.environ.get("PRAXIS_DB_URL")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["PRAXIS_DB_URL"] = f"sqlite:///{tmp}/eval.db"
        try:
            init_db()
            report = run_eval(fixtures, load_config(), threshold=args.threshold)
        finally:
            gc.collect()  # release engine/pool reference cycles so the file unlocks
            if previous_url is None:
                os.environ.pop("PRAXIS_DB_URL", None)
            else:
                os.environ["PRAXIS_DB_URL"] = previous_url
    print(format_report(report))
    return 0 if report.passed else 1


def _format_usage_report(summary) -> str:
    """Render the `praxis usage` report from a UsageSummary."""
    totals = summary.totals
    lines = [
        "LLM usage",
        f"  all time: {totals.calls} calls ({totals.cached_hits} from cache), "
        f"{totals.total_tokens:,} tokens "
        f"(prompt {totals.prompt_tokens:,} / completion {totals.completion_tokens:,}), "
        f"${totals.cost_usd:.4f}",
        f"  last {summary.days} days: {summary.recent.calls} calls "
        f"({summary.recent.cached_hits} from cache), "
        f"{summary.recent.total_tokens:,} tokens, ${summary.recent.cost_usd:.4f}",
        "  by stage:",
    ]
    for stage, t in sorted(summary.by_stage.items(), key=lambda kv: kv[1].cost_usd, reverse=True):
        label = stage or "(uncategorized)"
        lines.append(
            f"    {label}: {t.calls} calls, {t.total_tokens:,} tokens, ${t.cost_usd:.4f}"
        )
    lines.append("  by model:")
    for model, t in sorted(summary.by_model.items(), key=lambda kv: kv[1].cost_usd, reverse=True):
        lines.append(f"    {model}: {t.calls} calls, {t.total_tokens:,} tokens, ${t.cost_usd:.4f}")
    return "\n".join(lines)


def _cmd_usage(args) -> int:
    from sqlalchemy.exc import OperationalError

    from praxis.db import usage_summary

    if args.days < 1:
        print("error: --days must be >= 1", file=sys.stderr)
        return 1
    try:
        summary = usage_summary(days=args.days)
    except OperationalError as exc:
        if "no such table" in str(exc):
            print("No LLM usage recorded yet (run the pipeline first).")
            return 0
        raise
    if summary.totals.calls == 0:
        print("No LLM usage recorded yet (run the pipeline first).")
        return 0
    print(_format_usage_report(summary))
    return 0


def _cmd_review(args) -> int:
    from praxis.review import approve, pending_candidates, reject

    if args.review_action is None:
        pending = pending_candidates()
        if not pending:
            print("No candidates awaiting review.")
            return 0
        print("Candidates awaiting review (borderline):")
        for cand in pending:
            score = cand.feasibility_score if cand.feasibility_score is not None else "?"
            print(f"  [{cand.id}] {cand.title} — score {score} ({cand.url})")
            if cand.feasibility_reasoning:
                print(f"        {cand.feasibility_reasoning}")
        return 0

    if args.review_action == "approve":
        result = approve(args.candidate_id)
    elif args.review_action == "reject":
        result = reject(args.candidate_id)
    else:
        return 1

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    if result.action == "approved":
        if result.status in ("failed", "prototype_failed"):
            print(f"approved {result.candidate_id}: {result.title} — build ended {result.status}")
        else:
            suffix = f" ({result.prototype_path})" if result.prototype_path else ""
            print(f"approved {result.candidate_id}: {result.title} -> {result.status}{suffix}")
    else:
        print(f"rejected {result.candidate_id}: {result.title}")
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
    if args.command == "usage":
        return _cmd_usage(args)
    if args.command == "review":
        return _cmd_review(args)
    if args.command == "show":
        return _cmd_show(args)
    if args.command == "eval":
        try:
            return _cmd_eval(args)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            logging.error("praxis eval failed: %s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
