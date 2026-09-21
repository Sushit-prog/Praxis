"""Command-line entrypoint: `praxis run`, `praxis status`, `praxis show`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from praxis.design import PASS_IDS

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
        help="Also process new/failed/reviewed/blueprinted candidates from earlier runs.",
    )
    # Tri-state: absent -> defer to PRAXIS_CODER; --prototype/--no-prototype override it.
    run.add_argument(
        "--prototype",
        dest="prototype",
        action="store_const",
        const=True,
        default=None,
        help="Enable the Coder stage (OpenCode CLI) for this run; overrides PRAXIS_CODER.",
    )
    run.add_argument(
        "--no-prototype",
        dest="prototype",
        action="store_const",
        const=False,
        help="Disable the Coder stage for this run; overrides PRAXIS_CODER.",
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

    export_parser = sub.add_parser(
        "export",
        help="Export a blueprint + coding-agent prompt as one markdown file (build kit).",
    )
    export_parser.add_argument("candidate_id", type=int, help="Candidate id to export.")
    export_parser.add_argument(
        "--out",
        help="Output file path (default: ./build-kit-<candidate_id>.md).",
    )

    eval_parser = sub.add_parser(
        "eval", help="Run the golden-set evaluation harness against the Analyst and Architect."
    )
    eval_parser.add_argument(
        "--golden", help="Path to the golden-set JSON (default: bundled golden_candidates.json)."
    )
    eval_parser.add_argument(
        "--threshold", type=int, help="Feasibility threshold override for the Analyst."
    )

    memory_parser = sub.add_parser(
        "memory", help="Show recent human review decisions (agent memory)."
    )
    memory_parser.add_argument(
        "--limit", type=int, default=10, help="Max entries to show (default: 10)."
    )

    sub.add_parser("providers", help="Show live provider pool health (cooldowns, signals).")

    sub.add_parser(
        "doctor", help="Run pre-flight checks (env, provider keys, DB, coder)."
    )

    review = sub.add_parser("review", help="Review borderline candidates (human-in-the-loop gate).")
    review_sub = review.add_subparsers(dest="review_action")
    approve_parser = review_sub.add_parser(
        "approve", help="Approve a borderline candidate and build it."
    )
    approve_parser.add_argument("candidate_id", type=int, help="Candidate id to approve.")
    approve_parser.add_argument(
        "--prototype",
        dest="prototype",
        action="store_const",
        const=True,
        default=None,
        help="Also draft a prototype via the OpenCode CLI (overrides PRAXIS_CODER).",
    )
    reject_parser = review_sub.add_parser(
        "reject", help="Reject a borderline candidate; it will not be built."
    )
    reject_parser.add_argument("candidate_id", type=int, help="Candidate id to reject.")

    discover_parser = sub.add_parser(
        "discover",
        help="Scout + analyze a topic, print a decision table, then pick one to design.",
    )
    discover_parser.add_argument("--topic", required=True, help="Topic to scout for candidates.")
    discover_parser.add_argument("--source", choices=["arxiv", "github", "hn"], default="arxiv")
    discover_parser.add_argument(
        "--limit", type=int, default=10, help="Max candidates to scout (default: 10)."
    )
    discover_parser.add_argument(
        "--pick",
        type=int,
        default=None,
        metavar="N",
        help="Pick table row N non-interactively (skips all prompts).",
    )
    discover_parser.add_argument(
        "--focus", help="Focus note steering the design (skips the interactive focus prompt)."
    )

    design_parser = sub.add_parser(
        "design", help="Generate a multi-pass design document for a candidate."
    )
    design_parser.add_argument("candidate_id", type=int, help="Candidate id to design.")
    design_parser.add_argument("--focus", help="Optional focus note steering the design.")
    design_parser.add_argument(
        "--depth", choices=["standard", "deep"], default="standard", help="Design depth."
    )
    design_parser.add_argument(
        "--model",
        help="Design model override (default: PRAXIS_DESIGN_MODEL or groq/openai/gpt-oss-120b).",
    )
    design_parser.add_argument(
        "--pass",
        dest="rerun_pass",
        type=int,
        default=None,
        metavar="N",
        choices=range(1, len(PASS_IDS) + 1),
        help=(
            "Re-run only pass N (1=technique, 2=architecture, 3=data_contracts, "
            "4=plan, 5=hardware_fit), keeping the other stored passes."
        ),
    )
    design_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a partial (in_progress) design from the last completed "
            "pass instead of starting a new one."
        ),
    )
    design_parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Skip the chunked critic review (saves one call per section).",
    )

    return parser


def _load_env() -> str | None:
    """Load a .env file into the process environment (first command, before config).

    Uses ``find_dotenv(usecwd=True)`` so the search starts at the caller's
    working directory, and ``override=False`` so variables already exported in
    the real environment win over .env values. Loaded values land in
    ``os.environ``, which is all the pass-through litellm needs for the plain
    provider keys (GROQ_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY); the
    PRAXIS_<PROVIDER>_API_KEY overrides are picked up later by
    :func:`praxis.llm._inject_provider_key`. Returns the loaded path or None.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        logging.getLogger(__name__).warning("python-dotenv not installed; skipping .env load")
        return None
    path = find_dotenv(usecwd=True)
    if not path:
        return None
    load_dotenv(path, override=False)
    return path


def _ensure_schema() -> None:
    """Create the SQLite schema if it does not exist yet (idempotent).

    Runs once before any command so a fresh checkout has tables instead of
    "no such table" OperationalErrors from status/show/memory/review/usage.
    Also creates the parent directory of a configured SQLite file path, since
    SQLite cannot create missing directories itself.
    """
    import os
    from pathlib import Path

    from sqlalchemy.engine import make_url

    from praxis.db import init_db

    url = os.environ.get("PRAXIS_DB_URL")
    if url:
        parsed = make_url(url)
        if parsed.drivername.startswith("sqlite"):
            database = parsed.database  # SQLAlchemy-normalized path (None = in-memory)
            if database and database != ":memory:":
                Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    init_db()


def _cmd_run(args) -> int:
    from praxis.pipeline import format_summary, run

    result = run(
        source=args.source,
        topic=args.topic,
        limit=args.limit,
        resume=args.resume,
        prototype=args.prototype,
    )
    print(format_summary(result))
    # A batch-level stage failure (e.g. scout) must be visible in the exit code.
    return 1 if result.stage_failures else 0


def _cmd_export(args) -> int:
    from praxis.db import Candidate, get_session
    from praxis.export import export_blueprint

    out_path = export_blueprint(args.candidate_id, out=args.out)
    if out_path is None:
        session = get_session()
        try:
            exists = session.get(Candidate, args.candidate_id) is not None
        finally:
            session.close()
        if not exists:
            print(f"error: no candidate with id {args.candidate_id}", file=sys.stderr)
        else:
            print(
                f"error: candidate {args.candidate_id} has no blueprint "
                "(run the pipeline first)",
                file=sys.stderr,
            )
        return 1
    print(f"Exported build kit for candidate {args.candidate_id} -> {out_path}")
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
        lines.append(f"    {label}: {t.calls} calls, {t.total_tokens:,} tokens, ${t.cost_usd:.4f}")
    lines.append("  by model:")
    for model, t in sorted(summary.by_model.items(), key=lambda kv: kv[1].cost_usd, reverse=True):
        lines.append(f"    {model}: {t.calls} calls, {t.total_tokens:,} tokens, ${t.cost_usd:.4f}")
    return "\n".join(lines)


def _cmd_usage(args) -> int:
    from praxis.db import usage_summary

    if args.days < 1:
        print("error: --days must be >= 1", file=sys.stderr)
        return 1
    # Schema is guaranteed by _ensure_schema(); a zero-row ledger is the empty case.
    summary = usage_summary(days=args.days)
    if summary.totals.calls == 0:
        print("No LLM usage recorded yet (run the pipeline first).")
        return 0
    print(_format_usage_report(summary))
    return 0


def _cmd_memory(args) -> int:
    from praxis.db import recent_build_memory

    entries = recent_build_memory(args.limit)
    if not entries:
        print("No build memory recorded yet.")
        return 0
    print("Recent build memory:")
    for entry in entries:
        print(f"  [{entry.id}] {entry.decision} ({entry.outcome}): {entry.technique}")
    return 0


def _cmd_providers(args) -> int:
    from datetime import UTC, datetime

    from praxis.db import provider_health_rows

    # Schema is guaranteed by _ensure_schema(); no rows is the empty case.
    rows = provider_health_rows()
    if not rows:
        print("Provider pool: no health state recorded yet (run the pipeline first).")
        return 0

    now = datetime.now(UTC).replace(tzinfo=None)
    print("Provider pool health:")
    for row in rows:
        if row.state == "cooling_down":
            leftover_s = (
                round((row.cooldown_until - now).total_seconds()) if row.cooldown_until else 0
            )
            signal = f" ({row.last_signal})" if row.last_signal else ""
            state = f"cooling_down {max(0, leftover_s)}s left{signal}"
        else:
            state = "healthy"
        print(f"  [{row.job}] {row.provider}: {state}")
    return 0


def _cmd_doctor(args) -> int:
    from praxis.doctor import run_doctor_checks

    print("Praxis doctor — pre-flight checks:")
    failed = False
    for check in run_doctor_checks():
        if check.skipped:
            print(f"  [ -- ] {check.name}: {check.detail}")
            continue
        marker = " ok " if check.ok else "FAIL"
        print(f"  [{marker}] {check.name}: {check.detail}")
        if not check.ok:
            failed = True
            if check.hint:
                print(f"         fix: {check.hint}")
    if failed:
        print("doctor: some checks failed (see the fix hints above)", file=sys.stderr)
        return 1
    print("doctor: all checks passed")
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
        from praxis.providers import ProviderUnavailableError

        try:
            result = approve(args.candidate_id, prototype=args.prototype)
        except ProviderUnavailableError as exc:
            # Provider-level failure (bad keys or all rate-limited): the
            # candidate stays `reviewed` (retryable); report cleanly.
            print(f"error: {exc}", file=sys.stderr)
            return 1
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


def _discard_partial_design(candidate_id: int) -> None:
    """Mark a stale in_progress design failed so a fresh run starts clean.

    Without --resume, a re-run of `praxis design` starts over instead of
    silently continuing a partial design; --resume keeps it.
    """
    from sqlalchemy import select

    from praxis.db import Design, get_session

    session = get_session()
    try:
        row = session.scalars(
            select(Design)
            .where(Design.candidate_id == candidate_id, Design.status == "in_progress")
            .order_by(Design.id.desc())
            .limit(1)
        ).first()
        if row is not None:
            row.status = "failed"
            session.commit()
    finally:
        session.close()


def _design_and_write(
    candidate,
    profile,
    *,
    focus,
    depth="standard",
    model=None,
    rerun_passes=None,
    run_critic=True,
) -> int:
    """Run the design generator, write DESIGN/TASKS/AGENT_PROMPT, print paths."""
    from praxis.design import generate_design
    from praxis.design_io import write_design_files

    result = generate_design(
        candidate,
        profile,
        depth=depth,
        focus=focus,
        model=model,
        rerun_passes=rerun_passes,
        run_critic=run_critic,
    )
    if result.status != "complete" or not result.design_md:
        completed = ", ".join(result.completed_passes) or "none"
        print(
            f"error: design failed for candidate {result.candidate_id}: "
            f"{result.error or 'incomplete passes'} (completed passes: {completed}; "
            f"re-run the same command to resume)",
            file=sys.stderr,
        )
        return 1

    from praxis.db import get_session
    from praxis.design_io import load_passes as _load_passes

    session = get_session()
    try:
        from praxis.db import Design

        row = session.get(Design, result.design_id)
        passes = _load_passes(row) if row is not None else {}
    finally:
        session.close()

    out_dir = write_design_files(result, candidate, profile, passes=passes)
    print(
        f"design complete: candidate {result.candidate_id} "
        f"({len(result.defects)} critic defect(s) addressed)"
    )
    print(f"  {out_dir / 'DESIGN.md'}")
    print(f"  {out_dir / 'TASKS.md'}")
    print(f"  {out_dir / 'AGENT_PROMPT.md'}")
    print(
        f"  LLM usage: {result.calls} calls, {result.total_tokens:,} tokens, "
        f"${result.cost_usd:.4f}"
    )
    for defect in result.defects:
        print(
            f"  critic: [{defect.get('class', '?')}] {defect.get('section', '?')}: "
            f"{defect.get('defect', '')}"
        )
    return 0


def _cmd_discover(args) -> int:
    from praxis.config import load_config
    from praxis.design_io import record_pick
    from praxis.discover import discover, focus_note, format_table, prompt_pick

    profile = load_config()
    result = discover(args.source, args.topic, limit=args.limit, profile=profile)
    print(format_table(result))
    if not result.pickable:
        print("nothing to pick", file=sys.stderr)
        return 1

    if args.pick is not None:
        row = next((r for r in result.pickable if r.index == args.pick), None)
        if row is None:
            print(f"error: no pickable row #{args.pick}", file=sys.stderr)
            return 1
        focus = args.focus
    else:
        row = prompt_pick(result)
        if row is None:
            print("no pick made — done.")
            return 0
        print(f"\ntechnique to build:\n  {row.technique}\n")
        focus = args.focus if args.focus is not None else focus_note(row)

    print(f"\npicked #{row.index}: {row.candidate.title}")
    candidate_id = getattr(row.candidate, "id", None)
    if candidate_id is None:
        print("error: picked candidate is not persisted", file=sys.stderr)
        return 1
    record_pick(candidate_id, row.technique, focus)
    return _design_and_write(row.candidate, profile, focus=focus)


def _cmd_design(args) -> int:
    from praxis.config import load_config
    from praxis.design import PASS_IDS
    from praxis.discover import get_candidate

    candidate = get_candidate(args.candidate_id)
    if candidate is None:
        print(f"error: no candidate with id {args.candidate_id}", file=sys.stderr)
        return 1
    profile = load_config()
    rerun_passes = None
    if args.rerun_pass is not None:
        rerun_passes = [PASS_IDS[args.rerun_pass - 1]]
        print(f"re-running pass {args.rerun_pass} ({rerun_passes[0]}) only")
    elif not args.resume:
        _discard_partial_design(args.candidate_id)
    print(f"designing candidate {args.candidate_id}: {candidate.title}")
    return _design_and_write(
        candidate,
        profile,
        focus=args.focus,
        depth=args.depth,
        model=args.model,
        rerun_passes=rerun_passes,
        run_critic=not args.no_critic,
    )


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

    # Load .env before anything reads configuration so PRAXIS_* variables and
    # provider keys in .env behave exactly like exported environment variables.
    _load_env()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    # Every command reads or writes the ledger; make sure the tables exist so a
    # fresh checkout gets clean empty output instead of "no such table" errors.
    if args.command is not None:
        _ensure_schema()

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
    if args.command == "memory":
        return _cmd_memory(args)
    if args.command == "providers":
        return _cmd_providers(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "show":
        return _cmd_show(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "eval":
        try:
            return _cmd_eval(args)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            logging.error("praxis eval failed: %s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "discover":
        try:
            return _cmd_discover(args)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            logging.error("praxis discover failed: %s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "design":
        try:
            return _cmd_design(args)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            logging.error("praxis design failed: %s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
