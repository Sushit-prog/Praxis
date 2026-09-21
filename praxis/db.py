"""SQLAlchemy models and SQLite engine setup."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

DEFAULT_DB_PATH = "./praxis.db"


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    """A source item (paper, repo, post) that survived scouting."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    discovered_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    status: Mapped[str] = mapped_column(String(32), default="new", index=True)
    technique_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    feasibility_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feasibility_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    blueprints: Mapped[list[Blueprint]] = relationship(back_populates="candidate")


class Blueprint(Base):
    """A hardware-calibrated engineering plan for a candidate."""

    __tablename__ = "blueprints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    feasibility_score: Mapped[float] = mapped_column(Float)
    blueprint_md: Mapped[str] = mapped_column(Text)
    prototype_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    candidate: Mapped[Candidate] = relationship(back_populates="blueprints")


class Design(Base):
    """A multi-pass design document for a candidate (discover -> design flow).

    ``passes_json`` stores the per-pass outputs (goal, architecture, ...)
    individually, so a failed pass leaves a resumable partial design: later
    runs redo only the missing passes, keyed by pass id. ``defects`` holds the
    final critic pass's defect list; ``depth`` records standard/deep.
    """

    __tablename__ = "designs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="in_progress", index=True)
    depth: Mapped[str] = mapped_column(String(16), default="standard")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    focus: Mapped[str | None] = mapped_column(Text, nullable=True)
    passes_json: Mapped[str] = mapped_column(Text, default="{}")
    defects: Mapped[str] = mapped_column(Text, default="")
    design_md: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    candidate: Mapped[Candidate] = relationship()


class LLMUsage(Base):
    """A single recorded LLM call: tokens, estimated cost, latency, and context.

    Written by the LLM wrapper on every completion that reports usage. This is
    an observability ledger, so ``candidate_id`` is a plain indexed integer
    rather than a foreign key: a usage row must never block candidate writes.
    """

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    model: Mapped[str] = mapped_column(String(128))
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    candidate_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String(512), nullable=True)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)


class BuildMemory(Base):
    """A human review decision and its build outcome (agent memory).

    Recorded when a person approves or rejects a borderline candidate and fed
    back into future Analyst scoring, so the system learns which techniques are
    actually buildable on the target hardware.
    """

    __tablename__ = "build_memory"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    technique: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(16))  # approved | rejected
    outcome: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LLMCache(Base):
    """Cached LLM responses keyed by a hash of model + system + prompt.

    Re-processing the same candidate with identical inputs is served from here
    instead of spending another LLM call; the key is a full sha256 of the input
    material, so any prompt or model change is a cache miss by construction.
    """

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(128))
    response: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ProviderHealth(Base):
    """Per-job provider state for instant failover (healthy | cooling_down).

    Written by the provider pool in :mod:`praxis.providers`: an exhausted
    provider (rate limit / quota / context window) is marked ``cooling_down``
    until a cooldown timestamp, and healthy/successful calls clear it. Cooldowns
    are persisted so they survive restarts: ``praxis run --resume`` honors an
    exhausted provider the moment it warms up again.
    """

    __tablename__ = "provider_health"
    __table_args__ = (UniqueConstraint("job", "provider", name="uq_provider_health"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job: Mapped[str] = mapped_column(String(32), index=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(16), default="healthy")  # healthy | cooling_down
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_signal: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


def _db_url() -> str:
    return (
        os.environ.get("PRAXIS_DB_URL")
        or f"sqlite:///{os.environ.get('PRAXIS_DB_PATH', DEFAULT_DB_PATH)}"
    )


def get_engine():
    """Create the SQLite engine, honoring PRAXIS_DB_URL / PRAXIS_DB_PATH."""
    return create_engine(_db_url())


def get_session():
    """Return a session bound to the default engine."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)()


def init_db(engine=None) -> None:
    """Create all tables if they do not yet exist."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Designs (discover -> design flow)
# ---------------------------------------------------------------------------


def latest_design(candidate_id: int) -> Design | None:
    """Return the most recent design for a candidate, or None."""
    session = get_session()
    try:
        return session.scalars(
            select(Design)
            .where(Design.candidate_id == candidate_id)
            .order_by(Design.id.desc())
            .limit(1)
        ).first()
    finally:
        session.close()


def save_design_pass(design_id: int, pass_id: str, content: str) -> None:
    """Persist one design pass output incrementally (resumable partial designs)."""
    session = get_session()
    try:
        row = session.get(Design, design_id)
        if row is None:
            return
        passes = json.loads(row.passes_json or "{}")
        passes[pass_id] = content
        row.passes_json = json.dumps(passes)
        session.commit()
    finally:
        session.close()


def clear_design_pass(design_id: int, pass_id: str) -> None:
    """Drop one pass output so the next run regenerates it (--pass N)."""
    session = get_session()
    try:
        row = session.get(Design, design_id)
        if row is None:
            return
        passes = json.loads(row.passes_json or "{}")
        passes.pop(pass_id, None)
        row.passes_json = json.dumps(passes)
        session.commit()
    finally:
        session.close()


def design_usage_totals(candidate_id: int) -> tuple[int, int, float]:
    """LLM calls, total tokens, and estimated cost for one candidate's designs.

    Covers the design and design_critic stages, excluding cache hits, so the
    CLI can print a per-design spend footer like the pipeline's.
    """
    session = get_session()
    try:
        count, tokens, cost = session.execute(
            select(
                func.count(LLMUsage.id),
                func.coalesce(func.sum(LLMUsage.total_tokens), 0),
                func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
            ).where(
                LLMUsage.candidate_id == candidate_id,
                LLMUsage.stage.in_(("design", "design_critic")),
                LLMUsage.cached.is_(False),
            )
        ).one()
        return int(count), int(tokens), float(cost)
    finally:
        session.close()


def design_status_counts() -> dict[str, int]:
    """Return counts of designs grouped by status."""
    session = get_session()
    try:
        rows = session.execute(
            select(Design.status, func.count(Design.id)).group_by(Design.status)
        ).all()
        return {status: count for status, count in rows}
    finally:
        session.close()


def status_counts() -> dict[str, int]:
    """Return counts of candidates grouped by status."""
    session = get_session()
    try:
        rows = session.execute(
            select(Candidate.status, func.count(Candidate.id)).group_by(Candidate.status)
        ).all()
        return {status: count for status, count in rows}
    finally:
        session.close()


def latest_blueprint(candidate_id: int) -> Blueprint | None:
    """Return the most recent blueprint for a candidate, or None."""
    session = get_session()
    try:
        return session.scalars(
            select(Blueprint)
            .where(Blueprint.candidate_id == candidate_id)
            .order_by(Blueprint.id.desc())
            .limit(1)
        ).first()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# LLM usage aggregation
# ---------------------------------------------------------------------------


@dataclass
class UsageTotals:
    """Aggregate token and cost counters over a set of LLM calls."""

    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cached_hits: int = 0


@dataclass
class UsageSummary:
    """Full usage picture: totals, a recent window, and breakdowns."""

    totals: UsageTotals
    recent: UsageTotals
    by_stage: dict[str | None, UsageTotals]
    by_model: dict[str, UsageTotals]
    days: int


def usage_totals(*, session=None, since: datetime | None = None) -> UsageTotals:
    """Aggregate token/cost totals over recorded LLM calls, optionally since a date.

    Cached hits carry zero tokens/cost and are counted separately so the spend
    figures stay accurate while still surfacing what the cache saved.
    """
    owns_session = session is None
    session = session or get_session()
    try:
        stmt = select(
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.prompt_tokens), 0),
            func.coalesce(func.sum(LLMUsage.completion_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
        ).where(LLMUsage.cached.is_(False))
        if since is not None:
            stmt = stmt.where(LLMUsage.created_at >= since)
        count, prompt, completion, total, cost = session.execute(stmt).one()

        hits_stmt = select(func.count(LLMUsage.id)).where(LLMUsage.cached.is_(True))
        if since is not None:
            hits_stmt = hits_stmt.where(LLMUsage.created_at >= since)
        cached_hits = session.execute(hits_stmt).scalar_one()

        return UsageTotals(
            calls=count,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=float(cost),
            cached_hits=cached_hits,
        )
    finally:
        if owns_session:
            session.close()


def _usage_grouped(session, column) -> dict[str | None, UsageTotals]:
    """Group usage totals by a column (stage or model); cache hits excluded."""
    rows = session.execute(
        select(
            column,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.prompt_tokens), 0),
            func.coalesce(func.sum(LLMUsage.completion_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
        )
        .where(LLMUsage.cached.is_(False))
        .group_by(column)
    ).all()
    return {
        key: UsageTotals(
            calls=count,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=float(cost),
        )
        for key, count, prompt, completion, total, cost in rows
    }


def usage_summary(*, session=None, days: int = 30) -> UsageSummary:
    """Aggregate all-time and recent usage plus per-stage/per-model breakdowns."""
    owns_session = session is None
    session = session or get_session()
    try:
        # created_at is server-side UTC (CURRENT_TIMESTAMP); compare against a
        # naive UTC cutoff so the window is not skewed by the machine's TZ.
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        return UsageSummary(
            totals=usage_totals(session=session),
            recent=usage_totals(session=session, since=cutoff),
            by_stage=_usage_grouped(session, LLMUsage.stage),
            by_model=_usage_grouped(session, LLMUsage.model),
            days=days,
        )
    finally:
        if owns_session:
            session.close()


def recent_build_memory(limit: int = 5, *, session=None) -> list[BuildMemory]:
    """Most recent build-memory entries, newest first, one per candidate.

    Dedupes by candidate so a candidate that somehow accrued multiple memory
    rows never floods the prompt with stale duplicates; the newest row wins.
    """
    owns_session = session is None
    session = session or get_session()
    try:
        entries = session.scalars(select(BuildMemory).order_by(BuildMemory.id.desc())).all()
        seen: set[int] = set()
        deduped: list[BuildMemory] = []
        for entry in entries:
            if entry.candidate_id in seen:
                continue
            seen.add(entry.candidate_id)
            deduped.append(entry)
            if len(deduped) >= limit:
                break
        return deduped
    finally:
        if owns_session:
            session.close()


# ---------------------------------------------------------------------------
# Provider health (cooldowns survive restarts via this table)
# ---------------------------------------------------------------------------


def provider_health_rows(*, session=None) -> list[ProviderHealth]:
    """Return every provider-health row, oldest row id first (insertion order)."""
    owns_session = session is None
    session = session or get_session()
    try:
        return list(session.scalars(select(ProviderHealth).order_by(ProviderHealth.id)).all())
    finally:
        if owns_session:
            session.close()


def get_provider_health(job: str, provider: str, *, session=None) -> ProviderHealth | None:
    """Return the health row for a (job, provider) pair, or None if never recorded."""
    owns_session = session is None
    session = session or get_session()
    try:
        return session.scalars(
            select(ProviderHealth).where(
                ProviderHealth.job == job, ProviderHealth.provider == provider
            )
        ).first()
    finally:
        if owns_session:
            session.close()


def set_provider_health(
    job: str,
    provider: str,
    *,
    state: str,
    cooldown_until=None,
    last_signal: str | None = None,
    session=None,
) -> None:
    """Upsert the health row for a (job, provider) pair; best-effort only."""
    owns_session = session is None
    session = session or get_session()
    try:
        row = session.scalars(
            select(ProviderHealth).where(
                ProviderHealth.job == job, ProviderHealth.provider == provider
            )
        ).first()
        if row is None:
            row = ProviderHealth(job=job, provider=provider)
            session.add(row)
        row.state = state
        row.cooldown_until = cooldown_until
        row.last_signal = last_signal
        session.commit()
    finally:
        if owns_session:
            session.close()
