"""SQLAlchemy models and SQLite engine setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
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


@dataclass
class UsageSummary:
    """Full usage picture: totals, a recent window, and breakdowns."""

    totals: UsageTotals
    recent: UsageTotals
    by_stage: dict[str | None, UsageTotals]
    by_model: dict[str, UsageTotals]
    days: int


def usage_totals(*, session=None, since: datetime | None = None) -> UsageTotals:
    """Aggregate token/cost totals over recorded LLM calls, optionally since a date."""
    owns_session = session is None
    session = session or get_session()
    try:
        stmt = select(
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.prompt_tokens), 0),
            func.coalesce(func.sum(LLMUsage.completion_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
        )
        if since is not None:
            stmt = stmt.where(LLMUsage.created_at >= since)
        count, prompt, completion, total, cost = session.execute(stmt).one()
        return UsageTotals(
            calls=count,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=float(cost),
        )
    finally:
        if owns_session:
            session.close()


def _usage_grouped(session, column) -> dict[str | None, UsageTotals]:
    """Group usage totals by a column (stage or model)."""
    rows = session.execute(
        select(
            column,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.prompt_tokens), 0),
            func.coalesce(func.sum(LLMUsage.completion_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
        ).group_by(column)
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
