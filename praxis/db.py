"""SQLAlchemy models and SQLite engine setup."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
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
