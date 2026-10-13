"""Idea intake: turn a plain-text brief into a persisted candidate.

`praxis idea brief.md` validates a user-written brief (non-placeholder text,
at least one bullet under an MVP heading and one under an Out-of-scope
heading) and stores it as ``Candidate(source="idea")`` with the idempotency
key ``idea://<slug>-<hash8>`` over the normalized content: the same brief
run twice reuses the same candidate, edited content creates a new one.

The brief becomes the candidate's ``raw_text``; design passes receive it as
one untrusted delimited block via :func:`praxis.grounding.ground_candidate`.
"""

from __future__ import annotations

import hashlib
import re

from praxis.db import Candidate, get_session

SOURCE_IDEA = "idea"
IDEA_URL_PREFIX = "idea://"
# Status that only design/check/show accept: `praxis run --resume` (pipeline
# _unfinished_candidates), `praxis discover`, and `praxis review` all filter
# on other statuses, so idea candidates are never Analyst/Architect work.
IDEA_STATUS = "idea"

# Size caps: rejected before hashing so a giant file cannot become a candidate.
MAX_BRIEF_BYTES = 256_000  # file size on disk (read_brief)
MAX_BRIEF_CHARS = 100_000  # normalized length (validate_brief)

# Headings that mark the required sections (markdown headings only).
_MVP_HEADING_RE = re.compile(r"^#{1,6}[ \t]+.*\bMVP\b", re.IGNORECASE | re.MULTILINE)
_OUT_HEADING_RE = re.compile(
    r"^#{1,6}[ \t]+.*\bout[ \t-]+of[ \t-]+scope\b", re.IGNORECASE | re.MULTILINE
)
_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+\S")

# Placeholder text that must never become a candidate.
_PLACEHOLDER_RE = re.compile(
    r"\blorem ipsum\b|\bTODO\b|\bTBD\b|\bFIXME\b|\bplaceholder\b|\bfill[ \t-]me[ \t-]in\b",
    re.IGNORECASE,
)

_MIN_TOTAL_CHARS = 120
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class BriefError(ValueError):
    """A brief failed validation; message is user-facing."""


def normalize_brief(text: str) -> str:
    """Canonical form used for validation, hashing, and storage.

    Line endings collapse to ``\n``, trailing whitespace per line is
    dropped, and outer whitespace is stripped, so the same brief saved with
    different editors still maps to one candidate.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _section_bullets(text: str, heading_re: re.Pattern[str]) -> list[str]:
    """Non-empty bullet lines under the first heading matching ``heading_re``."""
    for match in heading_re.finditer(text):
        end = len(text)
        for heading in _HEADING_LINE_RE.finditer(text):
            if heading.start() >= match.end():
                end = heading.start()
                break
        bullets = [
            line.strip()
            for line in text[match.end() : end].splitlines()
            if _BULLET_RE.match(line)
        ]
        if bullets:
            return bullets
    return []


def title_from_brief(text: str) -> str:
    """First markdown heading's text, else first prose line, else 'idea brief'."""
    for match in _HEADING_LINE_RE.finditer(text):
        title = text[match.end() :].split("\n", 1)[0].strip()
        if title:
            return title[:100]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _HEADING_LINE_RE.match(stripped) or _BULLET_RE.match(stripped):
            continue
        return stripped[:100]
    return "idea brief"


def _slug(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")[:40].strip("-")
    return slug or "brief"


def brief_key(text: str) -> str:
    """Idempotency key ``idea://<slug>-<hash8>`` over the normalized text."""
    normalized = normalize_brief(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{IDEA_URL_PREFIX}{_slug(title_from_brief(normalized))}-{digest}"


def validate_brief(text: str) -> str:
    """Return the normalized brief, or raise :class:`BriefError`."""
    normalized = normalize_brief(text)
    if not normalized:
        raise BriefError("brief is empty")
    if len(normalized) > MAX_BRIEF_CHARS:
        raise BriefError(
            f"brief is too large ({len(normalized)} chars, limit {MAX_BRIEF_CHARS}); "
            "trim it to the MVP and out-of-scope essentials"
        )
    if len(normalized) < _MIN_TOTAL_CHARS:
        raise BriefError(
            f"brief is too thin ({len(normalized)} chars, need >= {_MIN_TOTAL_CHARS}); "
            "describe the MVP and what is out of scope"
        )
    placeholder = _PLACEHOLDER_RE.search(normalized)
    if placeholder:
        raise BriefError(
            f"brief contains placeholder text ({placeholder.group(0)!r}); write the real content"
        )
    mvp_bullets = _section_bullets(normalized, _MVP_HEADING_RE)
    if not mvp_bullets:
        raise BriefError(
            "brief needs at least one bullet under a heading containing 'MVP' "
            "(e.g. '## MVP')"
        )
    out_bullets = _section_bullets(normalized, _OUT_HEADING_RE)
    if not out_bullets:
        raise BriefError(
            "brief needs at least one bullet under a heading containing 'Out of scope' "
            "(e.g. '## Out of scope')"
        )
    return normalized


def intake_brief(text: str) -> tuple[Candidate, bool]:
    """Validate and persist a brief; returns (candidate, created).

    Idempotent on the normalized content: an identical brief returns the
    existing candidate with ``created=False``.
    """
    normalized = validate_brief(text)
    key = brief_key(normalized)
    session = get_session()
    try:
        existing = session.query(Candidate).filter(Candidate.url == key).first()
        if existing is not None:
            return existing, False
        candidate = Candidate(
            source=SOURCE_IDEA,
            url=key,
            title=title_from_brief(normalized),
            raw_text=normalized,
            status=IDEA_STATUS,
        )
        session.add(candidate)
        session.commit()
        return candidate, True
    finally:
        session.close()


def read_brief(path: str) -> str:
    """Read a brief file as UTF-8 (BOM tolerated), with size and format guards."""
    from pathlib import Path

    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError as exc:
        raise BriefError(f"no such brief file: {path}") from exc
    except OSError as exc:
        raise BriefError(f"cannot read brief file: {exc}") from exc
    if size > MAX_BRIEF_BYTES:
        raise BriefError(
            f"brief file is too large ({size} bytes, limit {MAX_BRIEF_BYTES})"
        )
    try:
        return p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BriefError(f"brief file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise BriefError(f"cannot read brief file: {exc}") from exc
