"""Scout: discover candidates from a source (arxiv/github/hn) and store new ones."""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from praxis.db import Candidate, get_session

logger = logging.getLogger(__name__)

TIMEOUT_S = 10
RETRY_ATTEMPTS = 3

ARXIV_API_URL = "https://export.arxiv.org/api/query"
GITHUB_API_URL = "https://api.github.com/search/repositories"
HN_API_URL = "https://hn.algolia.com/api/v1/search"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_RETRYABLE = (
    requests.Timeout,
    requests.ConnectionError,
    requests.HTTPError,
)


def _retry_decorator():
    return retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


@_retry_decorator()
def _get_json(
    url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> dict:
    """GET a URL and decode the JSON body, with retry/backoff."""
    resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


@_retry_decorator()
def _get_text(url: str, params: dict[str, Any] | None = None) -> str:
    """GET a URL and return the raw text body, with retry/backoff."""
    resp = requests.get(url, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.text


def _normalise(value: str | None) -> str:
    return " ".join((value or "").split())


def _fetch_arxiv(topic: str, limit: int) -> list[dict[str, str]]:
    """Fetch papers from the arXiv API sorted by submitted date (desc)."""
    params = {
        "search_query": f'ti:"{topic}" OR abs:"{topic}"',
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        xml_text = _get_text(ARXIV_API_URL, params=params)
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arxiv: malformed XML response: %s", exc)
        return []

    items: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        url = _normalise(entry.findtext("atom:id", namespaces=ATOM_NS))
        title = _normalise(entry.findtext("atom:title", namespaces=ATOM_NS))
        summary = _normalise(entry.findtext("atom:summary", namespaces=ATOM_NS))
        if not url or not title:
            continue
        items.append({"url": url, "title": title, "raw_text": summary})
    return items


def _fetch_github(topic: str, limit: int) -> list[dict[str, str]]:
    """Fetch repos from the GitHub search API sorted by stars (desc)."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "praxis",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"q": topic, "sort": "stars", "order": "desc", "per_page": limit}
    data = _get_json(GITHUB_API_URL, params=params, headers=headers)
    items = data.get("items") or []
    return [
        {
            "url": item.get("html_url") or "",
            "title": item.get("full_name") or item.get("name") or "",
            "raw_text": item.get("description") or "",
        }
        for item in items
        if item.get("html_url")
    ]


def _fetch_hn(topic: str, limit: int) -> list[dict[str, str]]:
    """Fetch stories from the HN Algolia API, sorted by points (desc)."""
    params = {"query": topic, "tags": "story", "hitsPerPage": limit}
    data = _get_json(HN_API_URL, params=params)
    hits = data.get("hits") or []
    items = []
    for hit in hits:
        object_id = hit.get("objectID")
        url = hit.get("url") or (
            f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""
        )
        title = _normalise(hit.get("title")) or "(untitled)"
        raw_text = _normalise(hit.get("story_text")) or title
        if not url:
            continue
        items.append(
            {
                "url": url,
                "title": title,
                "raw_text": raw_text,
                "points": int(hit.get("points") or 0),
            }
        )
    items.sort(key=lambda item: item["points"], reverse=True)
    for item in items:
        item.pop("points", None)
    return items


_FETCHERS = {
    "arxiv": "_fetch_arxiv",
    "github": "_fetch_github",
    "hn": "_fetch_hn",
}

VALID_SOURCES = frozenset(_FETCHERS)


def scout(source: str, topic: str, limit: int = 20) -> list[Candidate]:
    """Fetch candidates for a topic and insert new ones into the DB."""
    if source not in VALID_SOURCES:
        raise ValueError(f"unsupported source: {source!r}; expected one of {sorted(VALID_SOURCES)}")

    fetcher = globals()[_FETCHERS[source]]
    try:
        items = fetcher(topic, limit)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on fetch failure
        logger.warning("scout[%s]: fetch failed for topic %r: %s", source, topic, exc)
        return []

    session = get_session()
    try:
        existing = {url for (url,) in session.query(Candidate.url)}
        seen: set[str] = set()
        new_candidates: list[Candidate] = []
        for item in items:
            url = item["url"]
            if url in existing or url in seen:
                continue
            seen.add(url)
            candidate = Candidate(
                source=source,
                url=url,
                title=item["title"],
                raw_text=item.get("raw_text", ""),
                status="new",
            )
            session.add(candidate)
            new_candidates.append(candidate)
        session.commit()
        for candidate in new_candidates:
            session.refresh(candidate)
        return new_candidates
    finally:
        session.close()
