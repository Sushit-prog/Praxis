"""Scout agent tests: fetchers (mocked HTTP), retry, dedupe, graceful failures."""

from __future__ import annotations

import importlib

import responses

from praxis.agents.scout import (
    _fetch_arxiv,
    _fetch_github,
    _fetch_hn,
    _get_json,
    scout,
)
from praxis.db import Candidate

scout_module = importlib.import_module("praxis.agents.scout")

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001</id>
    <title> Attention Is All You Need </title>
    <summary> We propose the Transformer, a model
      based solely on attention. </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002</id>
    <title> A Second Paper </title>
    <summary> Another abstract. </summary>
  </entry>
</feed>
"""


@responses.activate
def test_fetch_arxiv_parses_atom():
    responses.add(
        responses.GET,
        "https://export.arxiv.org/api/query",
        body=ARXIV_XML,
        content_type="application/atom+xml",
    )
    items = _fetch_arxiv("attention", 2)

    assert len(items) == 2
    assert items[0]["url"] == "http://arxiv.org/abs/2401.00001"
    assert items[0]["title"] == "Attention Is All You Need"
    assert "based solely on attention" in items[0]["raw_text"]
    assert "sortBy=submittedDate" in responses.calls[0].request.url


@responses.activate
def test_fetch_arxiv_malformed_xml_returns_empty(caplog):
    responses.add(
        responses.GET,
        "https://export.arxiv.org/api/query",
        body="<feed><broken>",
        content_type="application/atom+xml",
    )
    with caplog.at_level("WARNING"):
        items = _fetch_arxiv("attention", 2)

    assert items == []
    assert "malformed XML" in caplog.text


@responses.activate
def test_fetch_github_parses_and_sends_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    responses.add(
        responses.GET,
        "https://api.github.com/search/repositories",
        json={
            "items": [
                {
                    "html_url": "https://github.com/foo/bar",
                    "full_name": "foo/bar",
                    "description": "A cool repo",
                }
            ]
        },
        status=200,
    )
    items = _fetch_github("cool", 1)

    assert items == [
        {"url": "https://github.com/foo/bar", "title": "foo/bar", "raw_text": "A cool repo"}
    ]
    assert responses.calls[0].request.headers["Authorization"] == "Bearer secret-token"


@responses.activate
def test_fetch_github_omits_token_when_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    responses.add(
        responses.GET,
        "https://api.github.com/search/repositories",
        json={"items": []},
        status=200,
    )
    _fetch_github("cool", 1)

    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_fetch_github_missing_items_returns_empty(caplog):
    responses.add(
        responses.GET,
        "https://api.github.com/search/repositories",
        json={"message": "rate limited"},
        status=200,
    )
    with caplog.at_level("WARNING"):
        items = _fetch_github("cool", 1)

    assert items == []


@responses.activate
def test_fetch_hn_parses_and_sorts_by_points():
    responses.add(
        responses.GET,
        "https://hn.algolia.com/api/v1/search",
        json={
            "hits": [
                {
                    "objectID": "1",
                    "title": "Low points",
                    "url": "https://example.com/low",
                    "points": 5,
                    "story_text": "text A",
                },
                {
                    "objectID": "2",
                    "title": "High points",
                    "url": None,
                    "points": 99,
                },
            ]
        },
        status=200,
    )
    items = _fetch_hn("topic", 2)

    assert [item["title"] for item in items] == ["High points", "Low points"]
    assert items[0]["url"] == "https://news.ycombinator.com/item?id=2"
    assert items[0]["raw_text"] == "High points"


@responses.activate
def test_get_json_retries_on_transient_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    responses.add(responses.GET, "https://example.com/data", json={}, status=500)
    responses.add(responses.GET, "https://example.com/data", json={"ok": True}, status=200)

    result = _get_json("https://example.com/data")

    assert result == {"ok": True}
    assert len(responses.calls) == 2


def test_scout_inserts_and_dedupes(db_session, monkeypatch):
    session = db_session
    session.add(
        Candidate(
            source="arxiv",
            url="http://arxiv.org/abs/old",
            title="Existing",
            raw_text="",
            status="new",
        )
    )
    session.commit()

    fetched = [
        {"url": "http://arxiv.org/abs/old", "title": "Existing", "raw_text": ""},
        {"url": "http://arxiv.org/abs/new", "title": "New", "raw_text": "body"},
        {"url": "http://arxiv.org/abs/new", "title": "New dup", "raw_text": "body"},
    ]
    monkeypatch.setattr(scout_module, "_fetch_arxiv", lambda topic, limit: fetched)

    result = scout("arxiv", "attention")

    assert len(result) == 1
    assert result[0].url == "http://arxiv.org/abs/new"
    assert result[0].status == "new"
    assert session.query(Candidate).count() == 2


def test_scout_graceful_when_fetch_fails(db_session, monkeypatch, caplog):
    def boom(topic, limit):
        raise ConnectionError("network down")

    monkeypatch.setattr(scout_module, "_fetch_arxiv", boom)
    with caplog.at_level("WARNING"):
        result = scout("arxiv", "attention")

    assert result == []
    assert "fetch failed" in caplog.text
    assert session_count(db_session) == 0


def test_scout_invalid_source_raises(db_session):
    import pytest

    with pytest.raises(ValueError, match="unsupported source"):
        scout("not-a-source", "topic")


def session_count(session) -> int:
    return session.query(Candidate).count()
