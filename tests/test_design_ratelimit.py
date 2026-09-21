"""Rate-limit handling: wait-and-retry, chain failover, TPM pacing, request cap.

Standalone from test_design.py: defines its own pass fixtures and DB setup so
the rate-limit behavior can be tested with patched sleeps and a tiny fake
call_llm.
"""

from __future__ import annotations

import json

import pytest

from praxis.config import HardwareProfile
from praxis.design import (
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_MAX_WAIT_S,
    TPM_WINDOW_S,
    UNTRUSTED_END,
    UNTRUSTED_START,
    _TokenWindow,
    _wait_for_tpm_budget,
    cap_pass_prompt_chars,
    generate_design,
    parse_retry_after,
    resolve_design_model,
    resolve_design_model_chain,
)

GOOD_PASSES = {
    "technique": "## Technique\n\npaper says this [source 1].\n",
    "architecture": "## Architecture\n\n- C1 — does things (inference).\n",
    "data_contracts": "## Data Model & Contracts\n\n### Data model\n```text\nrow\n```\n",
    "plan": (
        "## Phased Implementation Plan\n\n"
        "### Phase 1: Slice\n"
        "- [ ] TASK-001 Do it (acceptance: works; test: test_it)\n"
        "**Tests:** pytest.\n"
    ),
    "hardware_fit": "## Hardware & Budget Fit\n\n| C | 1 GB | 1 | $0 |\n",
}


class _Candidate:
    def __init__(self, cid=1):
        self.id = cid
        self.source = "arxiv"
        self.url = "https://arxiv.org/abs/2401.12345"
        self.title = "CPU Fine-Tune"
        self.raw_text = "Fine-tune a small transformer on CPU."
        self.technique_summary = "LoRA fine-tuning on CPU"
        self.feasibility_score = 8


def _rate_limit_error(message="Error: 429 rate limit exceeded, try again in 12.4s"):
    """A litellm RateLimitError-shaped exception (status_code=429)."""
    exc = RuntimeError(message)
    exc.status_code = 429
    return exc


def _fake_answerer(state):
    """A call_llm fake returning canned pass content, or raising on demand."""

    def fake(prompt, system=None, model=None, **kwargs):
        state["prompts"].append(prompt)
        state["models"].append(model)
        if state.get("raise") is not None:
            exc = state["raise"]
            state["raise"] = None
            raise exc
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    return fake


@pytest.fixture
def design_setup(tmp_path, monkeypatch):
    """Temp DB + offline grounding + patched sleeps; yields (cid, profile, waits)."""
    import importlib

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    import praxis.design as design_module
    import praxis.grounding as grounding_module
    from praxis.db import Base, Candidate

    engine = create_engine(f"sqlite:///{tmp_path / 'rl.db'}")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    candidate = Candidate(
        source="arxiv",
        url="https://arxiv.org/abs/2401.12345",
        title="CPU Fine-Tune",
        raw_text="Fine-tune a small transformer on CPU.",
        status="analyzed",
    )
    session.add(candidate)
    session.commit()
    candidate_id = candidate.id
    session.close()

    def fresh_session():
        return Session(bind=engine, expire_on_commit=False)

    for name in ("praxis.db", "praxis.design", "praxis.discover", "praxis.design_io"):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "get_session", fresh_session)

    def fake_ground(candidate):
        return grounding_module.Grounding(chunks=[], provenance=[], source_kind="none")

    monkeypatch.setattr(grounding_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(design_module, "ground_candidate", fake_ground)

    waits: list[float] = []
    monkeypatch.setattr(design_module, "_sleep", lambda s: waits.append(s))
    return candidate_id, HardwareProfile(), waits


# ---------------------------------------------------------------------------
# retry-after hint parsing
# ---------------------------------------------------------------------------


def test_parse_retry_after_groq_hint():
    assert parse_retry_after("Rate limit reached. Please try again in 12.4s") == 12.4
    assert parse_retry_after("Retry-After: 30") == 30.0
    assert parse_retry_after("rate limit exceeded (retry after 5 seconds)") == 5.0
    assert parse_retry_after("no hint here") is None
    assert parse_retry_after("") is None


# ---------------------------------------------------------------------------
# wait-and-retry on a rate-limited pass
# ---------------------------------------------------------------------------


def test_rate_limited_pass_waits_and_retries_same_call(design_setup, monkeypatch, capsys):
    """A Groq TPM rejection sleeps the provider's hint, then retries the pass."""
    import praxis.design as design_module

    candidate_id, profile, waits = design_setup
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b")

    state = {"prompts": [], "models": [], "raise": None}
    fake = _fake_answerer(state)
    monkeypatch.setattr(design_module, "call_llm", fake)
    # First call (pass 1) is rate-limited with Groq's hint.
    state["raise"] = _rate_limit_error("Rate limit reached. Please try again in 12.4s")

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)

    assert result.status == "complete"
    assert waits == [12.4 + design_module.RATE_LIMIT_WAIT_MARGIN_S]
    out = capsys.readouterr().out
    assert "waiting 14s for rate limit (groq)" in out
    assert "pass 1/5" in out


def test_rate_limit_waits_are_bounded(design_setup, monkeypatch):
    """A huge retry hint is capped at RATE_LIMIT_MAX_WAIT_S."""
    import praxis.design as design_module

    candidate_id, profile, waits = design_setup
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b")

    state = {"prompts": [], "models": [], "raise": None}
    monkeypatch.setattr(design_module, "call_llm", _fake_answerer(state))
    state["raise"] = _rate_limit_error("try again in 3600s")

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)

    assert result.status == "complete"
    assert waits == [RATE_LIMIT_MAX_WAIT_S]


def test_rate_limit_fails_after_bounded_retries(design_setup, monkeypatch):
    """Persistent rate limiting exhausts retries, then the run fails resumably."""
    import praxis.design as design_module

    candidate_id, profile, waits = design_setup
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b")

    attempts = {"n": 0}

    def always_rate_limited(prompt, system=None, model=None, **kwargs):
        attempts["n"] += 1
        raise _rate_limit_error("try again in 5s")

    monkeypatch.setattr(design_module, "call_llm", always_rate_limited)

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)

    assert result.status == "failed"
    assert "try again" in (result.error or "")
    # Pass 1 made 1 + RATE_LIMIT_MAX_RETRIES attempts; a wait preceded each retry.
    assert attempts["n"] == 1 + RATE_LIMIT_MAX_RETRIES
    assert len(waits) == RATE_LIMIT_MAX_RETRIES


def test_chain_failover_moves_to_next_provider(design_setup, monkeypatch):
    """A rate-limited primary fails over to the second chain entry."""
    candidate_id, profile, waits = design_setup
    monkeypatch.setenv(
        "PRAXIS_DESIGN_MODEL",
        "groq/openai/gpt-oss-120b, cerebras/gpt-oss-120b, openrouter/openai/gpt-oss-120b",
    )

    attempts = {"groq": 0}
    models_succeeded = []

    def flaky(prompt, system=None, model=None, **kwargs):
        if model.startswith("groq"):
            attempts["groq"] += 1
            raise _rate_limit_error("try again in 10s")
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                models_succeeded.append(model)
                return content
        if "Review it against the defect classes" in prompt:
            models_succeeded.append(model)  # critic call also succeeded
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    import praxis.design as design_module

    monkeypatch.setattr(design_module, "call_llm", flaky)

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)

    assert result.status == "complete"
    # Each of the 6 calls (5 passes + critic) retries groq to exhaustion
    # (1 + RATE_LIMIT_MAX_RETRIES attempts) before failing over to cerebras.
    assert attempts["groq"] == 6 * (1 + RATE_LIMIT_MAX_RETRIES)
    assert len(waits) == 6 * RATE_LIMIT_MAX_RETRIES
    assert len(models_succeeded) == 6
    assert all(m.startswith("cerebras") for m in models_succeeded)


# ---------------------------------------------------------------------------
# PRAXIS_DESIGN_MODEL chain parsing
# ---------------------------------------------------------------------------


def test_resolve_design_model_chain_parsing(monkeypatch):
    """The chain env var splits on commas; explicit model replaces the chain."""
    monkeypatch.delenv("PRAXIS_DESIGN_MODEL", raising=False)
    assert resolve_design_model_chain(None) == ["groq/openai/gpt-oss-120b"]

    monkeypatch.setenv(
        "PRAXIS_DESIGN_MODEL",
        " groq/openai/gpt-oss-120b, cerebras/gpt-oss-120b , openrouter/openai/gpt-oss-120b ",
    )
    chain = resolve_design_model_chain(None)
    assert chain == [
        "groq/openai/gpt-oss-120b",
        "cerebras/gpt-oss-120b",
        "openrouter/openai/gpt-oss-120b",
    ]
    assert resolve_design_model(None) == "groq/openai/gpt-oss-120b"
    assert resolve_design_model_chain("explicit/model") == ["explicit/model"]


# ---------------------------------------------------------------------------
# proactive TPM pacing
# ---------------------------------------------------------------------------


def test_tpm_pacing_waits_when_window_is_full(monkeypatch):
    """When used + request exceeds the model's TPM limit, the call waits."""
    from praxis.config import FactsSheet, ModelLimits

    facts = FactsSheet(model_limits=[ModelLimits(model="groq/openai/gpt-oss-120b", tpm=8000)])
    window = _TokenWindow()
    monkeypatch.setattr("praxis.design._token_window", window)
    waits: list[float] = []
    monkeypatch.setattr("praxis.design._sleep", lambda s: waits.append(s))

    # 7000 tokens used in the window; a ~2.5k-token request would not fit.
    window.record("groq", 7000)
    _wait_for_tpm_budget("groq/openai/gpt-oss-120b", "x" * 1000, facts, output_tokens=1500)
    assert waits, "expected a pacing wait when the TPM window is full"
    assert waits[0] > 0


def test_tpm_pacing_skips_when_window_has_room(monkeypatch):
    """Under the limit: no wait, no sleep."""
    from praxis.config import FactsSheet, ModelLimits

    facts = FactsSheet(model_limits=[ModelLimits(model="groq/openai/gpt-oss-120b", tpm=8000)])
    window = _TokenWindow()
    monkeypatch.setattr("praxis.design._token_window", window)
    waits: list[float] = []
    monkeypatch.setattr("praxis.design._sleep", lambda s: waits.append(s))

    window.record("groq", 1000)
    _wait_for_tpm_budget("groq/openai/gpt-oss-120b", "x" * 400, facts, output_tokens=500)
    assert waits == []


def test_tpm_pacing_ignores_unknown_model(monkeypatch):
    """A model with no known limits is NOT throttled, however big the request."""
    from praxis.config import FactsSheet

    monkeypatch.setattr("praxis.design._token_window", _TokenWindow())
    waits: list[float] = []
    monkeypatch.setattr("praxis.design._sleep", lambda s: waits.append(s))

    _wait_for_tpm_budget("someprovider/m", "x" * 999999, FactsSheet(), output_tokens=100000)
    assert waits == []


def test_tpm_window_drops_stale_events():
    """Entries older than the 60s window no longer count against the limit."""
    window = _TokenWindow()
    window.record("groq", 5000, now=0.0)
    assert window.used("groq", now=TPM_WINDOW_S - 1) == 5000
    assert window.used("groq", now=TPM_WINDOW_S + 5) == 0


# ---------------------------------------------------------------------------
# per-model limits: itpm/otpm axes and max_tokens clamping (item 1)
# ---------------------------------------------------------------------------


def test_pacing_respects_itpm_input_axis(monkeypatch):
    """A qwen-style entry: input alone can bust ITPM even with tiny output."""
    from praxis.config import FactsSheet, ModelLimits

    facts = FactsSheet(
        model_limits=[ModelLimits(model="groq/qwen/qwen3.8-27b", itpm=7000, otpm=1000)]
    )
    window = _TokenWindow()
    monkeypatch.setattr("praxis.design._token_window", window)
    waits: list[float] = []
    monkeypatch.setattr("praxis.design._sleep", lambda s: waits.append(s))

    # 6500 input tokens used; a 1500-token input + 200 output breaks ITPM.
    window.record("groq", 6500)
    _wait_for_tpm_budget("groq/qwen/qwen3.8-27b", "x" * 6000, facts, output_tokens=200)
    assert waits, "ITPM breach must trigger a wait"


def test_pacing_respects_otpm_output_axis(monkeypatch):
    """Output alone busting OTPM triggers a wait even when input fits."""
    from praxis.config import FactsSheet, ModelLimits

    facts = FactsSheet(
        model_limits=[ModelLimits(model="groq/qwen/qwen3.8-27b", itpm=7000, otpm=1000)]
    )
    window = _TokenWindow()
    monkeypatch.setattr("praxis.design._token_window", window)
    waits: list[float] = []
    monkeypatch.setattr("praxis.design._sleep", lambda s: waits.append(s))

    # Output budget nearly used up: an 800-token output request breaches OTPM.
    window.record("groq", 0, 900)
    _wait_for_tpm_budget("groq/qwen/qwen3.8-27b", "tiny", facts, output_tokens=800)
    assert waits, "OTPM breach must trigger a wait"


def test_resolve_pass_output_tokens_clamps_to_otpm():
    """max_tokens for a pass is min(planned, otpm); no limits -> unclamped."""
    from praxis.config import FactsSheet, ModelLimits
    from praxis.design import resolve_pass_output_tokens

    facts = FactsSheet(
        model_limits=[ModelLimits(model="groq/qwen/qwen3.8-27b", otpm=1000)]
    )
    assert resolve_pass_output_tokens("groq/qwen/qwen3.8-27b", facts) == 1000
    assert resolve_pass_output_tokens("groq/qwen/qwen3.8-27b", facts, planned=500) == 500
    # No entry for this model: no clamping.
    assert resolve_pass_output_tokens("cerebras/qwen-3.8-27b", facts) > 1000
    assert resolve_pass_output_tokens("any/model", None) > 1000


def test_pass_call_carries_otpm_clamped_max_tokens(design_setup, monkeypatch):
    """The design calls pass max_tokens through to call_llm, clamped to OTPM."""
    import praxis.design as design_module

    candidate_id, profile, _waits = design_setup
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/qwen/qwen3.8-27b")
    captured: list[int | None] = []

    def fake_call_llm(prompt, system=None, model=None, max_tokens=None, **kwargs):
        captured.append(max_tokens)
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)
    from praxis.config import load_facts

    result = generate_design(
        _Candidate(candidate_id), profile, facts=load_facts(), pace_seconds=0.0
    )
    assert result.status == "complete"
    assert captured, "expected design calls"
    # Every call's max_tokens is clamped to qwen3.8-27b's otpm=1000 from the YAML.
    assert max(t for t in captured if t is not None) == 1000
    assert all(t is not None and t <= 1000 for t in captured)


# ---------------------------------------------------------------------------
# per-pass request-size cap
# ---------------------------------------------------------------------------


def test_cap_pass_prompt_trims_only_grounding_body():
    """The cap cuts inside the untrusted block; constraints and tail survive."""
    prompt = (
        "Design the technique below.\n\n"
        "HARD CONSTRAINTS (treat as absolute):\n<constraints>\n\n"
        f"{UNTRUSTED_START}\nSOURCE MATERIAL:\n{'paper ' * 20000}\n{UNTRUSTED_END}\n\n"
        "YOUR TASK:\nRespond with the markdown content of the 'Technique' "
        "section only (start with its '## Technique' heading)."
    )
    capped = cap_pass_prompt_chars(prompt, output_tokens=1000, token_cap=3500)

    assert len(capped) < len(prompt)
    assert "<constraints>" in capped  # constraints untouched
    assert "start with its '## Technique' heading" in capped  # tail untouched
    assert "[source material trimmed to fit the token budget]" in capped
    assert capped.count(UNTRUSTED_START) == 1 and capped.count(UNTRUSTED_END) == 1
    assert capped.index(UNTRUSTED_START) < capped.index(UNTRUSTED_END)  # framing intact


def test_cap_pass_prompt_keeps_small_prompts_intact():
    """Under the cap: identical prompt object."""
    assert cap_pass_prompt_chars("small prompt") == "small prompt"


def test_cap_pass_prompt_without_grounding_block_returns_unchanged():
    """No untrusted delimiters to trim: unchanged."""
    prompt = f"constraints + task only, {'x' * 40000}"
    assert cap_pass_prompt_chars(prompt) == prompt


# ---------------------------------------------------------------------------
# item 2: "request too large" is permanent — skip fast, shrink once, fail clear
# ---------------------------------------------------------------------------


def _too_large_error(message="Error code: 413 - {'error': {'message': 'request too large'}}"):
    """A Groq-shaped request-too-large exception (HTTP 413)."""
    exc = RuntimeError(message)
    exc.status_code = 413
    return exc


def test_is_request_too_large_error_classification():
    """413 status and the provider-specific wordings classify; rate limits do not."""
    from praxis.design import is_request_too_large_error

    assert is_request_too_large_error(_too_large_error())
    assert is_request_too_large_error(
        RuntimeError("number of tokens is larger than the limit: 65536")
    )
    assert is_request_too_large_error(RuntimeError("Please reduce the length of the messages"))
    # Not a size error: rate limits and network failures stay in their lanes.
    assert not is_request_too_large_error(_rate_limit_error("try again in 5s"))
    assert not is_request_too_large_error(RuntimeError("connection reset"))


def test_too_large_skips_entry_without_wait_or_cooldown(design_setup, monkeypatch):
    """The too-large entry is skipped instantly: no sleep, no pool cooldown."""
    import praxis.design as design_module

    candidate_id, profile, waits = design_setup
    monkeypatch.setenv(
        "PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b, cerebras/gpt-oss-120b"
    )

    attempts: list[str] = []

    def too_large_on_groq(prompt, system=None, model=None, **kwargs):
        attempts.append(model)
        if model.startswith("groq"):
            raise _too_large_error()
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", too_large_on_groq)

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)

    assert result.status == "complete"
    # Groq rejected each call exactly once (no wait-and-retry), then cerebras.
    assert attempts.count("groq/openai/gpt-oss-120b") == 0 or True  # counting below
    groq_attempts = sum(1 for m in attempts if m.startswith("groq"))
    cerebras_successes = sum(1 for m in attempts if m.startswith("cerebras"))
    assert groq_attempts == 6  # 5 passes + critic, one instant rejection each
    assert cerebras_successes == 6
    assert waits == []  # never slept
    # No cooldown was recorded for groq: a smaller request may still go there.
    from praxis.db import get_session as patched_get_session
    from praxis.db import ProviderHealth

    session = patched_get_session()
    rows = session.query(ProviderHealth).all()
    session.close()
    groq_rows = [r for r in rows if r.provider == "groq" and r.job == "pipeline"]
    assert all(r.state != "cooling_down" for r in groq_rows)


def test_too_large_on_every_entry_shrinks_once_then_fails_clearly(design_setup, monkeypatch, capsys):
    """Every entry rejects the size: one shrink, one more round, clear error."""
    import praxis.design as design_module
    import praxis.grounding as grounding_module

    candidate_id, profile, waits = design_setup
    monkeypatch.setenv(
        "PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b, cerebras/gpt-oss-120b"
    )
    # A large source block gives the shrink something elastic to cut.
    big_ground = grounding_module.Grounding(
        chunks=["paper text " * 4000], provenance=["arXiv HTML"], source_kind="arxiv"
    )
    monkeypatch.setattr(design_module, "ground_candidate", lambda _c: big_ground)

    rounds: list[int] = [0]

    def always_too_large(prompt, system=None, model=None, **kwargs):
        rounds[0] += 1
        raise _too_large_error()

    monkeypatch.setattr(design_module, "call_llm", always_too_large)

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)

    assert result.status == "failed"
    assert "request too large for every design model entry" in (result.error or "")
    assert "groq/openai/gpt-oss-120b" in (result.error or "")
    assert "cerebras/gpt-oss-120b" in (result.error or "")
    # 6 calls x (2 entries x 2 rounds) = 24 attempts, but pass 1 fails the run;
    # only the first call happens: 2 entries x 2 rounds = 4 attempts.
    # First call happens twice per entry (original + shrunk prompt).
    assert rounds[0] == 4
    out = capsys.readouterr().out
    assert "shrinking the source material" in out


def test_too_large_never_enters_rate_limit_wait(design_setup, monkeypatch):
    """A too-large error followed by success never produces a pacing sleep."""
    import praxis.design as design_module

    candidate_id, profile, waits = design_setup
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b")

    calls = {"n": 0}

    def too_large_then_success(prompt, system=None, model=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _too_large_error("request too large: reduce the size of the messages")
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", too_large_then_success)

    result = generate_design(_Candidate(candidate_id), profile, pace_seconds=0.0)
    # Single-entry chain: every entry blocked -> shrink once -> blocked again
    # -> clear failure (no rate-limit waits anywhere).
    assert result.status == "failed"
    assert waits == []
