"""Tests for `praxis doctor`: env, provider key/endpoint checks, DB, coder status."""

from __future__ import annotations

import responses

from praxis.cli import main

DOTENV_KEYS = (
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEY",
    "PRAXIS_GROQ_API_KEY",
    "PRAXIS_OPENROUTER_API_KEY",
    "PRAXIS_CEREBRAS_API_KEY",
    "PRAXIS_MODEL",
    "PRAXIS_DESIGN_MODEL",
    "PRAXIS_CODER",
    "PRAXIS_DB_URL",
    "PRAXIS_DB_PATH",
)

MODELS_BODIES = {
    "https://api.groq.com/openai/v1/models": {
        "data": [
            {"id": "openai/gpt-oss-20b"},
            {"id": "openai/gpt-oss-120b"},
            {"id": "llama-3.3-70b-versatile"},
        ]
    },
    "https://openrouter.ai/api/v1/models": {"data": [{"id": "openai/gpt-oss-20b"}]},
    "https://api.cerebras.ai/v1/models": {"data": [{"id": "llama3.1-8b"}]},
}


def _isolate(monkeypatch, tmp_path, *, env_file: str | None):
    """Wipe provider/model env, point the DB at a temp file, and pin the .env check.

    Two leaks are guarded against: ``find_dotenv`` walks up the real
    filesystem (so the project's own .env would be found), and litellm loads
    the project .env at import time (so the machine's real keys would enter
    ``os.environ`` mid-test). Import praxis.llm first, then clean the env and
    control ``_load_env`` directly.
    """
    import praxis.llm  # noqa: F401 - trigger litellm's import-time .env load first

    for key in DOTENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{(tmp_path / 'doctor.db').as_posix()}")
    monkeypatch.chdir(tmp_path)
    if env_file is not None:
        (tmp_path / ".env").write_text(env_file, encoding="utf-8")
    env_path = str(tmp_path / ".env") if env_file is not None else None
    monkeypatch.setattr("praxis.cli._load_env", lambda: env_path)


def _mock_models_endpoints():
    for url, body in MODELS_BODIES.items():
        responses.add(responses.GET, url, json=body, status=200)


@responses.activate
def test_doctor_all_pass_with_temp_env(tmp_path, monkeypatch, capsys):
    """A temp .env + mocked endpoints: every check passes and the key is never printed."""
    secret = "gsk_super_secret_value_1234567890"
    _isolate(monkeypatch, tmp_path, env_file=f"GROQ_API_KEY={secret}\n")
    # The patched _load_env does not populate os.environ; emulate the loader.
    monkeypatch.setenv("GROQ_API_KEY", secret)
    monkeypatch.setenv("PRAXIS_MODEL", "groq/openai/gpt-oss-20b")
    _mock_models_endpoints()

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "all checks passed" in out
    assert "[ ok ] .env: found and loaded" in out
    assert f"provider groq: key present ({len(secret)} chars, never printed)" in out
    assert "authenticated ok" in out
    assert "configured model 'openai/gpt-oss-20b' is available" in out
    assert "database: schema creatable and writable" in out
    assert "PRAXIS_CODER: off (default)" in out
    assert secret not in out  # the key value is never printed
    assert "[ -- ] provider openrouter: no key configured" in out


@responses.activate
def test_doctor_fails_on_rejected_key(tmp_path, monkeypatch, capsys):
    """A 401 from the models endpoint fails the run with a one-line fix hint."""
    _isolate(monkeypatch, tmp_path, env_file="GROQ_API_KEY=gsk_wrong_key\n")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_wrong_key")
    responses.add(
        responses.GET,
        "https://api.groq.com/openai/v1/models",
        json={"error": "invalid"},
        status=401,
    )

    rc = main(["doctor"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL] provider groq models endpoint: key rejected (HTTP 401)" in captured.out
    assert "fix: check the key value for this provider" in captured.out
    assert "some checks failed" in captured.err


def test_doctor_fails_without_any_key(tmp_path, monkeypatch, capsys):
    """No provider key configured is a failed check with a hint."""
    _isolate(monkeypatch, tmp_path, env_file=None)

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL] provider keys: no provider key configured" in out
    assert "fix: add GROQ_API_KEY=" in out


@responses.activate
def test_doctor_network_error_hint(tmp_path, monkeypatch, capsys):
    """An unreachable provider gets the network hint, not the auth hint."""
    _isolate(monkeypatch, tmp_path, env_file=None)
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk_ok")
    responses.add(
        responses.GET,
        "https://api.cerebras.ai/v1/models",
        body=responses.ConnectionError("DNS failure"),
    )

    rc = main(["doctor"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "could not reach the provider (ConnectionError)" in captured.out
    assert "fix: could not reach the provider: check network/VPN/proxy" in captured.out
    assert "check the key value" not in captured.out


def test_doctor_reports_missing_env(tmp_path, monkeypatch, capsys):
    """No .env file and nothing exported: the .env check fails with a hint."""
    _isolate(monkeypatch, tmp_path, env_file=None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_present")

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL] .env: not found" in out
    assert "fix: copy .env.example to .env" in out


@responses.activate
def test_doctor_flags_model_missing_from_provider(tmp_path, monkeypatch, capsys):
    """The configured PRAXIS_MODEL must exist in the owning provider's list."""
    _isolate(monkeypatch, tmp_path, env_file=None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_ok")
    monkeypatch.setenv("PRAXIS_MODEL", "groq/gone-model-9000")
    responses.add(
        responses.GET,
        "https://api.groq.com/openai/v1/models",
        json={"data": [{"id": "openai/gpt-oss-20b"}]},
        status=200,
    )

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "configured model 'gone-model-9000' is not in the provider's list" in out
    assert "fix: set PRAXIS_MODEL" in out


def test_doctor_reports_coder_status(tmp_path, monkeypatch, capsys):
    """Invalid PRAXIS_CODER values fail; `off` reports the default terminal state."""
    _isolate(monkeypatch, tmp_path, env_file=None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_present")
    monkeypatch.setenv("PRAXIS_CODER", "nonsense")

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL] PRAXIS_CODER (PRAXIS_CODER='nonsense'): invalid value" in out
    assert "fix: set PRAXIS_CODER to off or opencode" in out

    monkeypatch.setenv("PRAXIS_CODER", "off")
    main(["doctor"])
    assert "PRAXIS_CODER: off (default)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# PRAXIS_DESIGN_MODEL chain verification
# ---------------------------------------------------------------------------


@responses.activate
def test_doctor_warns_when_chain_entry_missing(tmp_path, monkeypatch, capsys):
    """A design-chain entry the provider does not serve fails with a fix hint."""
    _isolate(monkeypatch, tmp_path, env_file="GROQ_API_KEY=gsk_k\n")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_k")
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/nonexistent-model")
    _mock_models_endpoints()

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL] design chain groq: groq/nonexistent-model does not exist at groq" in out
    assert "fix: fix or remove the entry in PRAXIS_DESIGN_MODEL" in out


@responses.activate
def test_doctor_chain_entry_available(tmp_path, monkeypatch, capsys):
    """Every chain entry present at its provider keeps doctor green."""
    _isolate(monkeypatch, tmp_path, env_file="GROQ_API_KEY=gsk_k\n")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_k")
    monkeypatch.setenv(
        "PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b"
    )
    _mock_models_endpoints()

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "design chain groq: groq/openai/gpt-oss-20b is available" in out
    assert "design chain groq: groq/openai/gpt-oss-120b is available" in out


@responses.activate
def test_doctor_chain_entry_without_key_is_skipped(tmp_path, monkeypatch, capsys):
    """A chain entry at a provider with no key is skipped, not a failure."""
    _isolate(monkeypatch, tmp_path, env_file="GROQ_API_KEY=gsk_k\n")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_k")
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "cerebras/llama3.1-8b")
    _mock_models_endpoints()

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "design chain cerebras" in out
    assert "no CEREBRAS_API_KEY configured" in out


# ---------------------------------------------------------------------------
# --deep: live 1-token probes per design-chain entry
# ---------------------------------------------------------------------------


def _deep_isolation(monkeypatch, tmp_path):
    """Isolate a deep doctor run: temp DB, fake key, no .env file."""
    import importlib

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from praxis.db import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'doctor.db'}")
    Base.metadata.create_all(engine)

    def fresh_session():
        return Session(bind=engine, expire_on_commit=False)

    for name in ("praxis.db", "praxis.providers", "praxis.llm", "praxis.design"):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "get_session", fresh_session)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b")
    # .env (auto-loaded by litellm's import) leaks real keys; scrub them so
    # key-presence reflects only what this test sets.
    for leaked in (
        "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY",
        "PRAXIS_OPENROUTER_API_KEY",
        "PRAXIS_CEREBRAS_API_KEY",
        "PRAXIS_GROQ_API_KEY",
    ):
        monkeypatch.delenv(leaked, raising=False)
    # _check_env probes the filesystem: a missing .env fails that check, which
    # is fine for these tests (we assert on the probe lines, not on .env).


def test_doctor_deep_probes_reported_per_chain_entry(monkeypatch, tmp_path):
    """--deep sends one probe per chain entry and reports failures with hints."""
    import praxis.doctor as doctor_module

    _deep_isolation(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b, cerebras/qwen-3.8-27b"
    )
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk_test")

    probed = []

    def fake_completion(**kwargs):
        probed.append(kwargs["model"])
        if kwargs["model"].startswith("cerebras"):
            exc = RuntimeError("402: payment required")
            exc.status_code = 402
            raise exc
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(doctor_module, "_completion_for_probe", lambda: fake_completion)

    checks = doctor_module.run_doctor_checks(deep=True)
    probe_checks = [c for c in checks if c.name.startswith("design chain probe")]
    assert [c.name for c in probe_checks] == [
        "design chain probe groq",
        "design chain probe cerebras",
    ]
    assert probed == ["groq/openai/gpt-oss-120b", "cerebras/qwen-3.8-27b"]
    groq_check = probe_checks[0]
    assert groq_check.ok is True
    cerebras_check = probe_checks[1]
    assert cerebras_check.ok is False
    assert "payment required" in cerebras_check.detail
    assert cerebras_check.hint  # fix hint present


def test_doctor_plain_mode_skips_probes(monkeypatch, tmp_path):
    """Without --deep no probe line appears (behaviour unchanged)."""
    import praxis.doctor as doctor_module

    _deep_isolation(monkeypatch, tmp_path)
    sent = {"n": 0}

    def fake_completion(**kwargs):
        sent["n"] += 1
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(doctor_module, "_completion_for_probe", lambda: fake_completion)
    monkeypatch.setattr(
        doctor_module,
        "_fetch_model_ids",
        lambda p, k: doctor_module.ModelsFetchResult(model_ids=["openai/gpt-oss-120b"]),
    )

    checks = doctor_module.run_doctor_checks(deep=False)
    assert sent["n"] == 0
    assert not [c for c in checks if c.name.startswith("design chain probe")]


def test_doctor_deep_probe_skips_entry_without_key(monkeypatch, tmp_path):
    """An entry whose provider has no key is skipped, not probed."""
    import praxis.doctor as doctor_module

    _deep_isolation(monkeypatch, tmp_path)
    # A provider no key exists for (set or leaked from .env).
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "unknownprov/some-model")
    monkeypatch.delenv("UNKNOWNPROV_API_KEY", raising=False)
    monkeypatch.delenv("PRAXIS_UNKNOWNPROV_API_KEY", raising=False)

    def fake_completion(**kwargs):
        raise AssertionError("no probe should be sent without a key")

    monkeypatch.setattr(doctor_module, "_completion_for_probe", lambda: fake_completion)

    checks = doctor_module.run_doctor_checks(deep=True)
    probe_checks = [c for c in checks if c.name.startswith("design chain probe")]
    assert len(probe_checks) == 1
    assert probe_checks[0].skipped is True
