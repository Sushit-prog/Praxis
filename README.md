# Praxis

A multi-agent system that watches **arXiv / GitHub / Hacker News**, extracts
implementable techniques, produces **hardware-calibrated engineering
blueprints**, and drafts prototypes.

The heavy lifting (LLM calls, code generation) happens remotely via API, so
Praxis runs fine on an 8 GB CPU-only machine.

## Pipeline

Praxis runs four agents sequentially, with retry/backoff between steps:

1. **Scout** — watches a source (`arxiv`, `github`, `hn`) for items matching a
   topic and stores promising ones as `Candidate`s.
2. **Analyst** — reads each candidate, extracts the implementable technique(s),
   and judges novelty and feasibility.
3. **Architect** — turns the analysis into a `Blueprint`: modules, milestones,
   feasibility score, and a prototype path calibrated to your `HardwareProfile`.
4. **Coder** — drafts a prototype from the blueprint's first phase by running
   the OpenCode CLI (`opencode run`) in a fresh scratch directory and stores
   the result path on the `Blueprint`.

> v1 status: all four agents are implemented — **Scout** (fetch + dedupe),
> **Analyst** (LLM technique extraction + feasibility scoring), **Architect**
> (hardware-calibrated `Blueprint` generation), and **Coder** (OpenCode CLI
> prototype drafting). Evaluator/Critic agents are explicitly deferred to v2.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

## Configure

Set `PRAXIS_MODEL` (default `groq/llama-3.1-8b-instant`) and any other vars in
a `.env` file or your shell. See `.env.example`:

| Variable | Purpose | Default |
|---|---|---|
| `PRAXIS_MODEL` | litellm model id | `groq/llama-3.1-8b-instant` |
| `PRAXIS_DB_PATH` | SQLite file | `./praxis.db` |
| `PRAXIS_DB_URL` | Full SQLAlchemy URL (overrides path) | — |
| `PRAXIS_CONFIG` | Hardware profile YAML | `./hardware_profile.yaml` |
| `PRAXIS_CPU_ONLY` / `PRAXIS_RAM_GB` / `PRAXIS_GPU` / `PRAXIS_MONTHLY_BUDGET_USD` | Hardware overrides | `true` / `8` / `false` / `15` |
| `PRAXIS_SCRATCH_ROOT` | Where the Coder creates prototype directories | `./scratch` |
| `PRAXIS_CODER_TIMEOUT_S` | OpenCode subprocess timeout | `600` |

The hardware profile (YAML or env) drives the Architect so blueprints match
your machine: `cpu_only`, `ram_gb`, `gpu`, `monthly_budget_usd`.

## Usage

```bash
praxis --help
praxis run --source arxiv --topic "..."      # 4-agent pipeline
```

The Coder stage shells out to the OpenCode CLI, so `opencode` must be on your
PATH and authenticated (`opencode auth login`). The exact invocation lives in
`praxis/agents/coder.py:_invoke_opencode`.

## Tests & lint

```bash
pytest
ruff check .
```

## Layout

```
praxis/
  config.py     # HardwareProfile + budget from YAML/env
  db.py         # SQLAlchemy models + SQLite engine
  llm.py        # litellm wrapper (mockable completion)
  agents/       # scout, analyst, architect, coder
  pipeline.py   # sequential orchestration with retry/backoff
  cli.py        # `praxis run --source ... --topic "..."`
tests/
  conftest.py   # fixtures incl. fake LLM client
```
