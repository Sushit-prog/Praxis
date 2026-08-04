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
praxis run --source arxiv --topic "retrieval augmented generation" --limit 20
praxis status
praxis show 42
```

`praxis run` runs the full four-agent pipeline and prints a summary with the
candidate titles:

```
Summary for topic='retrieval augmented generation' source=arxiv
  discovered: 3
  analyzed: 2
  rejected: 1
  blueprinted: 2
  prototyped: 1
  failed: 1
Candidates:
  - Realtime RAG with an index cache [prototyped] (scratch/proto-7-20260805-120000)
  - Compact embeddings on CPU [rejected]
  - Local reranker [prototype_failed]
```

A candidate that is rejected or fails at any stage is skipped, and the batch
continues — one bad candidate never aborts the run.

`praxis status` shows how many candidates are in each state:

```
Candidate counts by status:
  analyzed: 2
  blueprinted: 2
  new: 14
  prototyped: 1
  prototype_failed: 1
  rejected: 3
```

`praxis show <id>` prints the full blueprint markdown for a candidate.

The Coder stage shells out to the OpenCode CLI, so `opencode` must be on your
PATH and authenticated (`opencode auth login`). The exact invocation lives in
`praxis/agents/coder.py:_invoke_opencode`.

## Why these tradeoffs

Praxis is scoped the way it is on purpose; each choice keeps the system honest
about the 8 GB CPU-only machine it targets.

- **SQLite, not Postgres/Neo4j.** The data model is a simple pipeline ledger:
  candidates, blueprints, prototypes. SQLite is zero-ops, runs anywhere, and
  keeps every run auditable in a single file. A graph database only pays off
  when you want multi-hop queries across many runs (e.g. "which blueprints
  share a technique") — a v2 concern, not a v1 blocker.
- **No Temporal/worker queues.** The pipeline is four strictly sequential
  stages with per-candidate isolation. `run_with_retry` plus exponential
  backoff covers transient LLM/API failures, and a batch continues past
  individual candidate failures. A durable workflow engine earns its weight
  once stages become long-running, parallel, or resumable mid-batch — none of
  which is true today.
- **Four agents, deliberately.** Scout → Analyst → Architect → Coder is the
  smallest loop that turns a research item into a working prototype.
  Evaluator/Critic agents were cut from v1 on purpose: automated evaluation
  needs a prototype-quality bar and metrics that don't exist yet, so adding
  them now would be speculative. When v2 adds evaluation, it slots in after
  Coder without changing the earlier stages.
- **Delegate code generation to OpenCode.** Praxis doesn't reimplement an
  agent loop for writing code; it hands the blueprint's first milestone to a
  purpose-built coding agent running in an isolated scratch directory. The
  invocation is isolated in `_invoke_opencode` so the exact CLI flags can
  evolve without touching the rest of the system.
- **Only the LLM calls are heavy.** All model inference happens remotely via
  litellm/API. Praxis itself stays small and CPU/RAM-light, matching the
  hardware it is calibrated for.

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
