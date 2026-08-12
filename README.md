# Praxis

A multi-agent system that watches arXiv / GitHub / Hacker News, extracts implementable techniques, produces hardware-calibrated engineering blueprints, and drafts prototypes via OpenCode.

Praxis runs a four-stage agent pipeline over a batch of research candidates, persists every stage's result to a single SQLite file, and keeps each prototype in an isolated scratch directory. All model inference happens over the API, so Praxis itself runs comfortably on an 8 GB CPU-only machine.

## Architecture

```
 arXiv / GitHub / HN
           |
           v
   +-------------+          Scout: fetch + dedupe candidates for a topic.
   |    Scout    |          (HTTP only; no LLM call here.)
   +-------------+
           |
           | candidate
           v
   +-------------+          LLM API (litellm)     Analyst: extract the single
   |   Analyst   | -----------------------------> implementable technique and
   +-------------+                                score feasibility 0-10
           |                                      against the hardware profile.
           | accepted analysis
           v
   +-------------+          LLM API (litellm)     Architect: emit a hardware-
   |  Architect  | -----------------------------> calibrated blueprint in
   +-------------+                                markdown, with a phased
           |                                      build plan.
           | blueprint
           v
   +-------------+          external subprocess   Coder: scoped to the first
   |    Coder    | -----------------------------> milestone only, runs
   +-------------+                                `opencode run --auto <phase>`
           |                                      in a fresh scratch directory.
           | prototype path
           v
   +---------------------+
   | SQLite              |
   | candidates ·        |
   | blueprints ·        |
   | prototype paths     |
   +---------------------+
```

Every stage reads and writes the same SQLite ledger, so a run is fully auditable. Only the Analyst and Architect call the LLM directly; the Coder delegates code generation to the OpenCode CLI as a separate subprocess rather than making an LLM call of its own.

## How it works

The pipeline is orchestrated in `praxis/pipeline.py` as Scout -> Analyst -> Architect -> Coder. Stages are wrapped in `run_with_retry` with exponential backoff, and each candidate is processed in isolation: a candidate that is rejected or fails at any stage is marked and skipped, and the batch continues.

- **Scout** — fetches items matching the topic from one of `arxiv`, `github`, or `hn`, deduplicates them, and persists promising ones as `Candidate` rows (`status="new"`).
- **Analyst** — sends each candidate's text plus the target `HardwareProfile` to the LLM, which extracts the core implementable technique and scores feasibility from 0-10. Candidates scoring below the threshold (default 4) or explicitly rejected are persisted as `rejected`; the rest move on.
- **Architect** — turns the accepted analysis into a `Blueprint`: a markdown engineering plan with modules, milestones, and a phased build plan, calibrated to the same hardware profile. The first phase of that plan is what the Coder will build.
- **Coder** — extracts the first milestone from the blueprint's phased build plan and hands it to the OpenCode CLI (`opencode run --auto`) running in a fresh `scratch/proto-<candidate_id>-<timestamp>/` directory. The resulting path is recorded on the blueprint; a non-zero exit or timeout is recorded as `prototype_failed` rather than crashing the run.

## Design decisions

Praxis is scoped deliberately. Each choice below is a judgment about what the system needs today, not a limitation deferred out of sight.

| Decision | Trade-off accepted | Rationale |
|---|---|---|
| **SQLite, not Neo4j/Postgres** | No graph queries, no concurrent writers | A single-machine research assistant needs zero-ops, portable storage; the data model is a simple pipeline ledger. SQLAlchemy already abstracts the engine, so swapping to Postgres is a configuration change if concurrent writes ever become necessary. |
| **No Temporal/Redis/worker queues** | No durable workflows, no parallelism | At batch sizes in the tens, a plain retry/backoff loop in `run_with_retry` is sufficient and far simpler to reason about. A real queue is added only if Praxis is run against a large scheduled backlog. |
| **Four agents, not six** | No Evaluator/Critic in v1 | The original plan called for six agents. Cutting evaluation to v2 let the core discovery-to-prototype loop ship and get tested first, instead of bolting speculative machinery onto an unproven core. |
| **Coder invokes OpenCode** | Praxis does not generate code itself | Code generation is treated as a distinct, swappable capability with its own agent loop, tooling, and iteration strategy. Isolating it in `_invoke_opencode` means the coding approach can evolve without touching the rest of the system. |

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Usage

All three commands are installed as the `praxis` entrypoint.

Run the full pipeline for a topic (defaults to `arxiv`, up to 20 candidates):

```bash
praxis run --source arxiv --topic "retrieval augmented generation" --limit 20
praxis run --source github --topic "local vector search on CPU"
```

`--limit` caps how many candidates the Scout keeps; `-v`/`--verbose` enables DEBUG logging. A run prints a per-batch summary:

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

Inspect the ledger by status:

```bash
praxis status
```

```
Candidate counts by status:
  analyzed: 2
  blueprinted: 2
  new: 14
  prototyped: 1
  prototype_failed: 1
  rejected: 3
```

Print a candidate's blueprint markdown:

```bash
praxis show 42
```

The Coder stage requires `opencode` on your PATH and authenticated (`opencode auth login`). The exact invocation lives in `praxis/agents/coder.py:_invoke_opencode`.

## Configuration

Praxis reads environment variables directly from the process (there is no bundled `.env` loader). `.env.example` is a reference template for the full set; export them in your shell or source them through your own dotenv tooling.

### Hardware profile

The `HardwareProfile` constrains feasibility scoring and blueprint generation. Fields resolve in order: **environment variable -> YAML file (`PRAXIS_CONFIG`) -> default**.

| Field | Type | Default | Env override |
|---|---|---|---|
| `cpu_only` | bool | `true` | `PRAXIS_CPU_ONLY` |
| `ram_gb` | int | `8` | `PRAXIS_RAM_GB` |
| `gpu` | bool | `false` | `PRAXIS_GPU` |
| `monthly_budget_usd` | float | `15.0` | `PRAXIS_MONTHLY_BUDGET_USD` |

Defaults live in `praxis/config.py`; the default YAML file is `hardware_profile.yaml`.

### Model and pipeline

| Variable | Purpose | Default |
|---|---|---|
| `PRAXIS_MODEL` | litellm model id used by the Analyst/Architect | `groq/llama-3.1-8b-instant` |
| `PRAXIS_FEASIBILITY_THRESHOLD` | minimum feasibility score (0-10) for a candidate to be accepted | `4` |
| `PRAXIS_DB_PATH` | SQLite file path | `./praxis.db` |
| `PRAXIS_DB_URL` | full SQLAlchemy URL; overrides `PRAXIS_DB_PATH` | — |
| `PRAXIS_CONFIG` | path to the hardware profile YAML | `./hardware_profile.yaml` |
| `PRAXIS_SCRATCH_ROOT` | where the Coder creates prototype directories | `./scratch` |
| `PRAXIS_CODER_TIMEOUT_S` | timeout for the OpenCode subprocess | `600` |

## Testing & CI

```bash
pytest        # 54 tests
ruff check .  # lint
```

CI (`.github/workflows/ci.yml`) installs the package with dev extras and runs `ruff check .` then `pytest` on both Python 3.11 and 3.12. Tests mock the LLM client, HTTP fetches, and the OpenCode subprocess, so the suite runs offline and deterministically.

## Roadmap

Praxis is a working v1, and these are the intentional next phases:

- **Coder guardrails** — structural isolation on the OpenCode invocation plus a circuit-breaker on its tool calls, so a runaway prototype draft cannot burn unbounded time or tokens.
- **Human-in-the-loop review gate** — an approval step between Architect and Coder so no code is generated until a person signs off on the plan.
- **Agent memory** — persist review decisions and outcomes and feed them back into future Analyst scoring, so the system learns which techniques are actually buildable on target hardware.
- **Cost/token observability** — per-run and per-candidate spend accounting against `monthly_budget_usd`.
- **Confidence-aware routing** — treat borderline feasibility scores (near the threshold) as a separate routing decision rather than a binary accept/reject.
- **Pipeline resumability** — allow a run to resume from the last completed stage instead of restarting from Scout.
- **Minimal frontend** — a thin read-only view over the ledger and prototypes; the CLI stays the source of truth.

## License

MIT, as declared in `pyproject.toml`. A `LICENSE` file should be added to the repo to match.
