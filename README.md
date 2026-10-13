# Praxis

Praxis turns a research paper or repo into a hardware-calibrated implementation spec, verifies it with deterministic zero-LLM checks, and exports it for a coding agent to build.

[![CI](https://github.com/Sushit-prog/Praxis/actions/workflows/ci.yml/badge.svg)](https://github.com/Sushit-prog/Praxis/actions/workflows/ci.yml)

## What it does

```
Scout
  -> Analyst (feasibility gate, scores 0-10)
  -> Architect (blueprint)
  -> design engine (5 paced LLM passes: technique, architecture,
     data_contracts, plan, hardware_fit)
  -> praxis plan (docs pack: 6 LLM + 5 deterministic + optional Recommendations)
  -> praxis check (deterministic zero-LLM consistency rules)
  -> praxis handoff (allowlisted copy + secret scan + git init)
  -> coding agent builds from the exported pack
```

The Coder stage (`praxis run --prototype`) is optional and off by default. Everything after the design engine — `check`, `handoff`, `specs`, `doctor` — runs without a provider key.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

praxis doctor        # no key required — checks env, keys, DB, coder
praxis specs         # no key required — shows detected host + design constraints
praxis discover --topic "retrieval augmented generation" --limit 5   # needs a key
praxis design <id>   # needs a key
praxis plan <id>     # needs a key
praxis check <id>    # no key required
praxis handoff <id> ../MyProject   # no key required
```

`discover`, `design`, and `plan` call the LLM and fail fast with a clear message when no provider key is set. `check`, `handoff`, `specs`, and `doctor` are fully keyless.

## Sample output

`praxis check 5` (the MemGuard design pack) — seven warnings (five from the older single-pass Architect blueprint), two info, exit 0:

```
[WARN ] r10-stack-fit blueprint: ram-claim: states an 8 GB bound; usable headroom is 4 GB
[WARN ] r10-stack-fit blueprint: remote-host: unit prescribes a rented/remote host but the facts sheet is local-only (local_only=True)
[WARN ] r10-stack-fit blueprint: cuda-only: unit mentions CUDA tooling but the facts sheet is CPU-only (cpu_only=True)
[WARN ] r10-stack-fit blueprint: remote-host: unit prescribes a rented/remote host but the facts sheet is local-only (local_only=True)
[WARN ] r10-stack-fit blueprint: cuda-only: unit mentions CUDA tooling but the facts sheet is CPU-only (cpu_only=True)
[WARN ] r3-dependency plan: dependency conflict: pass 'security' requires 'httpx' (http client), pass 'plan' requires 'requests'; the two passes never mention each other's package
[WARN ] r5-metric prd: metric conflict: pass 'prd' prescribes 'cosine' for vector retrieval, but pass 'data_contracts' prescribes 'euclidean'
[INFO ] r8-prd-sections prd: prd sections: legacy pack: missing MVP (v0.2 packs require it)
[INFO ] r13-rules-categories rules: rules categories: legacy pack: missing General, Before coding, Security, Testing, Git (v0.2 packs require it)
0 error(s), 7 warning(s), 2 info
checked against: usable RAM 4 GB, cpu_only=True, local_only=True
```

The five `r10-stack-fit` warnings come from the **older single-pass Architect blueprint** for candidate 5 (a pre-v0.2 blueprint that prescribes a VPS and bitsandbytes on a CPU-only, local-only facts sheet); its eleven stored passes score zero. Facts come from `hardware_profile.yaml` (env vars and an optional host probe can override the snapshot), so every r10 verdict is explainable against the printed line. `praxis check 5 --json` returns the same findings as a JSON list with `rule_id`, `severity`, `pass_id`, `excerpt`, and `message` fields.

## Why the checker exists

An LLM-written decision record in the MemGuard design pack said "2k records × 3 MB ≈ 6 MB" when the correct product is ~6 GB. Because every downstream document (PRD, ARCHITECTURE, TASKS, TEST_PLAN, SECURITY, DECISIONS) was rendered from that single stored pass, the same arithmetic error landed in six files. `praxis check` was built to catch exactly this class of error — arithmetic mismatches, comparison-direction bugs, cross-pass dependency conflicts, and retrieval-metric conflicts — with deterministic rules that make zero LLM calls. It later caught real defects in its own MemGuard pack (the requests/httpx and Euclidean/cosine contradictions shown above), which were fixed at the stored-pass source.

## Rules

| Rule | Severity | What it detects |
|---|---|---|
| `r1-arithmetic` | error | Multiplication and basis-in-paren claims that do not add up (e.g. "2k × 3 MB ≈ 6 MB") |
| `r2-direction` | warning | Free-storage gates written `<=` instead of `>=` |
| `r4-leaving` | error | "Leaving N MB of the M MB budget" where used + left ≠ cap, or left < footprint |
| `r4-cap` | error | Component size written as a percentage or fraction of a cap that does not match the stated cap |
| `r3-dependency` | warning | Two passes prescribing different packages for one slot (e.g. `requests` in TASKS vs `httpx` in SECURITY) |
| `r5-metric` | warning | Two passes prescribing different retrieval metrics (e.g. Euclidean vs cosine) |
| `r7-recommendation-claims` | warning | The optional Recommendations pass prescribing GPU/VRAM or rented hosts, or citing GB/$ numbers absent from the fit table and facts |
| `r8-prd-sections` | warning (info on legacy packs) | PRD missing required sections (Problem, Users, Goal, Features, MVP, Out of scope, Success criteria) |
| `r9-task-structure` | warning (info on legacy packs) | `TASK-NNN` tasks in the plan without an Acceptance marker within 3 lines; stamped packs with no task ids at all |
| `r10-stack-fit` | warning | Prescribed models, RAM/VRAM footprints, CUDA tooling, or rented hosts that do not fit the facts sheet (stored passes + blueprint) |
| `r13-rules-categories` | warning (info on legacy packs) | RULES missing required categories (General, Before coding, Security, Testing, Git) |
| `registry-unknown` | warning | A pass key not in the known-pass registry |
| `registry-missing` | info | A known pass key absent from the stored design |

Errors fail the command (exit 1). Warnings and info are reported but exit 0. Errors block `praxis handoff` unless `--skip-check` is passed.

## praxis handoff

`praxis handoff <id> <target>` exports a finished design's docs pack into a fresh project directory and bootstraps it for a coding agent. It makes **zero LLM calls**.

**Copies (allowlist of 12 names; RECOMMENDATIONS.md only when present):** `.env.example`, `ARCHITECTURE.md`, `DECISIONS.md`, `DESIGN.md`, `MEMORY.md`, `PRD.md`, `README.md`, `RECOMMENDATIONS.md`, `RULES.md`, `SECURITY.md`, `TASKS.md`, `TEST_PLAN.md`.

**Refuses:**
- A non-empty target (unless `--force`, which overwrite-merges and never deletes extras).
- A target inside the Praxis repo, or a repo ancestor of the target (symlink- and case-insensitive-resolved).
- Stray files in `docs/` that are not on the allowlist (unless `--force`, which skips them rather than copying).
- Key-like strings in the copied content (e.g. `gsk_…`, `sk-…`, `Bearer …`); documented placeholders like `…_placeholder` will be ignored. On a hit it aborts and cleans up; the matched value is never printed.

After a clean copy it runs `git init` and writes `HANDOFF_PROMPT.md`, a plan-mode prompt that tells the agent to read the pack, obey RULES.md, propose a milestone-1 plan listing every ambiguity with `file:line`, and wait for approval before writing code.

## Worked example: MemGuard

MemGuard is a project built from a Praxis-generated pack by a coding agent. The design pack (`designs/005-memguard-persisting-verifier-signals-for-llm-age/docs/`) specifies a system that persists verifier signals (reward, confidence, label, uncertainty) for LLM agents and serves them as a local k-NN memory store. Phase 1 is the minimal end-to-end slice: an append-only SQLite store, a sentence-transformer embedding path, cosine k-NN retrieval, and a `run_phase1` entry point. It passed 11/11 tests in ~25 s on a CPU-only 8 GB machine. Phases 2-4 (governance admission, lifecycle archival, benchmark harness) are designed but not built.

[MemGuard repo link]

## Providers and configuration

The effective design/plan chain (probed Oct 10, 2026) is:

```
nvidia_nim/nvidia/nemotron-3-super-120b-a12b
  -> groq/openai/gpt-oss-120b
  -> openrouter/nvidia/nemotron-3.5-lightning:free
```

Job A (Analyst, Architect, design passes) uses three free providers directly via litellm: NVIDIA NIM, Groq, and OpenRouter. OpenRouter's free tier is capped at **50 requests/day** (counted only on successes), so the OpenRouter entry is failover insurance, not a workhorse. Cerebras remains configured but unused — its billing endpoint returns "payment required".

Job B (Coder, optional) routes through a local omniroute gateway that owns its own keys; opencode holds no direct provider keys.

**Environment variable names** (never values):

| Variable | Purpose |
|---|---|
| `NVIDIA_NIM_API_KEY` | Job A provider key |
| `GROQ_API_KEY` | Job A provider key |
| `OPENROUTER_API_KEY` | Job A provider key |
| `CEREBRAS_API_KEY` | Optional; billing dead |
| `PRAXIS_NVIDIA_NIM_API_KEY` etc. | Per-provider overrides |
| `PRAXIS_MODEL` | Analyst/Architect model (default `groq/openai/gpt-oss-20b`) |
| `PRAXIS_DESIGN_MODEL` | Design/plan chain, comma-separated |
| `PRAXIS_CODER` | `off` (default) or `opencode` |
| `PRAXIS_DB_PATH` / `PRAXIS_DB_URL` | SQLite location |
| `PRAXIS_CONFIG` | Hardware profile YAML path |

Per-model free-tier rate limits (`tpm`/`itpm`/`otpm`, plus optional `max_output`) are declared in `hardware_profile.yaml` under `model_limits`. A model with no entry is not throttled. Seeded values are labelled "observed on the free tier, Sep 2026, may change".

## Constraints and cost

Praxis runs CPU-only on an 8 GB RAM laptop (Intel Core, ~4 GB usable headroom, no CUDA). All model inference happens over the API; the local machine never runs a model. The design engine clamps every call to input + output ≤ ~5000 tokens.

Total LLM spend recorded in the ledger for the MemGuard design (84 calls, 53 from cache, 179,215 tokens): **$0.0786**. This is the cost of one full design + critic + plan run; it is not a per-project estimate.

## Honest limitations

- **Spec generation is the default.** The Coder stage is optional and experimental; it is off by default. `praxis export <id>` and `praxis handoff` are the supported paths for getting a design into a coding agent.
- **Free-tier providers.** The chain depends on NVIDIA NIM, Groq, and OpenRouter free tiers. Rate limits and model availability change; re-probe with `praxis doctor --deep` before relying on any entry.
- **No LLM-as-judge in eval.** The eval harness uses a deterministic rubric (required sections, hardware scoping, no GPU on CPU-only profiles, RAM within budget). Scoring blueprint quality via LLM-as-judge is future work.
- **No web view yet.** The CLI is the sole interface.
- **Renders are not byte-identical across reruns.** Every render includes live host facts (storage free, usable RAM), so a rerun on the same machine can differ in the volatile readings.
- **r3 and r5 are heuristic warnings, not proofs.** They catch common cross-pass conflicts (package slots, retrieval metrics) but do not prove consistency.
- **MemGuard Phase 2+ is not built.** Only Phase 1 (the end-to-end slice) exists; governance admission, lifecycle archival, and the benchmark harness are designed but not implemented.

## Roadmap

- Web view over the ledger and prototypes
- LLM-as-judge scoring in the eval harness
- Deterministic render output (pin volatile host facts)
- More check rules (additional cross-pass conflict classes)
- `praxis handoff` for non-Praxis packs

## Command reference

All commands are installed as the `praxis` entrypoint.

```bash
praxis run --source arxiv --topic "retrieval augmented generation" --limit 20
praxis run --source github --topic "local vector search on CPU"
praxis run --source arxiv --topic "..." --prototype   # opt into the Coder stage

praxis status          # candidate counts by status
praxis show 42         # print a blueprint's markdown
praxis usage           # LLM token usage and estimated spend
praxis usage --days 7

praxis export 42                    # writes ./build-kit-42.md
praxis export 42 --out my-kit.md

praxis eval                     # uses tests/fixtures/golden_candidates.json
praxis eval --golden my_set.json --threshold 5

praxis specs         # detected CPU/RAM/GPU + effective design constraints
praxis specs --json

praxis review                       # list borderline candidates
praxis review approve 42
praxis review reject 42
praxis memory        # recent human review decisions

praxis providers     # live provider pool health
praxis doctor        # pre-flight: env, keys, DB, coder
praxis doctor --deep # also probes every design-chain entry

praxis discover --topic "..." --source arxiv --limit 10
praxis discover --topic "..." --pick 2 --focus "skip the UI"

praxis design 42
praxis design 42 --focus "skip the UI"
praxis design 42 --pass 4
praxis design 42 --resume
praxis design 42 --no-critic
praxis design 42 --show

praxis plan 42
praxis plan 42 --resume
praxis plan 42 --pass 3
praxis plan 42 --critic
praxis plan 42 --no-critic

praxis check 42
praxis check 42 --json

praxis handoff 42 ../MyProject
praxis handoff 42 ../MyProject --force
praxis handoff 42 ../MyProject --skip-check
```

## Design engine (multi-pass DESIGN.md)

`praxis design <id>` turns an analyzed candidate into an implementable design document. The command works on any persisted candidate — accepted, borderline (`review approved`), blueprinted, or selected via `praxis discover --pick N`.

The design runs as five paced LLM passes routed through the same provider pool and response cache as the rest of the pipeline. Rate limits are handled gracefully: when a provider rejects a pass with a TPM/429 error, the engine parses the provider's `try again in Ns` hint, waits that long (bounded, ~90 s max), and retries before failing over.

**Per-model limits.** `hardware_profile.yaml` carries structured limits per provider/model — `tpm`, `itpm`, `otpm`, and optional `max_output`. Pacing checks every applicable axis over a 60 s sliding window, and each call's `max_tokens` is clamped to the model's per-request output ceiling.

**"Request too large" is permanent.** A provider rejecting a pass as too large (HTTP 413) is never waited on or retried at that size, and is not put into a cooldown — smaller requests may still succeed there.

**Compact context.** Passes exchange a structured design state (~600-800 tokens) instead of full text of earlier passes. Per-pass `max_tokens` is ~2000; gpt-oss models on Groq/Cerebras get `reasoning_effort="low"`. Every call targets input + output ≤ ~5000 tokens.

`PRAXIS_DESIGN_MODEL` accepts a comma-separated chain of provider/model ids: a provider that keeps failing fails over to the next entry instead of ending the run. Each pass receives the facts sheet as HARD CONSTRAINTS, the focus note, the relevant paper chunks, and the design state; every pass output is persisted as it completes, so an interrupted run resumes at the failed pass:

1. **Technique** — the paper's core method: inputs, outputs, algorithm steps, formulas, thresholds, and evaluation setup, each with a section citation.
2. **Architecture** — components, responsibilities, data flow, interfaces, a Mermaid diagram, and a decision record per key choice.
3. **Data Model & Contracts** — schemas, module/file tree, CLI/API surface, and the core algorithm as pseudocode.
4. **Phased Implementation Plan** — Phase 1 is the smallest end-to-end vertical slice; tasks carry stable ids (`TASK-001...`) with acceptance criteria and tests.
5. **Hardware & Budget Fit** — a per-component RAM/CPU/$ table against the facts sheet with totals vs headroom, rejected alternatives, degradation plan, risks, and cuts.

A skeptical critic reviews the design per section and flags defects (budget-table drift, unsourced claims, uncovered components); only flagged sections are regenerated in bounded rounds. `--no-critic` skips the review.

Output is saved to the DB and written to `designs/<NNN>-<slug>/`: `DESIGN.md`, `TASKS.md`, `AGENT_PROMPT.md`, and `DESIGN.partial.md` (written when a run fails or is incomplete).

### Planning docs pack (praxis plan)

`praxis plan <id>` turns a finished design into the hand-off bundle a coding agent implements from: eleven files under `designs/<NNN>-<slug>/docs/` — `PRD.md`, `ARCHITECTURE.md`, `DESIGN.md`, `RULES.md`, `TASKS.md`, `TEST_PLAN.md`, `SECURITY.md`, `DECISIONS.md`, `MEMORY.md`, `README.md`, and `.env.example` — plus an optional twelfth, `RECOMMENDATIONS.md`. Six of the eleven are paced LLM passes in their own `plan` stage (PRD, rules, test plan, security, README, env example); the other five are rendered deterministically from the stored design passes. The Recommendations pass is generated automatically for stamped v0.2 packs (or on demand with `--recommend` on legacy packs): a hardware-fit summary driven by a deterministic fit table over the stored passes and blueprint, policed by the `r7-recommendation-claims` rule.

The pack critic is off by default (`--critic` opts in). `MEMORY.md` is seed-only: Praxis writes it once and the coding agent maintains it from then on.

## Configuration

Praxis loads a `.env` file automatically at CLI startup (via python-dotenv, `override=False`, so shell variables win over the file). `.env.example` is a reference template — copy it to `.env` and fill in your provider keys.

### Hardware profile

Fields resolve in order: environment variable → YAML file (`PRAXIS_CONFIG`) → default.

| Field | Type | Default | Env override |
|---|---|---|---|
| `cpu_only` | bool | `true` | `PRAXIS_CPU_ONLY` |
| `ram_gb` | int | `8` | `PRAXIS_RAM_GB` |
| `gpu` | bool | `false` | `PRAXIS_GPU` |
| `monthly_budget_usd` | float | `15.0` | `PRAXIS_MONTHLY_BUDGET_USD` |

### Model and pipeline

| Variable | Purpose | Default |
|---|---|---|
| `PRAXIS_MODEL` | litellm model id used by the Analyst/Architect | `groq/openai/gpt-oss-20b` |
| `PRAXIS_CODER` | Coder stage mode: `off` (default) or `opencode` | `off` |
| `PRAXIS_FEASIBILITY_THRESHOLD` | minimum feasibility score (0-10) | `4` |
| `PRAXIS_BORDERLINE_MARGIN` | band above the threshold treated as `borderline` | `1` |
| `PRAXIS_DB_PATH` | SQLite file path | `./praxis.db` |
| `PRAXIS_DB_URL` | full SQLAlchemy URL; overrides `PRAXIS_DB_PATH` | — |
| `PRAXIS_CONFIG` | path to the hardware profile YAML | `./hardware_profile.yaml` |
| `PRAXIS_DESIGN_MODEL` | design/plan chain, comma-separated | `groq/openai/gpt-oss-120b` |
| `PRAXIS_LLM_CACHE` | disable the LLM response cache with `0`/`false` | `1` |
| `PRAXIS_FALLBACK_MODELS` | comma-separated fallback models | — |
| `PRAXIS_PROVIDERS` | Job A provider order | configured chain order |
| `PRAXIS_PROVIDER_COOLDOWN_S` | exhausted-provider cooldown | `60` |
| `PRAXIS_GROUNDING_CACHE` | disable the source-grounding disk cache | `1` |
| `PRAXIS_GROUNDING_CACHE_DIR` | grounding cache directory | `./.praxis-cache/grounding` |
| `PRAXIS_SCRATCH_ROOT` | where the Coder creates prototype directories | `./scratch` |
| `PRAXIS_CODER_TIMEOUT_S` | OpenCode subprocess timeout | `600` |
| `PRAXIS_CODER_MAX_FAILURES` | circuit-breaker threshold | `2` |
| `PRAXIS_CODER_COOLDOWN_S` | circuit-breaker cooldown | `300` |
| `PRAXIS_CODER_MODELS` | Job B opencode model ids | opencode default |
| `PRAXIS_CODER_PROVIDER_RETRIES` | max models per coder attempt | `3` |
| `PRAXIS_CODER_OPENCODE_FLAGS` | extra flags after `opencode run` | `--auto` |
| `PRAXIS_MAX_TOKENS` | first-attempt `max_tokens` | — |
| `PRAXIS_MAX_TOKENS_RETRY` | `max_tokens` for the truncation retry | — |

## Provider failover

Job A (Analyst + Architect) consumes three free providers — NVIDIA NIM, Groq, OpenRouter — directly via litellm. A health-aware pool skips a cooling-down provider instantly, with zero wasted LLM calls re-trying a dead key. Exhaustion (HTTP 429, rate-limit/quota markers) puts the provider into a cooldown persisted in the `provider_health` table, so cooldowns survive restarts.

Job B (Coder, optional) routes through the local omniroute gateway, which owns its own keys and does provider-level failover internally. Praxis rotates the `--model omniroute/<id>` flag across the ids in `PRAXIS_CODER_MODELS`.

## Testing and CI

```bash
pytest        # full suite (474 tests)
ruff check .  # lint
```

CI (`.github/workflows/ci.yml`) installs the package with dev extras and runs `ruff check .` then `pytest` on Python 3.11 and 3.12 (ubuntu-latest). Tests mock the LLM client, HTTP fetches, the OpenCode subprocess, and the eval-harness agent calls, so the suite runs offline and deterministically.

## License

MIT — see [LICENSE](LICENSE) for the full text.
