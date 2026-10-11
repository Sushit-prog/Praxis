"""Tests for praxis.consistency — pure functions over plain dicts.

No db_session, no engine, no LLM: every fixture is a ``{pass_id: text}``
string, and rules are exercised individually plus via ``run_checks``.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from praxis.config import FactsSheet
from praxis.consistency import (
    KNOWN_PASS_IDS,
    RULE_R1,
    RULE_R2,
    RULE_R3,
    RULE_R4_CAP,
    RULE_R4_LEAVING,
    RULE_R5,
    RULE_R8,
    RULE_R9,
    RULE_R10,
    RULE_R13,
    RULE_REGISTRY_MISSING,
    RULE_REGISTRY_UNKNOWN,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Finding,
    check_r1,
    check_r2,
    check_r3,
    check_r4,
    check_r5,
    check_r8,
    check_r9,
    check_r10,
    check_r13,
    run_checks,
)


def _rule(findings, rule_id):
    return [f for f in findings if f.rule_id == rule_id]


# ---------------------------------------------------------------------------
# R1 (error): arithmetic
# ---------------------------------------------------------------------------


def test_r1_bad_multiplication_flagged():
    findings = check_r1({"plan": "The fallback is 2k × 3 MB is 6 MB."})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity, f.pass_id) == (RULE_R1, SEVERITY_ERROR, "plan")
    assert "6 MB" in f.message and "6000 MB" in f.message


def test_r1_good_multiplication_ok():
    text = (
        "Why: 2 k × ~3 MB is 6 GB, not 6 MB. "
        "The cap had to come down; ~1,200 × ~3 MB ≈ 3.6 GB."
    )
    assert check_r1({"architecture": text}) == []


@pytest.mark.parametrize("sep", ["\u202f", "\u00a0"])
def test_r1_digit_separator_normalization(sep):
    assert check_r1({"plan": f"~1{sep}200 × ~3 MB ≈ 3.6 GB."}) == []


def test_r1_token_multiplication_ignored():
    text = "ensure total daily tokens × 30 days ≤ 8000 TPM × 30 days"
    assert check_r1({"plan": text}) == []


def test_r1_l2_norm_ignored():
    text = "yields a vector of length 384 with L2 norm ≈ 1.0 (inference)."
    assert check_r1({"prd": text}) == []


def test_r1_records_basis_good_and_bad():
    good = {"architecture": "hard cap of ~1,200 active records (~3 MB each ≈ 3.6 GB)."}
    assert check_r1(good) == []
    bad = {"architecture": "hard cap of ~1,200 active records (~3 MB each ≈ 6 GB)."}
    findings = check_r1(bad)
    assert len(findings) == 1
    assert findings[0].rule_id == RULE_R1
    assert findings[0].severity == SEVERITY_ERROR


def test_r1_records_without_basis_skipped():
    text = (
        "ACTIVE_LIMIT = 5000 records (≈7 MB of embeddings + metadata). "
        "Keep the SQLite database size under 100 MB (≈ 1 000 records)."
    )
    assert check_r1({"data_contracts": text}) == []


def test_r1_table_row_both_clauses_violated_names_both():
    line = (
        "| **Vector Index** (in-memory) | ~2.5 KB per record | 1 | "
        "384-dim float32 vectors. 1,000 records = 1.5 MB RAM. "
        "10,000 records = 15 MB RAM. |"
    )
    findings = check_r1({"hardware_fit": line})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity, f.pass_id) == (RULE_R1, SEVERITY_ERROR, "hardware_fit")
    assert "1,000 records = 1.5 MB" in f.message
    assert "10,000 records = 15 MB" in f.message


def test_r1_table_row_first_clause_correct_only_second_flagged():
    line = (
        "| **Vector Index** | 2 KB per record | 1 | "
        "1,000 records = 2 MB RAM. 10,000 records = 15 MB RAM. |"
    )
    findings = check_r1({"hardware_fit": line})
    assert len(findings) == 1
    f = findings[0]
    assert "10,000 records = 15 MB" in f.message
    assert "1,000 records = 2 MB" not in f.message


# ---------------------------------------------------------------------------
# R2 (warning): comparison direction
# ---------------------------------------------------------------------------


def test_r2_free_storage_le_gate_flagged():
    text = (
        "| Disk-space usage | After benchmark, measure size of `memguard.db` "
        "and any logs. | ≤ 5 GB free storage remaining on the working volume "
        "(per facts sheet). |"
    )
    findings = check_r2({"test_plan": text})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity, f.pass_id) == (RULE_R2, SEVERITY_WARNING, "test_plan")
    assert "<=" in f.message


def test_r2_free_storage_ge_floor_ok():
    assert check_r2({"readme": "- **Free storage:** ≥ 5 GB on the working volume"}) == []


def test_r2_free_tier_excluded():
    text = (
        "- Batch size for API calls ≤ 1 (to stay within Groq free\u2011tier TPM). "
        "monthly API cost must stay ≤ $15 (inference based on Groq free-tier TPM "
        "limit). ensure total daily tokens × 30 days ≤ 8000 TPM × 30 days "
        "(per free-tier limit)."
    )
    assert check_r2({"plan": text}) == []


def test_r2_thresholds_and_limits_ok():
    text = (
        "Success Rate ≥ No-Memory baseline; Latency ≤ 3 s on the i5 CPU; "
        "over-usage incurs cost ≤ $15.00 (hard budget cap)."
    )
    assert check_r2({"test_plan": text}) == []


def test_r2_ascii_ge_in_code_ignored():
    fenced = "```python\nif desc['confidence'] >= 0.70:\n    state = 'active'\n```"
    unfenced = "if desc['confidence'] >= 0.70:"
    assert check_r2({"data_contracts": fenced}) == []
    assert check_r2({"data_contracts": unfenced}) == []


def test_r2_cap_written_with_ge_flagged():
    text = "The application memory limit ≥ 8 GB would exceed the host."
    findings = check_r2({"rules": text})
    assert len(findings) == 1
    assert findings[0].rule_id == RULE_R2
    assert findings[0].severity == SEVERITY_WARNING


# ---------------------------------------------------------------------------
# R4 (error): leaving-of-cap and component-vs-cap
# ---------------------------------------------------------------------------

_ARCH_SENTENCE = (
    "Memory: ~1,200 × ~3 MB ≈ 3.6 GB, leaving ~400 MB of the 4 GB headroom "
    "for the app's own ~350 MB footprint."
)


def test_r4_leaving_ok():
    assert check_r4({"architecture": _ARCH_SENTENCE}) == []


def test_r4_leaving_sum_wrong_flagged():
    text = (
        "Memory: ~1,200 × ~3 MB ≈ 3.6 GB, leaving ~1.0 GB of the 4 GB headroom "
        "for the app's own ~350 MB footprint."
    )
    findings = check_r4({"architecture": text})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity, f.pass_id) == (RULE_R4_LEAVING, SEVERITY_ERROR, "architecture")


def test_r4_leaving_footprint_exceeds_left_flagged():
    text = (
        "Memory: ~1,200 × ~3 MB ≈ 3.6 GB, leaving ~300 MB of the 4 GB headroom "
        "for the app's own ~350 MB footprint."
    )
    findings = check_r4({"architecture": text})
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == RULE_R4_LEAVING
    assert f.severity == SEVERITY_ERROR
    assert "footprint" in f.message


def test_r4_component_vs_cap_ok():
    text = (
        "3500 MB to stay under 4 GB headroom; 200 MB RAM, fits 4 GB headroom; "
        "RAM: 350 MB < 4,000 MB (Safe)."
    )
    assert check_r4({"env_example": text}) == []


def test_r4_component_exceeding_cap_flagged():
    text = "The resident set 4,500 MB to stay under 4 GB headroom would thrash."
    findings = check_r4({"rules": text})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity) == (RULE_R4_CAP, SEVERITY_ERROR)
    assert "4,500" in f.excerpt


def test_r4_multiplication_not_rechecked():
    assert check_r4({"architecture": "~1,200 × ~3 MB ≈ 3.6 GB."}) == []


def test_r4_free_storage_gate_left_to_r2():
    text = "| ≤ 5 GB free storage remaining on the working volume (per facts sheet). |"
    assert check_r4({"test_plan": text}) == []


def test_r4_semicolon_guard_no_false_positive():
    text = (
        "- **RAM:** 8 GB total; plan for ≤ 3.5 GB resident app usage to stay "
        "inside the ~4 GB usable headroom shared with Windows"
    )
    assert check_r4({"readme": text}) == []


def test_r4_short_gap_guard_no_false_positive():
    text = (
        "# DB_SIZE_LIMIT_BYTES: Security limit for SQLite file size to fit within "
        "free storage constraints (5 GB available, keeping <100MB per design)"
    )
    assert check_r4({"env_example": text}) == []


# ---------------------------------------------------------------------------
# R3 (warning): dependency conflict across passes
# ---------------------------------------------------------------------------

# Verbatim from the real design-5 contradiction (TASKS.md:8 vs SECURITY.md:15).
_R3_BAD_PLAN = (
    "Install Python 3.11, required packages (`requests`, `sqlite3`, "
    "`sentence‑transformers`, `scikit‑learn`)."
)
_R3_BAD_SECURITY = (
    "The dependency list is deliberately minimal: core packages include "
    "`langgraph`, `sentence-transformers`, `sqlite3` (standard library), "
    "`httpx` for API calls, and `pytest` for testing."
)


def test_r3_dependency_conflict_flagged():
    findings = check_r3({"plan": _R3_BAD_PLAN, "security": _R3_BAD_SECURITY})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity) == (RULE_R3, SEVERITY_WARNING)
    assert f.pass_id == "plan"  # alphabetically first of the conflicting pair
    assert "requests" in f.message and "httpx" in f.message
    assert "plan" in f.message and "security" in f.message


def test_r3_dependency_clean_same_pass_acknowledged():
    assert check_r3({"security": "Use `httpx` instead of `requests` for calls."}) == []


def test_r3_dependency_clean_same_package_across_passes():
    plan = "Install `httpx` and `sentence-transformers`."
    security = "Core packages: `httpx`, `sentence-transformers`."
    assert check_r3({"plan": plan, "security": security}) == []


def test_r3_dependency_clean_english_word_requests():
    text = "Queue requests if TPM > 7,000 and retry with backoff."
    assert check_r3({"architecture": text}) == []


def test_r3_dependency_clean_third_pass_acknowledges_both():
    plan = "Install `requests`."
    security = "Use `httpx`, not `requests`."
    assert check_r3({"plan": plan, "security": security}) == []


def test_r3_dependency_silent_on_malformed_input():
    assert check_r3({"plan": None}) == []  # type: ignore[dict-item]
    assert check_r3({"plan": object()}) == []  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# R5 (warning): similarity-metric conflict across passes
# ---------------------------------------------------------------------------

# Verbatim from the real design-5 contradiction (DESIGN.md:257 vs PRD.md:22).
_R5_BAD_DATA_CONTRACTS = "1. Perform k‑NN over `records.embedding` (Euclidean distance)."
_R5_BAD_PRD = (
    "RetrievalEngine shall perform k-NN cosine similarity over stored embeddings, "
    "re-rank results using `score = a*sim + b*R_m - c*nu_m` and return the "
    "top-k records."
)


def test_r5_metric_conflict_flagged():
    findings = check_r5({"data_contracts": _R5_BAD_DATA_CONTRACTS, "prd": _R5_BAD_PRD})
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.severity) == (RULE_R5, SEVERITY_WARNING)
    assert f.pass_id == "prd"  # first pass (alphabetically) using the minority metric
    assert "cosine" in f.message and "euclidean" in f.message
    assert "cosine similarity" in f.excerpt  # excerpt is from the minority pass


def test_r5_metric_clean_all_cosine():
    passes = {
        "prd": "k-NN cosine similarity over stored embeddings.",
        "test_plan": "returns the top-k nearest vectors sorted by cosine similarity",
        "data_contracts": "Perform k‑NN over `records.embedding` (cosine distance).",
    }
    assert check_r5(passes) == []


def test_r5_metric_clean_non_retrieval_comparison():
    text = "Euclidean distance and cosine similarity rank unit vectors differently."
    assert check_r5({"architecture": text}) == []


def test_r5_metric_clean_dedup_threshold_not_retrieval():
    text = "Detect pairs of active records with cosine similarity > threshold."
    assert check_r5({"data_contracts": text}) == []


def test_r5_metric_clean_l2_norm_note():
    text = "yields a vector of length 384 with L2 norm ~ 1.0 (inference)."
    assert check_r5({"prd": text}) == []


def test_r5_metric_clean_fenced_non_retrieval_block():
    """r5 scans fences, but the metric claim still needs retrieval context."""
    text = (
        "# Scratch notes\n\n"
        "```\n"
        "Euclidean distance and cosine similarity rank unit vectors differently.\n"
        "```\n"
    )
    assert check_r5({"architecture": text}) == []


def test_r5_metric_silent_on_malformed_input():
    assert check_r5({"prd": None}) == []  # type: ignore[dict-item]
    assert check_r5({"prd": object()}) == []  # type: ignore[dict-item]


def test_new_rules_warnings_only_exit_semantics():
    """The real contradiction text yields warnings (exit 0), never errors."""
    passes = {
        "plan": _R3_BAD_PLAN,
        "security": _R3_BAD_SECURITY,
        "data_contracts": _R5_BAD_DATA_CONTRACTS,
        "prd": _R5_BAD_PRD,
    }
    findings = run_checks(passes)
    r3 = _rule(findings, RULE_R3)
    r5 = _rule(findings, RULE_R5)
    assert len(r3) == 1 and len(r5) == 1
    assert not [f for f in findings if f.severity == SEVERITY_ERROR]


# ---------------------------------------------------------------------------
# R8 (warning/info): PRD section presence — one finding per rule
# ---------------------------------------------------------------------------

_R8_BAD_PRD = "## Users\nOnly students use it.\n"

_R8_LEGACY_SYNONYMS_PRD = """# The Problem
x
## Target Users
x
### Goals
x
## Core Features
x
### MVP
x
## Non-Goals (explicitly out of scope for v1)
x
### Success Metrics (v1)
x
"""


def test_r8_missing_sections_one_finding_listing_all():
    findings = check_r8({"prd": _R8_BAD_PRD}, stamped=True)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == RULE_R8 and f.pass_id == "prd"
    assert f.severity == SEVERITY_WARNING
    assert "missing" in f.message
    assert "Problem" in f.message
    assert "MVP" in f.message
    assert "Success criteria" in f.message
    assert "Users" not in f.message.split("missing", 1)[1].split("(required")[0]


def test_r8_all_sections_present_clean():
    assert check_r8({"prd": _R8_LEGACY_SYNONYMS_PRD}, stamped=True) == []


def test_r8_plural_headings_accepted():
    prd = (
        "# Problems\nx\n## Users\nx\n### Goals\nx\n## Features\nx\n"
        "### MVPs\nx\n## Out of scopes\nx\n### Success criteria\nx\n"
    )
    assert check_r8({"prd": prd}, stamped=True) == []


def test_r8_any_heading_level_accepted():
    prd = (
        "###### Problem\nx\n#### Users\nx\n##### Goal\nx\n#### Features\nx\n"
        "###### MVP\nx\n#### Out of scope\nx\n##### Success criteria\nx\n"
    )
    assert check_r8({"prd": prd}, stamped=True) == []


def test_r8_case_insensitive():
    prd = (
        "# problem\nx\n## USERS\nx\n### Goal\nx\n## features\nx\n"
        "### Mvp\nx\n## OUT OF SCOPE\nx\n### success CRITERIA\nx\n"
    )
    assert check_r8({"prd": prd}, stamped=True) == []


def test_r8_design5_style_headings_only_mvp_missing():
    prd = """## Product Requirements
### Problem & Goal
x
### Users and Use Cases
x
### Functional Requirements
x
### Success Metrics (v1)
x
### Non-Goals (explicitly out of scope for v1)
x
### Open Questions
x
"""
    findings = check_r8({"prd": prd})
    assert len(findings) == 1
    assert findings[0].severity == SEVERITY_INFO
    assert "missing MVP" in findings[0].message


@pytest.mark.parametrize(
    "passes",
    [
        {},
        {"prd": None},  # type: ignore[dict-item]
        {"prd": object()},  # type: ignore[dict-item]
        {"prd": "   \n  "},
    ],
)
def test_r8_silent_on_missing_or_unparseable(passes):
    assert check_r8(passes) == []


def test_r8_severity_matrix_stamped_vs_legacy():
    passes = {"prd": _R8_BAD_PRD}
    legacy = check_r8(passes)
    assert len(legacy) == 1 and legacy[0].severity == SEVERITY_INFO
    assert "legacy pack" in legacy[0].message
    stamped = check_r8(passes, stamped=True)
    assert len(stamped) == 1 and stamped[0].severity == SEVERITY_WARNING
    assert "legacy pack" not in stamped[0].message
    assert all(f.severity != SEVERITY_ERROR for f in legacy + stamped)


# ---------------------------------------------------------------------------
# R9 (warning/info): TASK-NNN acceptance window in the plan pass
# ---------------------------------------------------------------------------


def test_r9_acceptance_next_line_clean():
    plan = (
        "- [ ] **TASK-001** Install the dependencies.\n"
        "  *Acceptance:* `pip list` shows every package.\n"
    )
    assert check_r9({"plan": plan}, stamped=True) == []


@pytest.mark.parametrize(
    "marker",
    ["*Acceptance:* done", "**Acceptance**", "Acceptance: done", "Accepts: done"],
)
def test_r9_acceptance_marker_variants(marker):
    plan = f"- [ ] TASK-001 Do the thing.\n  {marker}\n"
    assert check_r9({"plan": plan}, stamped=True) == []


def test_r9_acceptance_at_window_edge_ok():
    plan = (
        "- [ ] TASK-001 Do the thing.\n"
        "line2\n"
        "line3\n"
        "  Acceptance: yes\n"
    )
    assert check_r9({"plan": plan}, stamped=True) == []


def test_r9_acceptance_outside_window_flagged():
    plan = (
        "- [ ] TASK-001 Do the thing.\n"
        "line2\n"
        "line3\n"
        "line4\n"
        "  Acceptance: yes\n"
    )
    findings = check_r9({"plan": plan}, stamped=True)
    assert len(findings) == 1
    assert "TASK-001" in findings[0].message


def test_r9_task_without_acceptance_one_finding_lists_all():
    plan = (
        "- [ ] TASK-001 First.\n  Acceptance: ok\n"
        "- [ ] TASK-002 Second, no acceptance here.\n"
        "- [ ] TASK-003 Third, also missing.\n"
    )
    findings = check_r9({"plan": plan}, stamped=True)
    assert len(findings) == 1
    assert "TASK-002" in findings[0].message
    assert "TASK-003" in findings[0].message
    assert "TASK-001" not in findings[0].message


def test_r9_plain_checkbox_list_silent_on_legacy():
    plan = "- [ ] Install dependencies\n- [ ] Write the client\n"
    assert check_r9({"plan": plan}) == []


def test_r9_plain_checkbox_list_warns_on_stamped():
    plan = "- [ ] Install dependencies\n- [ ] Write the client\n"
    findings = check_r9({"plan": plan}, stamped=True)
    assert len(findings) == 1
    assert findings[0].severity == SEVERITY_WARNING
    assert "TASK-NNN" in findings[0].message


def test_r9_severity_matrix_stamped_vs_legacy():
    plan = "- [ ] TASK-001 No acceptance.\n"
    legacy = check_r9({"plan": plan})
    assert len(legacy) == 1 and legacy[0].severity == SEVERITY_INFO
    assert "legacy pack" in legacy[0].message
    stamped = check_r9({"plan": plan}, stamped=True)
    assert len(stamped) == 1 and stamped[0].severity == SEVERITY_WARNING
    assert all(f.severity != SEVERITY_ERROR for f in legacy + stamped)


@pytest.mark.parametrize(
    "passes",
    [
        {},
        {"plan": None},  # type: ignore[dict-item]
        {"plan": object()},  # type: ignore[dict-item]
        {"plan": "   "},
    ],
)
def test_r9_silent_on_missing_or_unparseable(passes):
    assert check_r9(passes) == []


def test_r9_silent_when_tasks_are_fenced():
    plan = "```python\n# see TASK-001 in the docs\n```\n- [ ] free-form work\n"
    assert check_r9({"plan": plan}) == []


# ---------------------------------------------------------------------------
# R13 (warning/info): RULES category headings — extras never flagged
# ---------------------------------------------------------------------------

_R13_BAD_RULES = "## General\nBe careful.\n## Security\nScan secrets.\n"

_R13_GUIDE_STYLE_RULES = """# Engineering Rules
## General
x
## Before Coding
x
## UI
x
## Security
x
## Testing
x
## Git
x
"""


def test_r13_missing_categories_one_finding_listing_all():
    findings = check_r13({"rules": _R13_BAD_RULES}, stamped=True)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == RULE_R13 and f.pass_id == "rules"
    assert f.severity == SEVERITY_WARNING
    assert "Before coding" in f.message
    assert "Testing" in f.message
    assert "Git" in f.message
    assert "General" not in f.message.split("missing", 1)[1].split("(required")[0]


def test_r13_guide_style_with_ui_extra_clean():
    assert check_r13({"rules": _R13_GUIDE_STYLE_RULES}, stamped=True) == []


def test_r13_before_you_code_synonym():
    rules = (
        "## General\nx\n## Before you code\nx\n## Security\nx\n"
        "## Testing\nx\n## Git\nx\n"
    )
    assert check_r13({"rules": rules}, stamped=True) == []


def test_r13_severity_matrix_stamped_vs_legacy():
    passes = {"rules": _R13_BAD_RULES}
    legacy = check_r13(passes)
    assert len(legacy) == 1 and legacy[0].severity == SEVERITY_INFO
    assert "legacy pack" in legacy[0].message
    stamped = check_r13(passes, stamped=True)
    assert len(stamped) == 1 and stamped[0].severity == SEVERITY_WARNING
    assert all(f.severity != SEVERITY_ERROR for f in legacy + stamped)


@pytest.mark.parametrize(
    "passes",
    [
        {},
        {"rules": None},  # type: ignore[dict-item]
        {"rules": object()},  # type: ignore[dict-item]
        {"rules": "prose without any headings"},
    ],
)
def test_r13_missing_pass_silent_but_prose_reports_all(passes):
    if "rules" not in passes or not isinstance(passes["rules"], str):
        assert check_r13(passes) == []
    else:
        findings = check_r13(passes)
        assert len(findings) == 1  # parseable prose: every category missing


# ---------------------------------------------------------------------------
# Guide fixtures (human-written, read-only): tests/fixtures/guide/
# ---------------------------------------------------------------------------

_GUIDE_DIR = Path(__file__).parent / "fixtures" / "guide"


def _guide(name: str) -> str:
    """A guide fixture's text with LF endings (identical for LF/CRLF checkouts)."""
    return (_GUIDE_DIR / name).read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_guide_prd_clean_stamped_and_unstamped():
    text = _guide("prd.md")
    assert check_r8({"prd": text}) == []
    assert check_r8({"prd": text}, stamped=True) == []


def test_guide_rules_clean_stamped_and_unstamped():
    text = _guide("rules.md")
    assert check_r13({"rules": text}) == []
    assert check_r13({"rules": text}, stamped=True) == []


def test_guide_tasks_silent_unstamped_warns_stamped():
    text = _guide("tasks.md")
    assert check_r9({"plan": text}) == []
    findings = check_r9({"plan": text}, stamped=True)
    assert len(findings) == 1
    assert findings[0].rule_id == RULE_R9
    assert findings[0].severity == SEVERITY_WARNING
    assert "TASK-NNN" in findings[0].message


# ---------------------------------------------------------------------------
# R10 (warning): stack fit vs the facts sheet (+ blueprint source)
# ---------------------------------------------------------------------------

_FACTS = FactsSheet()  # usable_ram_gb=4, ram_gb=8, cpu_only=True, local_only=True

_R10_NEG_FIXTURES = [
    "Avoid CUDA-only libraries entirely - the machine is CPU-only.",
    "The machine is not CUDA-capable; avoid bitsandbytes.",
    "Instead of a rented VPS, run everything locally.",
    "Rejected: a rented VPS with 8 GB RAM was ruled out.",
    "No GPU required for this workload; CPU inference only.",
    "Use httpx instead of requests for all API calls.",
    "The model does not fit in the available headroom.",
    "Neo4j is avoided; use SQLite instead.",
    "No docker containers are needed.",
    "### Non-Goals (explicitly out of scope for v1)\n"
    "- GPU acceleration or CUDA-dependent libraries.",
    "### Non-Goals (explicitly out of scope for v1)\n"
    "- Use of Neo4j or other graph databases.",
    "The generator and verifier both call the same Groq gpt-oss-20b model via the API.",
    "LLM_MODEL_NAME=gpt-oss-20b",
    "Respecting the 8 GB RAM, CPU-only, and $15/month budget constraints.",
    "Peak RSS must stay <= 4 GB to respect the headroom.",
    "Disk-space usage must leave >= 5 GB free storage.",
    "All tests run on a CPU-only machine with 8 GB usable RAM.",
]

_R10_POS_FIXTURES = [
    (
        "Loads a 2.7B Llama-2 model via bitsandbytes on a rented VPS.",
        {"cuda-only", "remote-host"},
    ),
    (
        "| 0. Setup | Environment provisioning | VPS with 8 GB RAM, Ubuntu 22.04, "
        "bitsandbytes | Verify CPU inference performance |",
        {"cuda-only", "remote-host"},
    ),
    ("The largest memory footprint comes from the quantized LLM (~4 GB).", {"ram-claim"}),
    (
        "Combined with embeddings and the database, total usage stays below 8 GB.",
        {"ram-claim"},
    ),
    (
        # negation scope is one unit: "false" silences its own sentence only
        "The verifier marks corrupt records as false. "
        "It then loads the model via bitsandbytes.",
        {"cuda-only"},
    ),
]


@pytest.mark.parametrize("text", _R10_NEG_FIXTURES)
def test_r10_negation_fixtures_silent(text):
    assert check_r10({"plan": text}, _FACTS) == []


@pytest.mark.parametrize("text,want", _R10_POS_FIXTURES)
def test_r10_positive_fixtures_fire(text, want):
    findings = check_r10({"plan": text}, _FACTS)
    assert findings, f"expected findings for {text!r}"
    assert {f.rule_id for f in findings} == {RULE_R10}
    assert {f.severity for f in findings} == {SEVERITY_WARNING}
    kinds = {f.message.split(":", 1)[0] for f in findings}
    assert want <= kinds


# The five known firing rows of the stored candidate-5 blueprint (regression).
_R10_BP5_FIXTURE = (
    "## Proposed Architecture\n"
    "> **Buildable on Target Hardware** - The largest memory footprint comes "
    "from the quantized LLM (~4 GB). Combined with embeddings and the database, "
    "total usage stays below 8 GB. All operations run on CPU; no GPU is "
    "required. The monthly $15 budget covers a modest VPS (e.g., DigitalOcean "
    "Droplet 8GB RAM) and storage costs.\n"
    "| **LLM Inference Engine** | Generates task outputs and verifier "
    "responses | 1.3 B or 2.7 B Llama-2 (or similar) quantized to 8-bit, "
    "loaded locally via bitsandbytes. |\n"
    "| **0. Setup** | Environment provisioning | VPS with 8 GB RAM, Ubuntu "
    "22.04, Python 3.10, `transformers`, `bitsandbytes`, `hnswlib`, `LMDB` | "
    "Verify CPU inference performance |\n"
)


def test_r10_blueprint5_regression_five_known_rows():
    findings = check_r10({}, _FACTS, blueprint=_R10_BP5_FIXTURE)
    assert len(findings) == 5
    assert {f.pass_id for f in findings} == {"blueprint"}
    assert {f.severity for f in findings} == {SEVERITY_WARNING}
    kinds = [f.message.split(":", 1)[0] for f in findings]
    assert sorted(kinds) == ["cuda-only", "cuda-only", "ram-claim", "remote-host", "remote-host"]
    # the ~4 GB restatement and both 8 GB RAM restatements stay silent: the
    # single ram-claim is the upper-bound sentence, reworded as a bound
    ram = [f for f in findings if f.message.startswith("ram-claim")]
    assert len(ram) == 1
    assert "states an 8 GB bound; usable headroom is 4 GB" in ram[0].message
    assert "claims" not in ram[0].message


def test_r10_vram_dedupe_no_double_ram_claim():
    text = "Inference requires over 40GB of VRAM in fp16 and a 3B-parameter UNet."
    kinds = {
        f.message.split(":", 1)[0] for f in check_r10({"plan": text}, _FACTS)
    }
    assert "vram-claim" in kinds
    assert "ram-claim" not in kinds


def test_r10_gpu_host_silent_for_cuda_and_vram():
    facts = FactsSheet(cpu_only=False, gpu=True)
    text = "Runs bitsandbytes kernels with 40 GB VRAM on the host."
    assert check_r10({"plan": text}, facts) == []


def test_r10_non_local_host_silent_for_remote():
    facts = FactsSheet(local_only=False)
    assert check_r10({"plan": "Runs on a rented VPS."}, facts) == []


def test_r10_facts_none_silent():
    assert check_r10({"plan": _R10_POS_FIXTURES[0][0]}, None) == []


def test_r10_design5_passes_zero_findings():
    import json
    import sqlite3

    db_path = Path(__file__).parent.parent / "praxis.db"
    if not db_path.is_file():
        pytest.skip("local praxis.db not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT passes_json FROM designs WHERE candidate_id=5 "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("candidate 5 has no stored design")
    passes = {
        k: v for k, v in json.loads(row[0]).items() if not k.startswith("_")
    }
    assert check_r10(passes, _FACTS) == []


def test_r10_guide_fixtures_zero_findings():
    guide_pass = {
        "prd.md": "prd",
        "rules.md": "rules",
        "tasks.md": "plan",
        "memory.md": "memory",
    }
    for name, pid in guide_pass.items():
        text = _guide(name)
        assert check_r10({pid: text}, _FACTS) == [], name


def test_r10_deterministic():
    passes = {"plan": _R10_POS_FIXTURES[0][0], "prd": _R10_POS_FIXTURES[2][0]}
    assert check_r10(passes, _FACTS) == check_r10(passes, _FACTS)


def test_run_checks_blueprints_scanned_under_blueprint_label():
    findings = run_checks({}, _FACTS, blueprint=_R10_BP5_FIXTURE)
    r10 = [f for f in findings if f.rule_id == RULE_R10]
    assert r10 and {f.pass_id for f in r10} == {"blueprint"}


# ---------------------------------------------------------------------------
# run_checks stamped wiring for the content rules
# ---------------------------------------------------------------------------


def test_run_checks_stamped_kwarg_flows_to_content_rules():
    passes = {"prd": _R8_BAD_PRD, "rules": _R13_BAD_RULES, "plan": "no ids here"}
    legacy = run_checks(passes)
    content = [f for f in legacy if f.rule_id in (RULE_R8, RULE_R9, RULE_R13)]
    assert {f.severity for f in content} == {SEVERITY_INFO}
    stamped = run_checks(passes, stamped=True)
    content = [f for f in stamped if f.rule_id in (RULE_R8, RULE_R9, RULE_R13)]
    assert {f.severity for f in content} == {SEVERITY_WARNING}
    assert not [f for f in stamped if f.severity == SEVERITY_ERROR]


# ---------------------------------------------------------------------------
# Registry and entry point
# ---------------------------------------------------------------------------


def test_registry_unknown_key_is_warning_missing_is_info():
    findings = run_checks({"plan": "ok", "weird": "x"})
    unknown = _rule(findings, RULE_REGISTRY_UNKNOWN)
    missing = _rule(findings, RULE_REGISTRY_MISSING)
    assert len(unknown) == 1
    assert unknown[0].severity == SEVERITY_WARNING
    assert unknown[0].pass_id == "weird"
    assert len(missing) == len(KNOWN_PASS_IDS) - 1
    assert all(f.severity == SEVERITY_INFO for f in missing)
    assert not [f for f in findings if f.severity == SEVERITY_ERROR]
    assert len(findings) == len(KNOWN_PASS_IDS)


def test_empty_passes_reports_every_known_pass_missing():
    findings = run_checks({})
    assert len(findings) == len(KNOWN_PASS_IDS)
    assert all(f.rule_id == RULE_REGISTRY_MISSING for f in findings)


def test_run_checks_deterministic_and_sorted():
    passes = {
        "plan": "2k × 3 MB is 6 MB.",
        "test_plan": "| ≤ 5 GB free storage remaining on the volume |",
        "architecture": _ARCH_SENTENCE,
    }
    first = run_checks(passes)
    second = run_checks(passes)
    assert first == second
    keys = [(f.pass_id, f.rule_id, f.excerpt) for f in first]
    assert keys == sorted(keys)
    assert any(f.rule_id == RULE_R1 for f in first)
    assert any(f.rule_id == RULE_R2 for f in first)


def test_finding_is_frozen():
    f = Finding(RULE_R1, SEVERITY_ERROR, "plan", "x", "y")
    with pytest.raises(FrozenInstanceError):
        f.rule_id = "z"
