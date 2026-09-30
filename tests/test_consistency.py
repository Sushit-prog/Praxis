"""Tests for praxis.consistency — pure functions over plain dicts.

No db_session, no engine, no LLM: every fixture is a ``{pass_id: text}``
string, and rules are exercised individually plus via ``run_checks``.
"""

from dataclasses import FrozenInstanceError

import pytest

from praxis.consistency import (
    KNOWN_PASS_IDS,
    RULE_R1,
    RULE_R2,
    RULE_R4_CAP,
    RULE_R4_LEAVING,
    RULE_REGISTRY_MISSING,
    RULE_REGISTRY_UNKNOWN,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Finding,
    check_r1,
    check_r2,
    check_r4,
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
