"""Eval harness tests: golden-set loading, blueprint rubric, eval runs, reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from praxis.agents.analyst import DEFAULT_FEASIBILITY_THRESHOLD, AnalysisResult
from praxis.config import HardwareProfile
from praxis.db import Blueprint
from praxis.eval import (
    AnalystEvalResult,
    ArchitectEvalResult,
    EvalReport,
    GoldenFixture,
    RubricCheck,
    format_report,
    load_golden_set,
    run_analyst_eval,
    run_eval,
    run_rubric,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "golden_candidates.json"

GOOD_BLUEPRINT = """# CPU Fine-Tune — Blueprint

## Problem Statement
Build a small CPU fine-tuner.

## Proposed Architecture
CPU-only with 8GB RAM; no GPU needed. Python + PyTorch CPU.

## Phased Build Plan
1. Data loader
2. Training loop

## Deferred to Later Versions
GPU acceleration.

## Difficulty & Time Estimate
Easy, 1 week.
"""


def make_fixture(
    fixture_id="cpu-finetune",
    verdict="accept",
    score_min=5,
    score_max=10,
    raw_text="Fine-tune a small transformer on a single CPU.",
):
    return GoldenFixture(
        id=fixture_id,
        source="arxiv",
        url=f"https://example.com/{fixture_id}",
        title=fixture_id,
        raw_text=raw_text,
        expected_verdict=verdict,
        score_min=score_min,
        score_max=score_max,
    )


def make_analysis(score=7, rejected=False):
    return AnalysisResult(
        technique_summary="t",
        feasibility_score=score,
        feasibility_reasoning="r",
        rejected=rejected,
    )


# --------------------------------------------------------------------------
# Golden set loading
# --------------------------------------------------------------------------


def test_bundled_golden_set_loads():
    fixtures = load_golden_set(FIXTURES)

    assert len(fixtures) >= 10
    assert len({f.id for f in fixtures}) == len(fixtures)
    assert {f.expected_verdict for f in fixtures} == {"accept", "reject"}
    for fixture in fixtures:
        assert 0 <= fixture.score_min <= fixture.score_max <= 10

    accepts = [f for f in fixtures if f.expected_verdict == "accept"]
    rejects = [f for f in fixtures if f.expected_verdict == "reject"]
    assert accepts and rejects
    # Bands must be consistent with the default feasibility threshold:
    # accept fixtures clear it, reject fixtures sit below it.
    assert all(f.score_min >= DEFAULT_FEASIBILITY_THRESHOLD for f in accepts)
    assert all(f.score_max < DEFAULT_FEASIBILITY_THRESHOLD for f in rejects)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            {"id": "x", "source": "arxiv", "url": "u", "title": "t", "raw_text": "r",
             "expected_verdict": "accept", "score_min": 5},
            "missing 'score_max'",
        ),
        (
            {"id": "x", "source": "arxiv", "url": "u", "title": "t", "raw_text": "r",
             "expected_verdict": "maybe", "score_min": 5, "score_max": 10},
            "expected_verdict must be 'accept' or 'reject'",
        ),
        (
            {"id": "x", "source": "arxiv", "url": "u", "title": "t", "raw_text": "r",
             "expected_verdict": "accept", "score_min": 9, "score_max": 2},
            "expected 0 <= score_min <= score_max <= 10",
        ),
    ],
)
def test_load_golden_set_rejects_invalid_fixture(tmp_path, entry, match):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_golden_set(path)


def test_load_golden_set_rejects_non_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON list"):
        load_golden_set(path)


def test_load_golden_set_rejects_duplicate_ids(tmp_path):
    entry = {
        "id": "dup",
        "source": "arxiv",
        "url": "u",
        "title": "t",
        "raw_text": "r",
        "expected_verdict": "accept",
        "score_min": 5,
        "score_max": 10,
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate fixture id\(s\): dup"):
        load_golden_set(path)


# --------------------------------------------------------------------------
# Deterministic blueprint rubric
# --------------------------------------------------------------------------


def test_rubric_good_blueprint_passes():
    checks = run_rubric(GOOD_BLUEPRINT, HardwareProfile())

    assert [c.name for c in checks] == [
        "required_sections",
        "milestones",
        "hardware_scoped_architecture",
        "no_gpu_requirement",
        "ram_within_profile",
    ]
    assert all(c.passed for c in checks)


def test_rubric_missing_section_fails():
    md = GOOD_BLUEPRINT.replace("## Deferred to Later Versions\nGPU acceleration.\n\n", "")

    failed = {c.name: c for c in run_rubric(md, HardwareProfile()) if not c.passed}

    assert "required_sections" in failed
    assert "Deferred to Later Versions" in failed["required_sections"].detail


def test_rubric_cuda_requirement_fails_on_cpu_profile():
    md = GOOD_BLUEPRINT.replace("no GPU needed", "requires CUDA kernels")

    failed = {c.name: c for c in run_rubric(md, HardwareProfile()) if not c.passed}

    assert "no_gpu_requirement" in failed


def test_rubric_negated_gpu_passes():
    md = GOOD_BLUEPRINT.replace("no GPU needed", "does not require a GPU")

    checks = run_rubric(md, HardwareProfile())

    assert all(c.passed for c in checks)


def test_rubric_deferred_gpu_passes():
    md = GOOD_BLUEPRINT.replace(
        "GPU acceleration.", "GPU acceleration and CUDA kernels (deferred to v2)."
    )

    checks = run_rubric(md, HardwareProfile())

    assert all(c.passed for c in checks)


def test_rubric_deferred_gpu_passes_with_crlf():
    # CRLF line endings must not disable the deferred-section handling.
    md = GOOD_BLUEPRINT.replace(
        "GPU acceleration.", "GPU acceleration and CUDA kernels (deferred to v2)."
    ).replace("\n", "\r\n")

    checks = run_rubric(md, HardwareProfile())

    assert all(c.passed for c in checks)


def test_rubric_oversized_ram_fails_unless_profile_allows_it():
    md = GOOD_BLUEPRINT.replace("Python + PyTorch CPU.", "Requires 32GB RAM for the index.")

    failed_8gb = {c.name: c for c in run_rubric(md, HardwareProfile(ram_gb=8)) if not c.passed}
    assert "ram_within_profile" in failed_8gb

    checks_64gb = run_rubric(md, HardwareProfile(ram_gb=64))
    assert all(c.passed for c in checks_64gb)


def test_rubric_negated_ram_requirement_passes():
    # "no GPU needed" belongs to a different clause and must not hide the
    # 32GB requirement; a real "does not require" must still be respected.
    md = GOOD_BLUEPRINT.replace("Python + PyTorch CPU.", "Requires 32GB RAM for the index.")
    assert "ram_within_profile" in {
        c.name for c in run_rubric(md, HardwareProfile(ram_gb=8)) if not c.passed
    }

    negated = GOOD_BLUEPRINT.replace(
        "Python + PyTorch CPU.", "Does not require more than 8GB RAM."
    )
    checks = run_rubric(negated, HardwareProfile(ram_gb=8))
    assert all(c.passed for c in checks)


def test_rubric_plan_without_milestones_fails():
    md = GOOD_BLUEPRINT.replace(
        "1. Data loader\n2. Training loop", "First build a data loader, then the training loop."
    )

    failed = {c.name: c for c in run_rubric(md, HardwareProfile()) if not c.passed}

    assert "milestones" in failed


def test_rubric_architecture_not_hardware_scoped_fails():
    md = GOOD_BLUEPRINT.replace(
        "CPU-only with 8GB RAM; no GPU needed. Python + PyTorch CPU.",
        "A modular design with clean interfaces and dependency injection.",
    )

    failed = {c.name: c for c in run_rubric(md, HardwareProfile()) if not c.passed}

    assert "hardware_scoped_architecture" in failed


def test_rubric_gpu_check_skipped_when_profile_allows_gpu():
    md = GOOD_BLUEPRINT.replace("no GPU needed", "requires CUDA")

    checks = run_rubric(md, HardwareProfile(gpu=True, cpu_only=False))

    assert all(c.passed for c in checks)


# --------------------------------------------------------------------------
# Analyst eval
# --------------------------------------------------------------------------


def _analyst_harness(analyses):
    """Build an analyze_fn that dispatches on the fixture id embedded in the URL."""

    def fake_analyze(candidate, profile, threshold=None):
        return analyses[candidate.url.rsplit("/", 1)[-1]]

    return fake_analyze


def test_run_analyst_eval_all_pass():
    fixtures = [make_fixture("f1", "accept", 5, 10), make_fixture("f2", "reject", 0, 3)]
    analyses = {
        "f1": make_analysis(score=8, rejected=False),
        "f2": make_analysis(score=1, rejected=True),
    }

    results = run_analyst_eval(fixtures, HardwareProfile(), analyze_fn=_analyst_harness(analyses))

    assert len(results) == 2
    assert all(r.passed for r in results)
    assert [r.actual_verdict for r in results] == ["accept", "reject"]


def test_run_analyst_eval_verdict_mismatch_fails():
    fixtures = [make_fixture("f1", "accept", 5, 10)]

    results = run_analyst_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=8, rejected=True)}),
    )

    assert results[0].passed is False
    assert results[0].actual_verdict == "reject"
    assert results[0].error is None


def test_run_analyst_eval_score_out_of_band_fails():
    fixtures = [make_fixture("f1", "accept", 7, 10)]

    results = run_analyst_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=5, rejected=False)}),
    )

    assert results[0].passed is False


def test_run_analyst_eval_crash_surfaces_as_failure():
    fixtures = [make_fixture("f1", "accept", 5, 10)]

    def fake_analyze(candidate, profile, threshold=None):
        raise RuntimeError("analyst boom")

    results = run_analyst_eval(fixtures, HardwareProfile(), analyze_fn=fake_analyze)

    assert results[0].passed is False
    assert results[0].actual_verdict == "error"
    assert results[0].error == "analyst boom"


def test_run_analyst_eval_score_at_threshold_is_accepted():
    # A score exactly at the default threshold (4) is not rejected (4 < 4 is false).
    fixtures = [make_fixture("f1", "accept", 4, 7)]

    results = run_analyst_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=4, rejected=False)}),
    )

    assert results[0].passed


def test_run_analyst_eval_passes_full_raw_text_to_agent():
    long_text = "word " * 3000
    fixtures = [make_fixture("long", "accept", 5, 10, raw_text=long_text)]
    seen = {}

    def fake_analyze(candidate, profile, threshold=None):
        seen["raw_len"] = len(candidate.raw_text)
        return make_analysis(score=6, rejected=False)

    results = run_analyst_eval(fixtures, HardwareProfile(), analyze_fn=fake_analyze)

    assert results[0].passed
    assert seen["raw_len"] == len(long_text)


# --------------------------------------------------------------------------
# Full eval (analyst + architect)
# --------------------------------------------------------------------------


def _make_blueprint(md):
    return Blueprint(candidate_id=1, feasibility_score=7.0, blueprint_md=md)


def test_run_eval_full_flow_passes():
    fixtures = [make_fixture("f1", "accept", 5, 10)]

    def fake_blueprint(candidate, analysis, profile):
        return _make_blueprint(GOOD_BLUEPRINT)

    report = run_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=7, rejected=False)}),
        blueprint_fn=fake_blueprint,
    )

    assert report.passed
    assert len(report.analyst) == 1 and report.analyst[0].passed
    assert len(report.architect) == 1 and report.architect[0].passed
    assert len(report.architect[0].checks) == 5


def test_run_eval_bad_blueprint_fails_report():
    fixtures = [make_fixture("f1", "accept", 5, 10)]

    def fake_blueprint(candidate, analysis, profile):
        return _make_blueprint("# Terse\n\nNo sections here at all.")

    report = run_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=7, rejected=False)}),
        blueprint_fn=fake_blueprint,
    )

    assert not report.passed
    architect = report.architect[0]
    assert not architect.passed
    assert any(c.name == "required_sections" and not c.passed for c in architect.checks)


def test_run_eval_analyst_rejection_reports_architect_failure():
    fixtures = [make_fixture("f1", "accept", 5, 10)]

    def fake_blueprint(candidate, analysis, profile):
        return _make_blueprint(GOOD_BLUEPRINT)

    report = run_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=2, rejected=True)}),
        blueprint_fn=fake_blueprint,
    )

    assert not report.passed
    assert report.analyst[0].passed is False
    assert report.architect[0].passed is False
    assert "no blueprint produced" in report.architect[0].error


def test_run_eval_skips_architect_for_reject_fixtures():
    fixtures = [make_fixture("f1", "reject", 0, 3)]

    def fake_blueprint(candidate, analysis, profile):
        raise AssertionError("blueprint should not be called for a rejected fixture")

    report = run_eval(
        fixtures,
        HardwareProfile(),
        analyze_fn=_analyst_harness({"f1": make_analysis(score=1, rejected=True)}),
        blueprint_fn=fake_blueprint,
    )

    assert report.analyst[0].passed
    assert report.architect == []


# --------------------------------------------------------------------------
# Report formatting
# --------------------------------------------------------------------------


def test_format_report_pass():
    report = EvalReport(
        analyst=[AnalystEvalResult("f1", "accept", "accept", 7, 5, 10)],
        architect=[ArchitectEvalResult("f1", [RubricCheck("required_sections", True)])],
    )

    text = format_report(report)

    assert "OVERALL: PASS" in text
    assert "analyst: 1/1 passed" in text
    assert "architect: 1/1 passed" in text


def test_format_report_failures():
    report = EvalReport(
        analyst=[
            AnalystEvalResult("ok", "accept", "accept", 8, 5, 10),
            AnalystEvalResult("bad", "accept", "reject", 2, 5, 10),
        ],
        architect=[
            ArchitectEvalResult(
                "bad",
                [
                    RubricCheck("milestones", False, "no milestones"),
                    RubricCheck("no_gpu_requirement", True),
                ],
            )
        ],
    )

    text = format_report(report)

    assert "OVERALL: FAIL" in text
    assert "analyst: 1/2 passed" in text
    assert "[PASS] ok" in text
    assert "[FAIL] bad" in text
    assert "milestones: no milestones" in text
