"""Deterministic consistency checks over stored design passes.

Zero-LLM, read-only: pure functions over ``{pass_id: text}`` dicts. Nothing
here touches the database, the network, or the model layer; the facts sheet
argument is accepted for API stability but v1 rules are text-internal.

Findings are data (rule id, severity, pass, excerpt, message) so callers decide
policy: the CLI exits nonzero only on ``severity == "error"``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from praxis.config import FactsSheet
from praxis.design import PASS_IDS
from praxis.planning import PACK_PASS_IDS

# Union of the core design passes and the docs-pack passes; a stored design is
# expected to carry exactly these keys.
KNOWN_PASS_IDS: tuple[str, ...] = tuple(PASS_IDS) + tuple(PACK_PASS_IDS)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

RULE_R1 = "r1-arithmetic"
RULE_R2 = "r2-direction"
RULE_R4_LEAVING = "r4-leaving"
RULE_R4_CAP = "r4-cap"
RULE_REGISTRY_UNKNOWN = "registry-unknown"
RULE_REGISTRY_MISSING = "registry-missing"


@dataclass(frozen=True)
class Finding:
    """One consistency problem in one pass."""

    rule_id: str
    severity: str
    pass_id: str
    excerpt: str
    message: str


# ---------------------------------------------------------------------------
# Shared normalization: whitespace, units, numbers, clauses
# ---------------------------------------------------------------------------

# Narrow no-break space (U+202F), no-break space (U+00A0) and thin space
# (U+2009) appear as digit separators in generated prose ("1 200 records").
_WS_TABLE = {0x202F: " ", 0x00A0: " ", 0x2009: " "}

_UNIT = r"KB|MB|GB|TB"
# Digit groups optionally separated by comma/space/tab (never newline):
# 4,000 / 1 000 / 3.6. Newline must not join two unrelated numbers.
_NUM = r"\d+(?:[,\t ]\d+)*(?:\.\d+)?"
_APPROX = 0.10  # relative tolerance for "~" / "approx" claims
_EXACT = 1e-6

_UNIT_TO_MB = {"KB": 0.001, "MB": 1.0, "GB": 1000.0, "TB": 1_000_000.0}


def _normalize(text: str) -> str:
    return text.translate(str.maketrans(_WS_TABLE))


def _strip_fences(text: str) -> str:
    """Drop ``` fenced blocks (code snippets are not prose claims)."""
    return re.sub(r"```.*?```", " ", text, flags=re.S)


def _num(s: str) -> float:
    return float(re.sub(r"[,\t ]", "", s))


def _to_mb(value: float, unit: str) -> float:
    return value * _UNIT_TO_MB[unit.upper()]


def _fmt(value: float) -> str:
    return f"{value:g}"


def _mismatch(claimed: float, computed: float, approx: bool) -> bool:
    denom = max(abs(claimed), abs(computed), 1e-9)
    rel = abs(claimed - computed) / denom
    return rel > (_APPROX if approx else _EXACT)


def _excerpt(text: str, start: int, end: int) -> str:
    return " ".join(text[max(0, start - 60) : end + 60].split())


def _clause_start(text: str, pos: int) -> int:
    """Start of the clause containing pos (decimal points are not splits)."""
    start = 0
    for m in _BOUNDARY.finditer(text, 0, pos):
        start = m.end()
    return start


def _clause(text: str, start: int, end: int) -> str:
    """The clause (split on . ; | and newlines) containing [start, end)."""
    left = _clause_start(text, start)
    nxt = _BOUNDARY.search(text, end)
    right = nxt.start() if nxt else len(text)
    return text[left:right]


# ---------------------------------------------------------------------------
# R1 (error): arithmetic — multiplication, basis-in-paren, table-row products
# ---------------------------------------------------------------------------

_MULT = re.compile(
    rf"(?P<n1>{_NUM})\s*(?P<s1>[kKmM])?\s*(?:[×]|\s[xX]\s)\s*~?"
    rf"(?P<n2>{_NUM})\s*(?P<u2>{_UNIT})\s*"
    rf"(?P<conn>≈|\bis\b|=|equals?)\s*~?"
    rf"(?P<n3>{_NUM})\s*(?P<u3>{_UNIT})",
    re.I,
)

_BASIS_PAREN = re.compile(
    rf"(?P<n>{_NUM})(?:\s+(?P<s>[kK]))?\s+(?:[A-Za-z][A-Za-z-]*\s+){{0,2}}records?\s*\(\s*~?"
    rf"(?P<x>{_NUM})\s*(?P<ux>{_UNIT})\s+(?:each|per\s+record|per\s+row|apiece)\b"
    rf"[^)]*?(?P<conn>≈|\bis\b|=|about|equals?)\s*~?"
    rf"(?P<t>{_NUM})\s*(?P<ut>{_UNIT})",
    re.I,
)

_TABLE_BASIS = re.compile(
    rf"(?P<approx>~?)(?P<x>{_NUM})\s*(?P<ux>{_UNIT})\s+per\s+record", re.I
)
_TABLE_TOTAL = re.compile(rf"(?P<n>{_NUM})\s*records?\s*=\s*(?P<t>{_NUM})\s*(?P<ut>{_UNIT})", re.I)

_COUNT_SCALE = {"k": 1000.0, "m": 1_000_000.0}
_CONN_APPROX = {"≈", "about"}


def _conn_is_approx(conn: str) -> bool:
    return conn.lower() in _CONN_APPROX


def check_r1(passes: Mapping[str, str], facts: FactsSheet | None = None) -> list[Finding]:
    del facts  # text-internal rule
    findings: list[Finding] = []
    for pass_id, raw in passes.items():
        text = _strip_fences(_normalize(raw))

        for m in _MULT.finditer(text):
            n1 = _num(m.group("n1")) * _COUNT_SCALE.get((m.group("s1") or "").lower(), 1.0)
            computed = n1 * _to_mb(_num(m.group("n2")), m.group("u2"))
            claimed = _to_mb(_num(m.group("n3")), m.group("u3"))
            if _mismatch(claimed, computed, _conn_is_approx(m.group("conn"))):
                lhs = (
                    f"{m.group('n1')}{m.group('s1') or ''} × "
                    f"{m.group('n2')} {m.group('u2')}"
                )
                findings.append(
                    Finding(
                        RULE_R1,
                        SEVERITY_ERROR,
                        pass_id,
                        _excerpt(text, m.start(), m.end()),
                        f"multiplication mismatch: '{lhs}' claims "
                        f"{_fmt(claimed)} MB, computes {_fmt(computed)} MB",
                    )
                )

        for m in _BASIS_PAREN.finditer(text):
            n = _num(m.group("n")) * _COUNT_SCALE.get((m.group("s") or "").lower(), 1.0)
            computed = n * _to_mb(_num(m.group("x")), m.group("ux"))
            claimed = _to_mb(_num(m.group("t")), m.group("ut"))
            if _mismatch(claimed, computed, _conn_is_approx(m.group("conn"))):
                count = f"{m.group('n')}{m.group('s') or ''} records"
                findings.append(
                    Finding(
                        RULE_R1,
                        SEVERITY_ERROR,
                        pass_id,
                        _excerpt(text, m.start(), m.end()),
                        f"records product mismatch: '{count}' at "
                        f"{_fmt(_num(m.group('x')))} {m.group('ux')} each claims "
                        f"{_fmt(claimed)} MB, computes {_fmt(computed)} MB",
                    )
                )

        for line in text.splitlines():
            basis = _TABLE_BASIS.search(line)
            if basis is None:
                continue
            per_record_mb = _to_mb(_num(basis.group("x")), basis.group("ux"))
            approx = bool(basis.group("approx"))
            violated: list[str] = []
            for total in _TABLE_TOTAL.finditer(line):
                claimed = _to_mb(_num(total.group("t")), total.group("ut"))
                computed = per_record_mb * _num(total.group("n"))
                if _mismatch(claimed, computed, approx):
                    violated.append(
                        f"'{total.group(0).strip()}' implies {_fmt(computed)} MB"
                    )
            if violated:
                findings.append(
                    Finding(
                        RULE_R1,
                        SEVERITY_ERROR,
                        pass_id,
                        _excerpt(line, basis.start(), basis.end()),
                        f"per-record basis '{basis.group(0).strip()}' contradicts: "
                        + "; ".join(violated),
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# R2 (warning): comparison direction on free-storage gates and caps
# ---------------------------------------------------------------------------

# "free-tier" is a rate-limit noun, not a storage quantity.
_FREE_TIER = re.compile(r"free\s*\W{0,3}tier", re.I)
_STORAGE_KW = re.compile(
    r"free\s+storage|storage\s+remaining|free\s+space|available\s+(?:free\s+)?storage",
    re.I,
)
_CAP_NOUN = re.compile(r"\b(?:cap|limit|maximum|max|budget)\b", re.I)
# The unit alternation must be parenthesized: a bare "KB|MB|GB|TB" would bind
# at the top level and match "GB" anywhere in the text.
_LE_GATE = re.compile(rf"(?:≤|<=)\s*{_NUM}\s*(?:{_UNIT})")
_GE_VALUE = re.compile(rf"(?:≥|>=)\s*{_NUM}\s*(?:{_UNIT})")
# Clause boundary: ";", "|", newline, and "." unless it sits between two
# digits (decimal numbers like 3.6 are not sentence ends).
_BOUNDARY = re.compile(r"(?<!\d)\.|\.(?!\d)|[;|\n]")


def check_r2(passes: Mapping[str, str], facts: FactsSheet | None = None) -> list[Finding]:
    del facts  # text-internal rule
    findings: list[Finding] = []
    for pass_id, raw in passes.items():
        text = _strip_fences(_normalize(raw))
        for clause in _BOUNDARY.split(text):
            if _FREE_TIER.search(clause):
                continue
            if _LE_GATE.search(clause) and _STORAGE_KW.search(clause):
                findings.append(
                    Finding(
                        RULE_R2,
                        SEVERITY_WARNING,
                        pass_id,
                        _excerpt(clause, 0, len(clause)),
                        "direction: free-storage requirement written with <= "
                        f"(should be >=): {clause.strip()[:90]}",
                    )
                )
            if _GE_VALUE.search(clause) and _CAP_NOUN.search(clause):
                findings.append(
                    Finding(
                        RULE_R2,
                        SEVERITY_WARNING,
                        pass_id,
                        _excerpt(clause, 0, len(clause)),
                        "direction: cap/limit written with >= "
                        f"(should be <=): {clause.strip()[:90]}",
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# R4 (error): leaving-of-cap and component-vs-cap
# ---------------------------------------------------------------------------

_LEAVING = re.compile(
    rf"leaving\s+~?(?P<left>{_NUM})\s*(?P<lu>{_UNIT})"
    rf"\s+of\s+(?:the\s+|a\s+)?~?(?P<cap>{_NUM})\s*(?P<cu>{_UNIT})"
    rf"\s+(?P<ctx>headroom|ceiling|budget|limit|constraint)",
    re.I,
)
_USED_BEFORE = re.compile(rf"(?:≈|=|\bis\b)\s*~?(?P<v>{_NUM})\s*(?P<u>{_UNIT})", re.I)
_FOOTPRINT = re.compile(
    rf"for\s+(?:the\s+app['’]s\s+own\s+)?~?(?P<fp>{_NUM})\s*(?P<fu>{_UNIT})\s+footprint",
    re.I,
)


def _gap(tokens: int) -> str:
    return rf"(?:\s+\w+\s*,?){{0,{tokens}}}"


# Bare operators need a cap noun in their clause; a short gap keeps
# "N GB available, keeping <100MB" from parsing as a component-vs-cap claim.
_BARE_CAP = re.compile(
    rf"(?P<a>{_NUM})\s*(?P<ua>{_UNIT}){_gap(1)}\s*(?P<op><=|≤|<)\s*"
    rf"(?:the\s+|a\s+)?~?(?P<b>{_NUM})\s*(?P<ub>{_UNIT})",
    re.I,
)
_CTX_CAP = re.compile(
    rf"(?P<a>{_NUM})\s*(?P<ua>{_UNIT}){_gap(5)}\s*"
    rf"(?P<op>within|under|fits\s+inside|fits|inside)\s+"
    rf"(?:the\s+|a\s+)?~?(?P<b>{_NUM})\s*(?P<ub>{_UNIT})",
    re.I,
)
_BARE_OPS = {"<", "<=", "≤"}
_CAP_NOUN_R4 = re.compile(r"\b(?:headroom|ceiling|cap|limit|budget|constraint)\b", re.I)


def check_r4(passes: Mapping[str, str], facts: FactsSheet | None = None) -> list[Finding]:
    del facts  # text-internal rule
    findings: list[Finding] = []
    for pass_id, raw in passes.items():
        text = _strip_fences(_normalize(raw))

        for m in _LEAVING.finditer(text):
            left = _to_mb(_num(m.group("left")), m.group("lu"))
            cap = _to_mb(_num(m.group("cap")), m.group("cu"))
            failures: list[str] = []
            sent_start = _clause_start(text, m.start())
            used_matches = list(_USED_BEFORE.finditer(text, sent_start, m.start()))
            if used_matches:
                used = used_matches[-1]
                total = _to_mb(_num(used.group("v")), used.group("u")) + left
                if _mismatch(total, cap, approx=True):
                    failures.append(
                        f"used {_fmt(_to_mb(_num(used.group('v')), used.group('u')))} MB "
                        f"+ left {_fmt(left)} MB != cap {_fmt(cap)} MB"
                    )
            fp = _FOOTPRINT.search(text, m.end(), m.end() + 120)
            if fp:
                footprint = _to_mb(_num(fp.group("fp")), fp.group("fu"))
                if left + 1e-9 < footprint:
                    failures.append(
                        f"left {_fmt(left)} MB < footprint {_fmt(footprint)} MB"
                    )
            if failures:
                findings.append(
                    Finding(
                        RULE_R4_LEAVING,
                        SEVERITY_ERROR,
                        pass_id,
                        _excerpt(text, m.start(), m.end()),
                        "leaving-of-cap: " + "; ".join(failures),
                    )
                )

        for bare in (False, True):
            pattern = _BARE_CAP if bare else _CTX_CAP
            for m in pattern.finditer(text):
                if "×" in text[max(0, m.start() - 20) : m.start()]:
                    continue  # multiplication operand, not a standalone claim
                op = m.group("op")
                if bare and op in _BARE_OPS:
                    if not _CAP_NOUN_R4.search(_clause(text, m.start(), m.end())):
                        continue
                a = _to_mb(_num(m.group("a")), m.group("ua"))
                b = _to_mb(_num(m.group("b")), m.group("ub"))
                exceeds = a >= b if op == "<" else a > b
                if exceeds:
                    findings.append(
                        Finding(
                            RULE_R4_CAP,
                            SEVERITY_ERROR,
                            pass_id,
                            _excerpt(text, m.start(), m.end()),
                            "component exceeds cap: "
                            f"{m.group('a')} {m.group('ua')} {op} "
                            f"{m.group('b')} {m.group('ub')} is false "
                            f"({_fmt(a)} MB > {_fmt(b)} MB)",
                        )
                    )
    return findings


# ---------------------------------------------------------------------------
# Registry: unknown pass key (warning), missing pass key (info)
# ---------------------------------------------------------------------------


def _registry(passes: Mapping[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    for key in passes:
        if key not in KNOWN_PASS_IDS:
            findings.append(
                Finding(
                    RULE_REGISTRY_UNKNOWN,
                    SEVERITY_WARNING,
                    key,
                    key,
                    f"unknown pass id {key!r}; expected one of {len(KNOWN_PASS_IDS)} known passes",
                )
            )
    for pass_id in KNOWN_PASS_IDS:
        if pass_id not in passes:
            findings.append(
                Finding(
                    RULE_REGISTRY_MISSING,
                    SEVERITY_INFO,
                    pass_id,
                    pass_id,
                    f"pass {pass_id!r} missing from the stored design",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

RULES = (check_r1, check_r2, check_r4)


def run_checks(passes: Mapping[str, str], facts: FactsSheet | None = None) -> list[Finding]:
    """Run every rule plus the registry over stored pass text.

    Pure and deterministic: the same inputs always yield the same findings in
    the same order (sorted by pass id, rule id, excerpt).
    """
    findings: list[Finding] = list(_registry(passes))
    for rule in RULES:
        findings.extend(rule(passes, facts))
    findings.sort(key=lambda f: (f.pass_id, f.rule_id, f.excerpt))
    return findings
