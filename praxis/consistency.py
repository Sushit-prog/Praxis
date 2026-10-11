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
RULE_R3 = "r3-dependency"
RULE_R5 = "r5-metric"
RULE_R8 = "r8-prd-sections"
RULE_R9 = "r9-task-structure"
RULE_R13 = "r13-rules-categories"
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


def check_r1(
    passes: Mapping[str, str], facts: FactsSheet | None = None, **_ignored: object
) -> list[Finding]:
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


def check_r2(
    passes: Mapping[str, str], facts: FactsSheet | None = None, **_ignored: object
) -> list[Finding]:
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


def check_r4(
    passes: Mapping[str, str], facts: FactsSheet | None = None, **_ignored: object
) -> list[Finding]:
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
# R3 (warning): two passes prescribe different packages for the same slot
# ---------------------------------------------------------------------------

# Curated software slots (v1): each maps a slot to the packages that can fill
# it. Only backticked tokens count, so the English word "requests" ("queue
# requests") can never match the HTTP-client slot.
_DEPENDENCY_SLOTS: dict[str, frozenset[str]] = {
    "http client": frozenset({"requests", "httpx", "urllib3", "aiohttp"}),
    "llm orchestration": frozenset({"langgraph", "langchain"}),
    "embeddings": frozenset({"sentence-transformers", "fastembed"}),
}
_BACKTICK = re.compile(r"`([^`\n]+)`")


def _check_r3(passes: Mapping[str, str]) -> list[Finding]:
    occurrences: dict[str, dict[str, set[str]]] = {
        slot: {} for slot in _DEPENDENCY_SLOTS
    }
    excerpts: dict[tuple[str, str], str] = {}
    for pass_id, raw in passes.items():
        text = _strip_fences(_normalize(raw))
        for m in _BACKTICK.finditer(text):
            token = m.group(1).strip().lower().replace("_", "-")
            for slot, packages in _DEPENDENCY_SLOTS.items():
                if token in packages:
                    occurrences[slot].setdefault(token, set()).add(pass_id)
                    excerpts.setdefault((slot, token), _excerpt(text, m.start(), m.end()))
    findings: list[Finding] = []
    for slot, by_package in occurrences.items():
        packages = sorted(by_package)
        # A conflicting pair: two different packages that never appear in the
        # same pass. Co-presence in one pass = acknowledged comparison.
        pair: tuple[str, str] | None = None
        for i, a in enumerate(packages):
            for b in packages[i + 1 :]:
                if by_package[a].isdisjoint(by_package[b]):
                    pair = (a, b)
                    break
            if pair:
                break
        if pair is None:
            continue
        a, b = pair
        anchor = min(by_package[a] | by_package[b])
        findings.append(
            Finding(
                RULE_R3,
                SEVERITY_WARNING,
                anchor,
                excerpts[(slot, a)],
                f"dependency conflict: pass {sorted(by_package[a])[0]!r} requires "
                f"{a!r} ({slot}), pass {sorted(by_package[b])[0]!r} requires "
                f"{b!r}; the two passes never mention each other's package",
            )
        )
    return findings


def check_r3(
    passes: Mapping[str, str], facts: FactsSheet | None = None, **_ignored: object
) -> list[Finding]:
    del facts  # text-internal rule
    try:
        return _check_r3(passes)
    except Exception:  # noqa: BLE001 - a broken rule must stay silent, not crash check
        return []


# ---------------------------------------------------------------------------
# R5 (warning): two passes prescribe different similarity metrics for
# vector retrieval
# ---------------------------------------------------------------------------

_METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cosine", re.compile(r"\bcosine\b", re.I)),
    ("euclidean", re.compile(r"\beuclidean\b|\bL2\s+(?:distance|metric)\b", re.I)),
    ("dot product", re.compile(r"\bdot[\s-]product\b", re.I)),
    ("manhattan", re.compile(r"\bmanhattan\b", re.I)),
)
# Context that makes a metric claim about vector retrieval, as opposed to a
# dedup threshold or a norm note ("L2 norm ~ 1.0"). Bare "similarity" is
# deliberately absent.
_RETRIEVAL_CTX = re.compile(
    r"k[\s‐\-‑–—]?nn|nearest|retriev|vector\s+search|"
    r"over\s+(?:all\s+)?(?:stored\s+)?embeddings",
    re.I,
)

# A period inside a backticked identifier (`records.embedding`) is not a
# clause boundary; mask those dots 1:1 so excerpts stay position-accurate.
_DOT_MASK = "\x01"
_BACKTICK_SPAN = re.compile(r"`[^`\n]+`")


def _mask_backtick_dots(text: str) -> str:
    return _BACKTICK_SPAN.sub(lambda m: m.group(0).replace(".", _DOT_MASK), text)


def _primary_metric(text: str) -> tuple[str, int, int] | None:
    """First retrieval-context metric claim in document order, if any."""
    pos = 0
    for clause in _BOUNDARY.split(text):
        clause_start = pos
        pos += len(clause) + 1  # +1 for the split delimiter
        if not _RETRIEVAL_CTX.search(clause):
            continue
        best: tuple[int, str, int, int] | None = None
        for metric, pattern in _METRIC_PATTERNS:
            m = pattern.search(clause)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), metric, clause_start + m.start(), clause_start + m.end())
        if best:
            return best[1], best[2], best[3]
    return None


def _check_r5(passes: Mapping[str, str]) -> list[Finding]:
    by_metric: dict[str, list[str]] = {}
    excerpts: dict[str, str] = {}
    primary_pass: dict[str, str] = {}
    for pass_id, raw in passes.items():
        # Unlike r1/r2/r4, r5 deliberately does NOT strip fenced blocks: the
        # prescribed metric usually lives in the pseudocode fence, which is
        # still a prescription, not a mere snippet.
        text = _normalize(raw)
        primary = _primary_metric(_mask_backtick_dots(text))
        if primary is None:
            continue
        metric, start, end = primary
        primary_pass[pass_id] = metric
        by_metric.setdefault(metric, []).append(pass_id)
        excerpts.setdefault(pass_id, _excerpt(text, start, end).replace(_DOT_MASK, "."))
    if len(by_metric) < 2:
        return []
    anchor_pass = min(primary_pass)
    anchor_metric = primary_pass[anchor_pass]
    findings: list[Finding] = []
    for metric in sorted(by_metric):
        if metric == anchor_metric:
            continue
        pass_id = min(by_metric[metric])
        anchor_others = [p for p in sorted(by_metric[anchor_metric]) if p != anchor_pass]
        detail = f"but pass {anchor_pass!r} prescribes '{anchor_metric}'"
        if anchor_others:
            detail += f" (also: {', '.join(anchor_others)})"
        findings.append(
            Finding(
                RULE_R5,
                SEVERITY_WARNING,
                pass_id,
                excerpts[pass_id],
                f"metric conflict: pass {pass_id!r} prescribes '{metric}' for "
                f"vector retrieval, {detail}",
            )
        )
    return findings


def check_r5(
    passes: Mapping[str, str], facts: FactsSheet | None = None, **_ignored: object
) -> list[Finding]:
    del facts  # text-internal rule
    try:
        return _check_r5(passes)
    except Exception:  # noqa: BLE001 - a broken rule must stay silent, not crash check
        return []


# ---------------------------------------------------------------------------
# R8/R9/R13 (warning on stamped v0.2 packs, info on legacy packs): content
# shape of the PRD, the plan's tasks, and the RULES categories. One finding
# per rule, listing every missing item; never an error in v0.2.0.
# ---------------------------------------------------------------------------


def _pass_text(passes: Mapping[str, str], pass_id: str) -> str | None:
    """The pass's text, or None when the pass is absent/unparseable/blank."""
    raw = passes.get(pass_id)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw


def _norm_label(text: str) -> str:
    """Casefold a heading/label; hyphens, underscores and & become spaces."""
    text = text.translate(str.maketrans({"-": " ", "_": " ", "&": " ", "–": " ", "—": " "}))
    return " ".join(text.casefold().split())


def _syn_re(synonym: str) -> re.Pattern[str]:
    """Whole-word match for a synonym, accepting simple plurals per word."""
    words = r"\s+".join(re.escape(w) + r"s?" for w in synonym.split())
    return re.compile(rf"\b{words}\b", re.I)


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")


def _headings(text: str) -> list[str]:
    """Normalized heading labels of a fence-stripped document, in order."""
    labels: list[str] = []
    for line in _strip_fences(_normalize(text)).splitlines():
        m = _HEADING.match(line)
        if m and m.group(1).strip():
            labels.append(_norm_label(m.group(1)))
    return labels


def _has_heading(headings: list[str], synonym: str) -> bool:
    pattern = _syn_re(synonym)
    return any(pattern.search(h) for h in headings)


def _content_finding(
    rule_id: str,
    pass_id: str,
    excerpt: str,
    missing: list[str],
    *,
    stamped: bool,
    item: str,
) -> Finding:
    joined = ", ".join(missing)
    if stamped:
        severity = SEVERITY_WARNING
        message = f"missing {joined} (required in v0.2 packs)"
    else:
        severity = SEVERITY_INFO
        message = f"legacy pack: missing {joined} (v0.2 packs require it)"
    return Finding(rule_id, severity, pass_id, excerpt, f"{item}: {message}")


# Each section maps its display label to the synonym list; a heading matches
# when any synonym appears as whole words (simple plurals accepted).
_R8_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Problem", ("problem",)),
    ("Users", ("users", "target users")),
    ("Goal", ("goal",)),
    ("Features", ("features", "core features", "functional requirements")),
    ("MVP", ("mvp",)),
    ("Out of scope", ("out of scope", "non goals")),
    ("Success criteria", ("success criteria", "success metrics")),
)


def check_r8(
    passes: Mapping[str, str],
    facts: FactsSheet | None = None,
    *,
    stamped: bool = False,
    **_ignored: object,
) -> list[Finding]:
    del facts  # text-internal rule
    text = _pass_text(passes, "prd")
    if text is None:
        return []
    try:
        headings = _headings(text)
        missing = [
            label
            for label, syns in _R8_SECTIONS
            if not any(_has_heading(headings, s) for s in syns)
        ]
        if not missing:
            return []
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "prd")
        return [
            _content_finding(
                RULE_R8, "prd", first[:120], missing, stamped=stamped, item="prd sections"
            )
        ]
    except Exception:  # noqa: BLE001 - a broken rule must stay silent, not crash check
        return []


# The plan pass is where TASK-NNN tasks live (TASKS.md is a render of it).
_TASK_ID = re.compile(r"TASK-(?P<id>\d+)", re.I)
_ACCEPTANCE_WINDOW = 3
_ACCEPTANCE = re.compile(
    r"\s*(?:[-+•]\s*)?accept(?:s|ance)?(?:\s*:|\s*$)", re.I
)


def _is_acceptance_line(line: str) -> bool:
    stripped = line.replace("*", "").replace("_", "").replace("`", "")
    return bool(_ACCEPTANCE.match(stripped))


def check_r9(
    passes: Mapping[str, str],
    facts: FactsSheet | None = None,
    *,
    stamped: bool = False,
    **_ignored: object,
) -> list[Finding]:
    del facts  # text-internal rule
    text = _pass_text(passes, "plan")
    if text is None:
        return []
    try:
        lines = _strip_fences(_normalize(text)).splitlines()
        tasks: list[tuple[str, int]] = []
        for i, line in enumerate(lines):
            m = _TASK_ID.search(line)
            if m:
                tasks.append((f"TASK-{m.group('id')}", i))
        if not tasks:
            if not stamped:
                return []
            return [
                Finding(
                    RULE_R9,
                    SEVERITY_WARNING,
                    "plan",
                    "plan",
                    "task structure: no TASK-NNN ids in plan "
                    "(required in v0.2 packs)",
                )
            ]
        uncovered = [
            tid
            for tid, i in tasks
            if not any(
                _is_acceptance_line(ln)
                for ln in lines[i + 1 : i + 1 + _ACCEPTANCE_WINDOW]
            )
        ]
        if not uncovered:
            return []
        excerpt = next(
            (ln.strip()[:120] for ln in lines if _TASK_ID.search(ln)), "plan"
        )
        return [
            _content_finding(
                RULE_R9,
                "plan",
                excerpt,
                uncovered,
                stamped=stamped,
                item=(
                    f"tasks without an Acceptance marker within "
                    f"{_ACCEPTANCE_WINDOW} lines"
                ),
            )
        ]
    except Exception:  # noqa: BLE001 - a broken rule must stay silent, not crash check
        return []


_R13_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("General", ("general",)),
    ("Before coding", ("before coding", "before you code")),
    ("Security", ("security",)),
    ("Testing", ("testing",)),
    ("Git", ("git",)),
)


def check_r13(
    passes: Mapping[str, str],
    facts: FactsSheet | None = None,
    *,
    stamped: bool = False,
    **_ignored: object,
) -> list[Finding]:
    del facts  # text-internal rule
    text = _pass_text(passes, "rules")
    if text is None:
        return []
    try:
        headings = _headings(text)
        missing = [
            label
            for label, syns in _R13_CATEGORIES
            if not any(_has_heading(headings, s) for s in syns)
        ]
        if not missing:
            return []
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "rules")
        return [
            _content_finding(
                RULE_R13, "rules", first[:120], missing, stamped=stamped, item="rules categories"
            )
        ]
    except Exception:  # noqa: BLE001 - a broken rule must stay silent, not crash check
        return []


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

RULES = (
    check_r1,
    check_r2,
    check_r4,
    check_r3,
    check_r5,
    check_r8,
    check_r9,
    check_r13,
)


def run_checks(
    passes: Mapping[str, str],
    facts: FactsSheet | None = None,
    *,
    stamped: bool = False,
) -> list[Finding]:
    """Run every rule plus the registry over stored pass text.

    Pure and deterministic: the same inputs always yield the same findings in
    the same order (sorted by pass id, rule id, excerpt). ``stamped`` marks a
    v0.2 pack (both design and pack layers stamped); content rules r8/r9/r13
    warn on stamped packs and emit legacy-pack info otherwise.
    """
    findings: list[Finding] = list(_registry(passes))
    for rule in RULES:
        findings.extend(rule(passes, facts, stamped=stamped))
    findings.sort(key=lambda f: (f.pass_id, f.rule_id, f.excerpt))
    return findings
