"""Structured/pipe response parsing with explicit unresolved-citation records."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from dvcf.interfaces import Edge


def strip_wrappers(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.fullmatch(r"```(?:json|python|text)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def json_candidates(text: str) -> tuple[str, ...]:
    """Extract balanced JSON containers without counting braces inside strings."""
    text = strip_wrappers(text)
    candidates: list[str] = [text]
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        stack: list[str] = []
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack or (stack[-1], char) not in (("[", "]"), ("{", "}")):
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start:index + 1])
                    break
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate.strip()))


def _parse_value(candidate: str) -> Any:
    attempts = [candidate, re.sub(r",\s*([}\]])", r"\1", candidate)]
    attempts.append(re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', attempts[-1]))
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
    try:
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError) as error:
        raise ValueError("Response contains no parseable structured value.") from error


def parse_json_object(text: str) -> dict[str, Any]:
    for candidate in json_candidates(text):
        try:
            value = _parse_value(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Expected a JSON object in the model response.")


def normalize_variable_name(name: str) -> str:
    name = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", name.strip())
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def clean_evidence_label(label: str) -> str:
    label = label.strip().strip("[](){}")
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", label).upper()


def evidence_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(label).strip() for label in value if str(label).strip())
    text = str(value).strip()
    if text.upper() in {"", "NONE", "N/A", "NO EVIDENCE"}:
        return ()
    text = re.sub(r"^evidence\s*:\s*", "", text, flags=re.IGNORECASE).strip("[](){}")
    return tuple(part for part in re.split(r"[;,|\s]+", text) if part)


@dataclass(frozen=True)
class PairJudgment:
    var_a: str
    var_b: str
    score_a_to_b: float
    score_b_to_a: float
    reason: str
    raw_labels: tuple[str, ...]
    valid_labels: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    evidence_map: Mapping[str, str]


def _score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Directional scores must be numeric.")
    try:
        score = float(value)
    except (ValueError, TypeError):
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", str(value))
        if not match:
            raise ValueError("Directional score is missing or invalid.")
        score = float(match.group())
    if not isfinite(score):
        raise ValueError("Directional scores must be finite.")
    return max(0.0, min(1.0, score))


def _field(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise ValueError(f"Missing response field: {names[0]}.")


def _response_records(text: str) -> list[dict[str, Any]]:
    for candidate in json_candidates(text):
        try:
            value = _parse_value(candidate)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("judgments"), list):
            return value["judgments"]
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
        if isinstance(value, dict) and any(key in value for key in ("var_A", "var_a", "A", "source")):
            # JSON-lines responses may contain more than one such object.
            records = []
            for line in strip_wrappers(text).splitlines():
                try:
                    row = _parse_value(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and any(key in row for key in ("var_A", "var_a", "A", "source")):
                    records.append(row)
            return records or [value]
    rows: list[dict[str, Any]] = []
    for line in strip_wrappers(text).splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip("|").strip()
        if not line or re.fullmatch(r"[-:|\s]+", line):
            continue
        parts = [value.strip() for value in line.split("|")]
        if len(parts) < 6:
            continue
        if parts[0].lower() in {"var_a", "variable a"} and "score" in parts[2].lower():
            continue
        rows.append(dict(var_a=parts[0], var_b=parts[1], score_a_to_b=parts[2],
                         score_b_to_a=parts[3], reason=parts[4], evidence="|".join(parts[5:])))
    return rows


def parse_pair_judgments(
    response: str, *, expected_pairs: Sequence[Edge], aliases: Mapping[str, Sequence[str]],
    evidence_map: Mapping[str, str]
) -> tuple[PairJudgment, ...]:
    """Parse all expected pairs, reorient reversed rows, and retain unknown labels.

    Missing/conflicting rows raise; parsing failure never manufactures zero scores.
    """
    if any(a == b for a, b in expected_pairs) or len({frozenset(pair) for pair in expected_pairs}) != len(expected_pairs):
        raise ValueError("Expected pairs must be distinct non-self unordered pairs.")
    variables = tuple(dict.fromkeys(variable for pair in expected_pairs for variable in pair))
    alias_map: dict[str, str] = {}
    for variable in variables:
        for alias in (variable, *aliases.get(variable, ())):
            normalized = normalize_variable_name(alias)
            if not normalized or (normalized in alias_map and alias_map[normalized] != variable):
                raise ValueError("Variable aliases must be non-empty and unambiguous.")
            alias_map[normalized] = variable
    expected = {frozenset(pair): pair for pair in expected_pairs}
    available = {clean_evidence_label(key): value for key, value in evidence_map.items()}
    if len(available) != len(evidence_map) or any(not key or not value for key, value in available.items()):
        raise ValueError("Evidence labels must resolve to distinct non-empty identifiers.")
    parsed: dict[Edge, PairJudgment] = {}
    for row in _response_records(response):
        if not isinstance(row, dict):
            raise ValueError("Each judgment must be an object.")
        a = alias_map.get(normalize_variable_name(str(_field(row, ("var_A", "var_a", "A", "source")))))
        b = alias_map.get(normalize_variable_name(str(_field(row, ("var_B", "var_b", "B", "target")))))
        if a is None or b is None or frozenset((a, b)) not in expected:
            raise ValueError("Response contains an unexpected variable pair.")
        canonical = expected[frozenset((a, b))]
        forward = _score(_field(row, ("score_A_to_B", "score_a_to_b", "forward", "a_to_b", "A_to_B", "A_to_B_score")))
        reverse = _score(_field(row, ("score_B_to_A", "score_b_to_a", "backward", "b_to_a", "B_to_A", "B_to_A_score")))
        if (a, b) != canonical:
            forward, reverse = reverse, forward
        evidence_value = row.get("evidence", row.get("evidence_labels", row.get("labels",
                             row.get("A_to_B_evidence", row.get("forward_evidence", "")))))
        reason = row.get("reason", row.get("A_to_B_reason", row.get("forward_reason", "")))
        raw = evidence_labels(evidence_value)
        cleaned = tuple(clean_evidence_label(label) for label in raw)
        valid = tuple(label for label in cleaned if label in available)
        unknown = tuple(label for label in cleaned if label not in available)
        judgment = PairJudgment(*canonical, forward, reverse, str(reason),
                                raw, valid, unknown, {label: available[label] for label in valid})
        if canonical in parsed and parsed[canonical] != judgment:
            raise ValueError("Response contains conflicting judgments for the same pair.")
        parsed[canonical] = judgment
    missing = [pair for pair in expected_pairs if pair not in parsed]
    if missing:
        raise ValueError(f"Missing judgments for {len(missing)} requested pair(s).")
    return tuple(parsed[pair] for pair in expected_pairs)
