import re
from dataclasses import dataclass
from typing import Any


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_key_value(skill_text: str) -> tuple[str, str]:
    if not skill_text:
        return "", ""

    for line in skill_text.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            return key.strip(), value.strip()
    return "", ""


@dataclass
class GradeResult:
    semantic_match: bool
    quality_pass: bool
    degenerate_flag: bool
    correctness_component: float
    non_degeneracy_component: float
    final_score: float
    scored_correct: bool
    skill_key: str
    skill_value: str


def _bool_match(expected_norm: str, answer_norm: str) -> bool:
    if expected_norm == "true":
        return any(token in answer_norm for token in ["true", "yes"])
    if expected_norm == "false":
        return any(token in answer_norm for token in ["false", "no"])
    return False


def _quality_check(answer: str) -> tuple[bool, bool]:
    """
    Returns (quality_pass, degenerate_flag).
    quality_pass is True if output looks reasonably coherent.
    degenerate_flag is True when output looks repetitive/garbled.
    """
    raw = answer.strip()
    norm = normalize(raw)
    if not norm:
        return False, True

    tokens = re.findall(r"[a-z0-9_]+", norm)
    token_count = len(tokens)
    if token_count == 0:
        return False, True

    alpha_token_count = sum(1 for tok in tokens if re.search(r"[a-z]", tok))
    digit_token_count = sum(1 for tok in tokens if tok.isdigit())
    alpha_token_ratio = alpha_token_count / token_count
    digit_token_ratio = digit_token_count / token_count

    # Repetition signals
    unique_ratio = len(set(tokens)) / token_count
    max_run = 1
    run = 1
    for i in range(1, token_count):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1

    # Non-ascii noise ratio
    non_ascii = sum(1 for ch in raw if ord(ch) > 127)
    non_ascii_ratio = non_ascii / max(len(raw), 1)
    digit_chars = sum(1 for ch in raw if ch.isdigit())
    alpha_chars = sum(1 for ch in raw if ch.isalpha())
    digit_char_ratio = digit_chars / max(len(raw), 1)
    alpha_char_ratio = alpha_chars / max(len(raw), 1)

    degenerate = False
    # Repetitive token loops.
    if token_count >= 4 and unique_ratio <= 0.35:
        degenerate = True
    if max_run >= 4:
        degenerate = True

    # Numeric-junk heavy outputs: catches things like "014 32060 03333 3352".
    if token_count >= 4 and alpha_token_count == 0:
        degenerate = True
    if token_count >= 4 and digit_token_ratio >= 0.60 and alpha_token_ratio <= 0.25:
        degenerate = True
    if token_count >= 4 and digit_char_ratio >= 0.45 and alpha_char_ratio <= 0.25:
        degenerate = True
    if token_count >= 5 and digit_token_count >= 2 and alpha_token_count <= 3:
        degenerate = True

    # Extremely short outputs are allowed, but multiple bare numeric fragments are suspicious.
    if token_count <= 3 and alpha_token_count == 0 and digit_token_count >= 2:
        degenerate = True

    if non_ascii_ratio > 0.15:
        degenerate = True

    return (not degenerate), degenerate


def grade_answer(answer: str, expected: str, skill_text: str = "") -> GradeResult:
    norm_answer = normalize(answer)
    norm_expected = normalize(expected)

    skill_key, skill_value = parse_key_value(skill_text)
    if norm_expected in {"true", "false"}:
        semantic_match = _bool_match(norm_expected, norm_answer)
    else:
        semantic_match = norm_expected in norm_answer
    quality_pass, degenerate_flag = _quality_check(answer)
    correctness_component = 0.5 if semantic_match else 0.0
    non_degeneracy_component = 0.5 if quality_pass else 0.0
    final_score = correctness_component + non_degeneracy_component
    # "Correctness" should reflect answer correctness only, not full-credit quality.
    scored_correct = semantic_match

    return GradeResult(
        semantic_match=semantic_match,
        quality_pass=quality_pass,
        degenerate_flag=degenerate_flag,
        correctness_component=correctness_component,
        non_degeneracy_component=non_degeneracy_component,
        final_score=final_score,
        scored_correct=scored_correct,
        skill_key=skill_key,
        skill_value=skill_value,
    )


def build_results_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    total_questions = len(items)
    total_score = float(sum(float(row.get("final_score", 0.0)) for row in items))
    correctness_score = float(sum(float(row.get("correctness_component", 0.0)) for row in items))
    non_degeneracy_score = float(sum(float(row.get("non_degeneracy_component", 0.0)) for row in items))
    payload = {
        "final_score_out_of_100": total_score,
        "correctness_out_of_50": correctness_score,
        "non_degeneracy_out_of_50": non_degeneracy_score,
        "total_questions": total_questions,
        "items": items,
    }
    return payload
