from pathlib import Path
import re

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_model_source(model_id: str = MODEL_ID) -> str:
    """
    Prefer a local Hugging Face snapshot path when available.
    This avoids network metadata calls in sandboxed/offline runs.
    """
    home = Path.home()
    cache_root = home / ".cache" / "huggingface" / "hub"
    cache_key = model_id.replace("/", "--")
    model_cache_dir = cache_root / f"models--{cache_key}" / "snapshots"
    if model_cache_dir.exists():
        snapshots = sorted([p for p in model_cache_dir.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1])
    return model_id

QUESTION_OVERRIDES = {
    "retry_budget": "What is the retry budget?",
    "request_timeout_seconds": "What is the request timeout in seconds?",
    "backoff_strategy": "What backoff strategy is required?",
    "auth_scheme": "Which authentication scheme should be used?",
    "log_level": "What log level should the agent use?",
    "max_requests_per_minute": "What is the max requests per minute limit?",
    "cache_ttl_seconds": "What is the cache TTL in seconds?",
    "tool_retry_on_429": "Should the tool retry on HTTP 429?",
    "error_response_format": "What error response format is required?",
    "forbidden_action": "What action is explicitly forbidden?",
}


def _parse_key_value(skill_path: Path) -> tuple[str, str]:
    for raw_line in skill_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            return key.strip(), value.strip()
    raise ValueError(f"Missing key:value pair in skill file: {skill_path}")


def _skill_sort_key(skill_path: Path) -> tuple[int, str]:
    m = re.search(r"skill_(\d+)", skill_path.name)
    idx = int(m.group(1)) if m else 10**9
    return (idx, skill_path.name)


def _question_for_key(key: str) -> str:
    if key in QUESTION_OVERRIDES:
        return QUESTION_OVERRIDES[key]
    human = key.replace("_", " ")
    return f"What is the value for {human}?"


def load_cases(skills_dir: str = "skills") -> list[dict[str, str]]:
    root = Path(skills_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing skills directory: {root}")

    skill_files = sorted(root.glob("skill_*.md"), key=_skill_sort_key)
    if not skill_files:
        raise FileNotFoundError(f"No skill files found under: {root}")

    cases: list[dict[str, str]] = []
    for skill_path in skill_files:
        key, value = _parse_key_value(skill_path)
        cases.append(
            {
                "question": _question_for_key(key),
                "expected": value,
                "skill_file": str(skill_path),
            }
        )
    return cases


CASES = load_cases()
