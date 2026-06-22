"""Tiny Nebius (OpenAI-compatible) JSON wrapper for the support-triage agent.

Single entry point: ``call_json(system_prompt, user_message, schema_keys)``.
Returns a dict whose keys are a superset of ``schema_keys``. Always parses
the model's text output as JSON, with a single retry on parse failure.

Provider
--------
Uses Nebius AI Studio via its OpenAI-compatible REST API. Configured via:

- ``NEBIUS_API_KEY`` (required) — Nebius API key.
- ``NEBIUS_BASE_URL`` (optional) — defaults to ``https://api.studio.nebius.com/v1/``.

We deliberately use the upstream ``openai`` SDK (already installed) pointed
at the Nebius base URL. This keeps the surface tiny and provider-portable
— swapping back to Anthropic, Together, Groq, or vLLM is a one-line change.

Design notes
------------
- Determinism: ``temperature=0`` everywhere. No top-p tuning. Same input
  → same output (modulo provider-side nondeterminism we can't control).
- Retry: exactly one retry on JSON parse / missing-key validation. The
  retry reformulates the user turn with the parser's error and the
  explicit schema keys required. API/network/auth/rate-limit errors are
  not retried — they propagate.
- Caching: Nebius applies automatic prefix caching on supported models
  (no explicit cache_control directive). We keep ``cache_system`` as a
  no-op flag so existing callers don't break.
- Logging: when ``LOG_LLM_USAGE`` is truthy in the environment, a single
  stderr line is printed per call with model and prompt/completion/total
  tokens. No prompt content ever leaves the process.

CLI is intentionally not provided — this module is purely a helper.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Auto-load `.env` at module import so any caller (CLI or library) sees the
# vars without explicit setup. Idempotent and silently no-ops if the file
# is missing or python-dotenv is unavailable.
def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_dotenv_if_present()

# Default model picked for triage / specialist / rerank: strong generalist
# with reliable JSON adherence on Nebius.
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

DEFAULT_BASE_URL = "https://api.studio.nebius.ai/v1/"
LOG_USAGE_ENV = "LOG_LLM_USAGE"

# Lazy-initialized module-level singleton.
_CLIENT: Any = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised on missing API key or repeated JSON-parse failure.

    Attributes
    ----------
    raw : str
        The most recent raw text returned by the model (empty on init/auth
        failures before any call was made).
    attempts : int
        How many model calls were issued before raising (0 if the failure
        happened before any call).
    """

    def __init__(self, message: str, *, raw: str = "", attempts: int = 0):
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def get_client():
    """Return a process-wide ``openai.OpenAI`` singleton pointed at Nebius.

    Reads ``NEBIUS_API_KEY`` from the environment; ``NEBIUS_BASE_URL``
    overrides the default endpoint. Raises ``LLMError`` if the key is
    missing or the SDK can't be initialized.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise LLMError(
            "NEBIUS_API_KEY is not set. Export it (or put it in .env) "
            "before invoking the LLM."
        )

    base_url = os.environ.get("NEBIUS_BASE_URL", "").strip() or DEFAULT_BASE_URL

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise LLMError(
            "openai SDK not installed. Run `pip install openai`."
        ) from exc

    try:
        _CLIENT = OpenAI(base_url=base_url, api_key=api_key)
    except Exception as exc:
        raise LLMError(f"failed to construct Nebius client: {exc}") from exc
    return _CLIENT


# ---------------------------------------------------------------------------
# JSON parsing (lenient)
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*\n", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")


def _parse_json_lenient(raw: str) -> dict:
    """Best-effort JSON parse.

    Strips ``` json fences if present, tries ``json.loads``; on failure
    extracts the first balanced ``{...}`` block (string-aware brace
    counting) and retries. Raises ``ValueError`` if nothing parses to a
    dict.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response from model")

    stripped = _FENCE_OPEN_RE.sub("", text, count=1)
    stripped = _FENCE_CLOSE_RE.sub("", stripped).strip()

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None

    if obj is None:
        start = stripped.find("{")
        if start == -1:
            raise ValueError("no JSON object found in model response")
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            raise ValueError("unbalanced braces in model response")
        try:
            obj = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"json parse failed: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    return obj


# ---------------------------------------------------------------------------
# Usage logging
# ---------------------------------------------------------------------------


def _log_usage(resp: Any, model: str) -> None:
    """Emit a single stderr line summarising token usage.

    No-op unless ``LOG_LLM_USAGE`` is truthy in the environment. Never
    logs prompt content.
    """
    flag = os.environ.get(LOG_USAGE_ENV, "").strip().lower()
    if flag in ("", "0", "false", "no", "off"):
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)
    print(
        f"llm model={model} prompt={prompt} completion={completion} total={total}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _extract_text(resp: Any) -> str:
    """Pull the assistant content text out of an OpenAI-shaped response."""
    try:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return ""
        content = getattr(msg, "content", "") or ""
        return content.strip()
    except Exception:
        return ""


def _validate_schema(obj: dict, schema_keys: list[str]) -> None:
    missing = [k for k in schema_keys if k not in obj]
    if missing:
        raise ValueError(f"missing required keys: {missing}")


def call_json(
    system_prompt: str,
    user_message: str,
    schema_keys: list[str],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    cache_system: bool = True,  # accepted for API compat; Nebius auto-caches.
    client: Any = None,
) -> dict:
    """Send a JSON-only chat turn and return the parsed dict.

    Parameters
    ----------
    system_prompt:
        Sent as the system role message. Nebius applies automatic prefix
        caching on supported models, so reuse across rows is still
        cost-efficient even though we don't pass a ``cache_control`` flag.
    user_message:
        Sent as the user role message.
    schema_keys:
        Keys that must be present in the parsed response. Extras are
        allowed.
    model, max_tokens, cache_system:
        ``cache_system`` is a no-op flag retained for backwards-compatible
        call sites. The other two pass through to the API.
    client:
        Optional pre-built OpenAI-compatible client (handy for tests).

    Raises
    ------
    LLMError
        If JSON parsing or schema validation fails after the retry, or if
        the API key is missing.
    """
    cli = client if client is not None else get_client()

    user_turn = user_message
    last_raw = ""
    last_err: Exception | None = None

    for attempt in (1, 2):
        resp = cli.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_turn},
            ],
        )
        _log_usage(resp, model)
        last_raw = _extract_text(resp)
        try:
            obj = _parse_json_lenient(last_raw)
            _validate_schema(obj, schema_keys)
            return obj
        except ValueError as exc:
            last_err = exc
            if attempt == 2:
                break
            user_turn = (
                f"{user_message}\n\nYour previous response was not valid JSON. "
                f"Error: {exc}. Reply with ONLY a JSON object containing keys: "
                f"{schema_keys}. No prose, no markdown fences."
            )

    raise LLMError(
        f"LLM returned invalid JSON after retry: {last_err}",
        raw=last_raw,
        attempts=2,
    )


# ---------------------------------------------------------------------------
# Self-check (run as `python code/llm.py`)
# ---------------------------------------------------------------------------


def _self_check() -> int:
    """Smoke test: confirm get_client raises a clear LLMError without a key."""
    global _CLIENT
    _CLIENT = None
    saved = os.environ.pop("NEBIUS_API_KEY", None)
    try:
        try:
            get_client()
        except LLMError as exc:
            print(f"OK: get_client() raised LLMError as expected: {exc}", file=sys.stderr)
            return 0
        print("FAIL: get_client() did not raise without NEBIUS_API_KEY", file=sys.stderr)
        return 2
    finally:
        if saved is not None:
            os.environ["NEBIUS_API_KEY"] = saved


if __name__ == "__main__":
    sys.exit(_self_check())
