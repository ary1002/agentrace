"""Async judge client over litellm — single entry point for LLM-as-judge metrics."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)

_SCHEMA_SUFFIX = (
    "Respond ONLY with a JSON object matching this schema. No prose, no markdown "
    "fences, no explanation outside the JSON object."
)

_DEFAULT_SYSTEM = (
    "You are an expert LLM agent evaluator. You evaluate agent execution "
    "traces and provide structured assessments. Always respond with valid JSON only."
)

_RETRY_HINT = (
    "\n\nYour previous response could not be parsed as JSON. Respond ONLY "
    "with a valid JSON object."
)


def _load_litellm():
    try:
        import litellm
        from litellm.exceptions import APIError, RateLimitError
    except ImportError as e:
        raise ImportError(
            "litellm is required for LLM-as-judge metrics. "
            "Install with: pip install agentrace[judge]"
        ) from e
    return litellm, APIError, RateLimitError


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    t = t.removeprefix("```json")
    t = t.removeprefix("```")
    t = t.removesuffix("```")
    return t.strip()


def _message_content_as_text(content: Any) -> str:
    """Normalize ``choice.message.content`` (str or provider-specific list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if t is None and isinstance(block.get("content"), str):
                    t = block.get("content")
                if t is not None:
                    parts.append(str(t))
                elif block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _schema_expects_list_value(response_schema: dict[str, Any]) -> bool:
    for v in response_schema.values():
        if v is list:
            return True
        if isinstance(v, str) and "list" in v.lower():
            return True
    return False


def _flatten_scalar_fields(obj: Any, out: dict[str, Any], depth: int = 0) -> None:
    """Collect string-keyed scalar / non-object list values; skip list-of-dict arrays."""
    if depth > 16:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict):
                _flatten_scalar_fields(v, out, depth + 1)
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    continue
                out[k] = v
            else:
                out[k] = v
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                _flatten_scalar_fields(it, out, depth)


def _normalize_key(s: str) -> str:
    return str(s).lower().strip().replace(" ", "_").replace("-", "_")


_SCORE_KEY_ALIASES: tuple[str, ...] = (
    "score",
    "quality_score",
    "quality",
    "rating",
    "value",
    "overall_score",
    "evaluation_score",
    "numeric_score",
    "score_value",
    "confidence",
)

_REASONING_KEY_ALIASES: tuple[str, ...] = (
    "reasoning",
    "explanation",
    "rationale",
    "summary",
    "comment",
    "analysis",
    "justification",
    "notes",
)

_SUGGESTED_FIX_ALIASES: tuple[str, ...] = (
    "suggested_fix",
    "fix",
    "recommendation",
    "action",
    "remediation",
)


def _lookup_flat(flat: dict[str, Any], *aliases: str) -> Any:
    norm_to_val: dict[str, Any] = {}
    for k, v in flat.items():
        norm_to_val[_normalize_key(k)] = v
    for a in aliases:
        nk = _normalize_key(a)
        if nk in norm_to_val:
            return norm_to_val[nk]
    return None


def _fill_required_from_flat_aliases(
    flat: dict[str, Any], response_schema: dict[str, Any]
) -> dict[str, Any]:
    """Map common alternate key names onto canonical schema keys."""
    out: dict[str, Any] = dict(flat)
    required = set(response_schema.keys())
    for req in required:
        if req in out and out[req] is not None and out[req] != "":
            continue
        v: Any = None
        if req == "score":
            v = _lookup_flat(flat, *_SCORE_KEY_ALIASES)
        elif req == "reasoning":
            v = _lookup_flat(flat, *_REASONING_KEY_ALIASES)
        elif req == "suggested_fix":
            v = _lookup_flat(flat, *_SUGGESTED_FIX_ALIASES)
        else:
            v = _lookup_flat(flat, req)
        if v is not None:
            out[req] = v
    return out


def _coerce_parsed_for_schema(parsed: Any, response_schema: dict[str, Any]) -> dict[str, Any]:
    """Return a flat dict that contains all ``response_schema`` keys.

    Models often wrap the payload (e.g. ``{{"result": {{"score": ...}}}}``),
    return a one-element list, or split fields between the root and a nested
    object. Unwrap/merge before key validation.
    """
    required = set(response_schema.keys())

    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            raise JudgeParseError("expected a JSON object or a single-element array of objects")

    if not isinstance(parsed, dict):
        raise JudgeParseError("expected JSON object")

    if required <= parsed.keys():
        return parsed

    for v in parsed.values():
        if isinstance(v, dict) and required <= v.keys():
            return dict(v)

    merged: dict[str, Any] = dict(parsed)
    for v in parsed.values():
        if isinstance(v, dict):
            merged.update(v)
    if required <= merged.keys():
        return merged

    def _deep_find(obj: Any, depth: int = 0) -> dict[str, Any] | None:
        if depth > 12:
            return None
        if isinstance(obj, dict):
            if required <= obj.keys():
                return obj
            for v in obj.values():
                got = _deep_find(v, depth + 1)
                if got is not None:
                    return got
        if isinstance(obj, list):
            for it in obj:
                got = _deep_find(it, depth + 1)
                if got is not None:
                    return got
        return None

    found = _deep_find(parsed)
    if found is not None:
        return found

    if not _schema_expects_list_value(response_schema):
        flat: dict[str, Any] = {}
        _flatten_scalar_fields(parsed, flat)
        filled = _fill_required_from_flat_aliases(flat, response_schema)
        if required <= filled.keys():
            return filled

    missing = sorted(required - set(parsed.keys()))
    raise JudgeParseError(f"missing required keys: {missing}")


@dataclass
class JudgeResponse:
    raw: str
    parsed: dict
    model: str
    prompt_tokens: int
    completion_tokens: int


class JudgeError(Exception):
    """Raised when all retries are exhausted."""


class JudgeParseError(Exception):
    """Raised when the response is valid JSON but missing required keys."""


class JudgeClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def _validate_keys(self, parsed: dict, response_schema: dict) -> None:
        required = set(response_schema.keys())
        present = set(parsed.keys()) if isinstance(parsed, dict) else set()
        missing = required - present
        if missing:
            raise JudgeParseError(f"missing required keys: {sorted(missing)}")

    async def judge(
        self,
        prompt: str,
        response_schema: dict,
        system: str | None = None,
    ) -> JudgeResponse:
        litellm, APIError, RateLimitError = _load_litellm()

        sys_msg = system if system is not None else _DEFAULT_SYSTEM
        base = prompt.rstrip() + "\n\n" + _SCHEMA_SUFFIX
        last_exc: BaseException | None = None

        for attempt in range(self.max_retries):
            user_content = base + (_RETRY_HINT if attempt > 0 else "")
            messages: list[dict[str, str]] = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content},
            ]
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
            except RateLimitError as e:
                last_exc = e
                await asyncio.sleep(2**attempt)
                continue
            except APIError as e:
                last_exc = e
                continue

            try:
                choice = response.choices[0]
                content = choice.message.content
                raw = _message_content_as_text(content)
                cleaned = _strip_json_fences(raw)
                parsed_raw = json.loads(cleaned)
                parsed = _coerce_parsed_for_schema(parsed_raw, response_schema)
                self._validate_keys(parsed, response_schema)
                usage = getattr(response, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                return JudgeResponse(
                    raw=raw,
                    parsed=parsed,
                    model=self.model,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                )
            except (json.JSONDecodeError, JudgeParseError) as e:
                last_exc = e
                continue

        assert last_exc is not None
        raise JudgeError("judge call failed after all retries") from last_exc

    async def judge_batch(
        self,
        prompts: list[str],
        response_schema: dict,
        system: str | None = None,
        concurrency: int = 5,
    ) -> list[JudgeResponse]:
        sem = asyncio.Semaphore(max(concurrency, 1))
        results: list[JudgeResponse | None] = [None] * len(prompts)

        async def run_one(i: int, p: str) -> None:
            async with sem:
                try:
                    results[i] = await self.judge(p, response_schema, system=system)
                except Exception as e:
                    _LOG.warning("judge_batch item %s failed: %s", i, e)
                    results[i] = JudgeResponse(
                        raw="",
                        parsed={},
                        model=self.model,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )

        await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
        out: list[JudgeResponse] = []
        for r in results:
            if r is None:
                out.append(
                    JudgeResponse(
                        raw="",
                        parsed={},
                        model=self.model,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
                )
            else:
                out.append(r)
        return out
