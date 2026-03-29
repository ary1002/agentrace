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
                raw = content if isinstance(content, str) else str(content or "")
                cleaned = _strip_json_fences(raw)
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError("expected JSON object", cleaned, 0)
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
