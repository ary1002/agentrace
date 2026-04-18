"""OpenAI Python SDK integration by monkey-patching ``client.chat.completions.create``."""

from __future__ import annotations

_openai_patched = False


def patch_openai() -> None:
    global _openai_patched
    if _openai_patched:
        return
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        return

    from agentrace.capture.adapters._span_utils import (
        compute_cost,
        get_tracer,
        record_exception,
        set_span_attributes,
    )

    original_sync = Completions.create

    def patched_sync(self_inner, *args, **kwargs):
        tracer = get_tracer()
        model = kwargs.get("model", args[0] if args else "unknown")
        messages = kwargs.get("messages", [])
        input_dict = {"model": model, "messages": messages}

        with tracer.start_as_current_span(f"openai/{model}") as span:
            span.set_attribute("agentrace.span_type", "llm_call")
            span.set_attribute("agentrace.framework", "openai")
            try:
                response = original_sync(self_inner, *args, **kwargs)
                usage = getattr(response, "usage", None)
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                cost = compute_cost(
                    str(model), int(prompt_tokens), int(completion_tokens)
                )

                output_dict = {
                    "content": (
                        response.choices[0].message.content if response.choices else ""
                    ),
                    "finish_reason": (
                        response.choices[0].finish_reason if response.choices else ""
                    ),
                }
                set_span_attributes(
                    span,
                    "llm_call",
                    input_dict,
                    output_dict,
                    "openai",
                    int(prompt_tokens),
                    int(completion_tokens),
                    cost,
                )
                msg = response.choices[0].message if response.choices else None
                tool_calls = getattr(msg, "tool_calls", None) if msg else None
                for tc in tool_calls or []:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "unknown") if fn else "unknown"
                    arguments = getattr(fn, "arguments", "") if fn else ""
                    with tracer.start_as_current_span(f"tool/{name}") as tspan:
                        set_span_attributes(
                            tspan,
                            "tool_call",
                            {"tool_name": name, "arguments": arguments},
                            {},
                            "openai",
                        )
                return response
            except Exception as e:
                record_exception(span, e)
                raise

    original_async = AsyncCompletions.create

    async def patched_async(self_inner, *args, **kwargs):
        tracer = get_tracer()
        model = kwargs.get("model", args[0] if args else "unknown")
        messages = kwargs.get("messages", [])
        input_dict = {"model": model, "messages": messages}

        with tracer.start_as_current_span(f"openai/{model}") as span:
            span.set_attribute("agentrace.span_type", "llm_call")
            span.set_attribute("agentrace.framework", "openai")
            try:
                response = await original_async(self_inner, *args, **kwargs)
                usage = getattr(response, "usage", None)
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                cost = compute_cost(
                    str(model), int(prompt_tokens), int(completion_tokens)
                )

                output_dict = {
                    "content": (
                        response.choices[0].message.content if response.choices else ""
                    ),
                    "finish_reason": (
                        response.choices[0].finish_reason if response.choices else ""
                    ),
                }
                set_span_attributes(
                    span,
                    "llm_call",
                    input_dict,
                    output_dict,
                    "openai",
                    int(prompt_tokens),
                    int(completion_tokens),
                    cost,
                )
                msg = response.choices[0].message if response.choices else None
                tool_calls = getattr(msg, "tool_calls", None) if msg else None
                for tc in tool_calls or []:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "unknown") if fn else "unknown"
                    arguments = getattr(fn, "arguments", "") if fn else ""
                    with tracer.start_as_current_span(f"tool/{name}") as tspan:
                        set_span_attributes(
                            tspan,
                            "tool_call",
                            {"tool_name": name, "arguments": arguments},
                            {},
                            "openai",
                        )
                return response
            except Exception as e:
                record_exception(span, e)
                raise

    Completions.create = patched_sync  # type: ignore[method-assign, assignment]
    AsyncCompletions.create = patched_async  # type: ignore[method-assign, assignment]
    _openai_patched = True
