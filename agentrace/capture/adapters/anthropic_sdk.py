"""Anthropic Python SDK integration by monkey-patching ``client.messages.create``."""

from __future__ import annotations

_anthropic_patched = False


def _anthropic_output_dict(response) -> dict:
    text_content = ""
    tool_use_blocks: list[dict] = []
    for block in getattr(response, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text" and not text_content:
            text_content = getattr(block, "text", "") or ""
        elif btype == "tool_use":
            tool_use_blocks.append(
                {
                    "tool_name": getattr(block, "name", ""),
                    "tool_use_id": getattr(block, "id", ""),
                    "input": getattr(block, "input", {}),
                }
            )
    return {
        "content": text_content,
        "tool_calls": tool_use_blocks,
        "stop_reason": getattr(response, "stop_reason", "") or "",
    }


def patch_anthropic() -> None:
    global _anthropic_patched
    if _anthropic_patched:
        return
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        return

    from agentrace.capture.adapters._span_utils import (
        compute_cost,
        get_tracer,
        record_exception,
        set_span_attributes,
    )

    original_sync = Messages.create

    def patched_sync(self_inner, *args, **kwargs):
        tracer = get_tracer()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")
        input_dict = {"model": model, "messages": messages, "system": system}

        with tracer.start_as_current_span(f"anthropic/{model}") as span:
            try:
                response = original_sync(self_inner, *args, **kwargs)
                usage = getattr(response, "usage", None)
                prompt_tokens = usage.input_tokens if usage else 0
                completion_tokens = usage.output_tokens if usage else 0
                cost = compute_cost(str(model), int(prompt_tokens), int(completion_tokens))
                output_dict = _anthropic_output_dict(response)
                set_span_attributes(
                    span,
                    "llm_call",
                    input_dict,
                    output_dict,
                    "anthropic",
                    int(prompt_tokens),
                    int(completion_tokens),
                    cost,
                )
                for block in getattr(response, "content", None) or []:
                    if getattr(block, "type", None) == "tool_use":
                        bname = getattr(block, "name", "unknown")
                        with tracer.start_as_current_span(f"tool/{bname}") as tspan:
                            set_span_attributes(
                                tspan,
                                "tool_call",
                                {
                                    "tool_name": bname,
                                    "input": getattr(block, "input", {}),
                                },
                                {},
                                "anthropic",
                            )
                return response
            except Exception as e:
                record_exception(span, e)
                raise

    original_async = AsyncMessages.create

    async def patched_async(self_inner, *args, **kwargs):
        tracer = get_tracer()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")
        input_dict = {"model": model, "messages": messages, "system": system}

        with tracer.start_as_current_span(f"anthropic/{model}") as span:
            try:
                response = await original_async(self_inner, *args, **kwargs)
                usage = getattr(response, "usage", None)
                prompt_tokens = usage.input_tokens if usage else 0
                completion_tokens = usage.output_tokens if usage else 0
                cost = compute_cost(str(model), int(prompt_tokens), int(completion_tokens))
                output_dict = _anthropic_output_dict(response)
                set_span_attributes(
                    span,
                    "llm_call",
                    input_dict,
                    output_dict,
                    "anthropic",
                    int(prompt_tokens),
                    int(completion_tokens),
                    cost,
                )
                for block in getattr(response, "content", None) or []:
                    if getattr(block, "type", None) == "tool_use":
                        bname = getattr(block, "name", "unknown")
                        with tracer.start_as_current_span(f"tool/{bname}") as tspan:
                            set_span_attributes(
                                tspan,
                                "tool_call",
                                {
                                    "tool_name": bname,
                                    "input": getattr(block, "input", {}),
                                },
                                {},
                                "anthropic",
                            )
                return response
            except Exception as e:
                record_exception(span, e)
                raise

    Messages.create = patched_sync
    AsyncMessages.create = patched_async
    _anthropic_patched = True
