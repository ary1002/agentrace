"""LlamaIndex integration helpers for binding AgentTrace session providers."""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider


def _try_set_global_handler(handler, **kwargs) -> bool:
    try:
        handler(**kwargs)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def setup_llamaindex(provider: TracerProvider | None = None) -> None:
    """Best-effort LlamaIndex OTel handler setup for the active provider."""
    if provider is None:
        return
    try:
        from llama_index.core.instrumentation import set_global_handler
    except ImportError:
        return

    # LlamaIndex handler signatures vary by release; probe common forms.
    attempts = [
        {"handler": "opentelemetry", "tracer_provider": provider},
        {"handler": "opentelemetry", "provider": provider},
        {"handler": "opentelemetry", "trace_provider": provider},
    ]
    for kwargs in attempts:
        if _try_set_global_handler(set_global_handler, **kwargs):
            return


def reset_llamaindex() -> None:
    """Best-effort reset for LlamaIndex global instrumentation handler."""
    try:
        from llama_index.core.instrumentation import set_global_handler
    except ImportError:
        return
    for kwargs in ({"handler": None}, {}):
        if _try_set_global_handler(set_global_handler, **kwargs):
            return
