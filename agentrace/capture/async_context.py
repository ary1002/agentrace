"""OpenTelemetry context attach/detach — single import surface for ``opentelemetry.context``.

Callers should use these helpers instead of importing ``opentelemetry.context`` directly.
"""

from __future__ import annotations

from contextvars import Token

from opentelemetry.context import Context, attach, detach, get_current


def attach_context(parent_ctx: Context) -> Token[Context]:
    """Attach ``parent_ctx`` and return an opaque token for :func:`detach_context`."""
    return attach(parent_ctx)


def detach_context(token: Token[Context]) -> None:
    """Detach a context previously attached; call from ``finally`` blocks."""
    detach(token)


def get_current_context() -> Context:
    """Return the current OpenTelemetry context."""
    return get_current()
