"""LlamaIndex: rely on native OpenTelemetry when a global TracerProvider is set."""


def setup_llamaindex() -> None:
    """
    LlamaIndex has native OTel instrumentation via llama_index.core.instrumentation.
    When agentrace.trace() sets the global TracerProvider, LlamaIndex will
    automatically attach its spans to the same provider and they will appear
    in the AgentTrace.

    This function is a no-op — it exists only for documentation and to provide
    a consistent import surface. Call it if you want to be explicit:
        from agentrace.capture.adapters.llamaindex import setup_llamaindex
        setup_llamaindex()
    """
    pass  # intentional — LlamaIndex auto-detects the OTel provider
