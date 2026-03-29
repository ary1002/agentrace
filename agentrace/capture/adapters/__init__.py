"""Optional integrations: LangChain, LangGraph, SDKs, LlamaIndex, CrewAI."""

from agentrace.capture.adapters.anthropic_sdk import patch_anthropic
from agentrace.capture.adapters.crewai import instrument_crew, make_task_callback
from agentrace.capture.adapters.langchain import AgentTraceCallbackHandler
from agentrace.capture.adapters.langgraph import instrument_graph, traced_node
from agentrace.capture.adapters.llamaindex import setup_llamaindex
from agentrace.capture.adapters.openai_sdk import patch_openai

__all__ = [
    "patch_openai",
    "patch_anthropic",
    "AgentTraceCallbackHandler",
    "traced_node",
    "instrument_graph",
    "instrument_crew",
    "make_task_callback",
    "setup_llamaindex",
]
