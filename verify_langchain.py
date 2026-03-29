import asyncio
import uuid

import agentrace
from agentrace import AgentTraceCallbackHandler


async def test():
    async with agentrace.trace(session_id="lc_test", task="test") as ctx:
        handler = AgentTraceCallbackHandler()
        run_id = uuid.uuid4()
        # Simulate LangChain callbacks manually
        handler.on_llm_start(
            serialized={"name": "ChatOpenAI"},
            prompts=["Hello"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=type(
                "R",
                (),
                {
                    "generations": [
                        [type("G", (), {"text": "Hi there"})()],
                    ],
                    "llm_output": {
                        "token_usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                        },
                    },
                },
            )(),
            run_id=run_id,
        )

    t = ctx.agent_trace
    llm_spans = [s for s in t.spans if s.span_type == "llm_call"]
    print(f"llm_call spans: {len(llm_spans)}")  # should be 1
    print(f"framework: {llm_spans[0].framework}")  # should be 'langchain'
    print(f"tokens: {llm_spans[0].token_count}")  # should show 10 + 5


asyncio.run(test())
