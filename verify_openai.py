import asyncio

import agentrace


async def test():
    async with agentrace.trace(session_id="oai_test", task="test") as ctx:
        try:
            import openai

            client = openai.AsyncOpenAI(api_key="sk-fake")
            await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )
        except Exception:
            pass  # API call will fail with fake key — that's fine

    t = ctx.agent_trace
    llm_spans = [s for s in t.spans if s.span_type == "llm_call"]
    print(f"llm_call spans: {len(llm_spans)}")  # should be >= 1
    print(f"framework: {llm_spans[0].framework}")  # should be 'openai'


asyncio.run(test())
