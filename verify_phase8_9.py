import asyncio

import agentrace
from agentrace.dataset.dataset import Dataset
from agentrace.metrics.llm_judge.judge_client import JudgeClient
from agentrace.report.cli_reporter import CLIReporter
from agentrace.runner.models import EvalTask


async def test_judge():
    client = JudgeClient(model="claude-sonnet-4-20250514")
    response = await client.judge(
        prompt="Evaluate this: agent called 'search' then returned answer. Score quality 0-1.",
        response_schema={"score": "float", "reasoning": "str"},
    )
    print(f'Judge response score: {response.parsed["score"]}')
    print(f"Tokens used: {response.prompt_tokens} + {response.completion_tokens}")


async def test_evaluate():
    async def my_agent(query: str) -> str:
        import openai

        client = openai.AsyncOpenAI()
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
        )
        return "answer"

    dataset = Dataset(
        [
            EvalTask(
                id="t1",
                query="What is the capital of France?",
                expected_tools=[],
                optimal_steps=2,
                difficulty="easy",
            ),
        ]
    )

    result = await agentrace.evaluate(
        agent=my_agent,
        dataset=dataset,
        metrics=[
            "tool_selection_accuracy",
            "step_efficiency",
            "reasoning_coherence",
            "trajectory_optimality",
        ],
        judge_model="claude-sonnet-4-20250514",
    )

    CLIReporter().print_results(result)

    print(f"Failure dist: {result.failure_dist}")
    assert "trajectory_optimality" in result.aggregate_scores
    print("All assertions passed.")


asyncio.run(test_judge())
asyncio.run(test_evaluate())
