"""
Trigger a HolidayPlannerWorkflow execution on Temporal Cloud.

The worker (worker.py) must already be running before executing this script.

Usage:
    uv run python temporal/run_workflow.py
    uv run python temporal/run_workflow.py --prompt "Beach holiday in July, budget £3000"
"""
import asyncio
import os
import sys
import uuid
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows import HolidayPlannerWorkflow, HolidayRequest  # noqa: E402

TASK_QUEUE = "holiday-planner"

DEFAULT_PROMPT = (
    "I want a 7-night holiday in June with a budget of £2000. "
    "I enjoy warm weather, great food, and cultural experiences. "
    "I'm open to Europe, Asia, or South America — find me the best value."
)


async def main(user_prompt: str) -> None:
    host = os.environ["TEMPORAL_HOST"]
    namespace = os.environ["TEMPORAL_NAMESPACE"]
    api_key = os.environ["TEMPORAL_API_KEY"]

    print(f"Connecting to Temporal Cloud: {host} / namespace={namespace}")
    client = await Client.connect(
        host,
        namespace=namespace,
        api_key=api_key,
        tls=True,
        data_converter=pydantic_data_converter,
    )

    request = HolidayRequest(user_prompt=user_prompt)
    workflow_id = f"holiday-{uuid.uuid4()}"

    print(f"Starting workflow id={workflow_id}")
    print(f"Prompt: {user_prompt}\n")

    result = await client.execute_workflow(
        HolidayPlannerWorkflow.run,
        request,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    print("\n" + "=" * 60)
    print("  WORKFLOW COMPLETE")
    print("=" * 60)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger a holiday planner workflow")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Holiday request prompt")
    args = parser.parse_args()
    asyncio.run(main(args.prompt))
