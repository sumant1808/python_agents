"""
Temporal Cloud worker for the holiday planner.

Connects to Temporal Cloud via API key + TLS, then polls the 'holiday-planner'
task queue. Keep this running in a terminal while submitting workflow requests.

Required env vars (add to .env):
    TEMPORAL_HOST       e.g. myns.abc123.tmprl.cloud:7233
    TEMPORAL_NAMESPACE  e.g. myns.abc123
    TEMPORAL_API_KEY    Temporal Cloud API key

Usage:
    uv run python temporal/worker.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pydantic"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker, WorkerConfig
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from workflows import HolidayPlannerWorkflow  # noqa: E402

TASK_QUEUE = "holiday-planner"


async def main() -> None:
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
    print("Connected.")

    # WorkerConfig is a TypedDict — configure_worker() auto-discovers
    # temporal_activities from HolidayPlannerWorkflow.__pydantic_ai_agents__
    # and extends the activities list. It also configures the Pydantic data
    # converter and workflow sandbox passthrough modules.
    config: WorkerConfig = {
        "task_queue": TASK_QUEUE,
        "workflows": [HolidayPlannerWorkflow],
        "activities": [],
        "workflow_runner": SandboxedWorkflowRunner(),
    }
    config = PydanticAIPlugin().configure_worker(config)

    worker = Worker(client, **config)
    print(f"Worker polling task queue: '{TASK_QUEUE}'  (Ctrl+C to stop)")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
