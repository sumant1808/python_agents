"""
Wraps the pydantic-ai holiday_agent with TemporalAgent for durable execution.
Every LLM call and every tool call becomes a separate Temporal Activity.
"""
import sys
import os

# Add pydantic/ scripts directory to path so we can import holiday_planner directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pydantic"))

from dotenv import load_dotenv

load_dotenv()

from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.models.google import GoogleModel

from holiday_planner import holiday_agent  # noqa: E402  (sys.path patched above)

# Pre-register the model — TemporalAgent forbids creating Model instances
# inside a workflow (non-deterministic), so all models must be registered here.
temporal_holiday_agent: TemporalAgent = TemporalAgent(
    holiday_agent,
    name="holiday_planner",
    models={"gemini-3-flash-preview": GoogleModel("gemini-3-flash-preview")},
)
