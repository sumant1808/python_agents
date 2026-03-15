"""
Temporal Workflow definition for the holiday planner.

HolidayPlannerWorkflow orchestrates the pydantic-ai agent run as a durable
workflow. Each LLM call and tool call (find_cities, find_flights_and_hotels,
prioritize_options) is automatically dispatched as an individual Temporal Activity
by TemporalAgent — no manual @activity.defn wiring needed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pydantic"))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel
from temporalio import workflow
from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow

from holiday_planner import HolidayDeps, HolidayPlan  # noqa: E402
from agent import temporal_holiday_agent  # noqa: E402


class HolidayRequest(BaseModel):
    """Serializable workflow input — passed across the Temporal task queue boundary."""

    user_prompt: str
    budget_gbp: float = 2000.0
    duration_nights: int = 7
    travel_month: str = "June"
    origin_city: str = "London"


@workflow.defn
class HolidayPlannerWorkflow(PydanticAIWorkflow):
    # PydanticAIPlugin.configure_worker() reads this attribute to auto-register
    # all temporal_activities from the wrapped TemporalAgent.
    __pydantic_ai_agents__ = [temporal_holiday_agent]

    @workflow.run
    async def run(self, req: HolidayRequest) -> HolidayPlan:
        deps = HolidayDeps(
            budget_gbp=req.budget_gbp,
            duration_nights=req.duration_nights,
            travel_month=req.travel_month,
            origin_city=req.origin_city,
        )
        # temporal_holiday_agent.run() dispatches every model request and
        # every tool call as a separate Activity, giving full durability.
        result = await temporal_holiday_agent.run(req.user_prompt, deps=deps)
        return result.output
