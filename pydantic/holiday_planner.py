import asyncio
import hashlib
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class FlightOption(BaseModel):
    airline: str
    destination: str
    duration_hours: float
    price_gbp: float
    cabin_class: Literal["economy", "premium_economy", "business"]


class HotelOption(BaseModel):
    name: str
    city: str
    stars: int
    price_per_night_gbp: float
    nights: int
    total_cost_gbp: float


class DestinationPackage(BaseModel):
    region: Literal["Europe", "Asia", "South America"]
    city: str
    flight: FlightOption
    hotel: HotelOption
    total_package_cost_gbp: float


class HolidayPlan(BaseModel):
    query: str
    recommended_packages: list[DestinationPackage] = Field(
        description="Up to 3 packages sorted cheapest-first"
    )
    top_pick: DestinationPackage
    reasoning: str
    travel_tips: list[str]


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

@dataclass
class HolidayDeps:
    budget_gbp: float = 2000.0
    duration_nights: int = 7
    travel_month: str = "June"
    origin_city: str = "London"


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

REGION_CITIES: dict[str, list[str]] = {
    "Europe": ["Paris", "Rome", "Barcelona", "Amsterdam"],
    "Asia": ["Tokyo", "Bangkok", "Bali", "Singapore"],
    "South America": ["Buenos Aires", "Rio de Janeiro", "Cartagena", "Lima"],
}

REGION_AIRLINES: dict[str, list[str]] = {
    "Europe": ["British Airways", "easyJet", "Ryanair"],
    "Asia": ["Cathay Pacific", "Singapore Airlines", "Qatar Airways"],
    "South America": ["LATAM", "Iberia", "Air France"],
}

REGION_FLIGHT_BASE_GBP: dict[str, float] = {
    "Europe": 180.0,
    "Asia": 420.0,
    "South America": 680.0,
}

REGION_HOTEL_BASE_GBP: dict[str, float] = {
    "Europe": 120.0,
    "Asia": 80.0,
    "South America": 95.0,
}

REGION_DURATION_HOURS: dict[str, float] = {
    "Europe": 2.5,
    "Asia": 11.0,
    "South America": 14.0,
}

HOTEL_NAMES: list[str] = [
    "The Grand {city} Hotel",
    "{city} Central Boutique",
    "{city} Comfort Inn",
]


def _seed(city: str) -> int:
    return int(hashlib.md5(city.encode()).hexdigest()[:8], 16)


def _mock_flights(city: str, region: str) -> list[dict]:
    base = REGION_FLIGHT_BASE_GBP[region]
    airlines = REGION_AIRLINES[region]
    duration = REGION_DURATION_HOURS[region]
    s = _seed(city)
    multipliers = [1.0, 1.3, 1.7]
    cabins: list[Literal["economy", "premium_economy", "business"]] = [
        "economy", "premium_economy", "business"
    ]
    return [
        {
            "airline": airlines[i % len(airlines)],
            "destination": city,
            "duration_hours": round(duration + (s % 3) * 0.5 * (i + 1) * 0.1, 1),
            "price_gbp": round(base * multipliers[i] + (s % 50), 2),
            "cabin_class": cabins[i],
        }
        for i in range(3)
    ]


def _mock_hotels(city: str, region: str, nights: int) -> list[dict]:
    base = REGION_HOTEL_BASE_GBP[region]
    s = _seed(city)
    multipliers = [1.0, 1.5, 2.2]
    stars = [3, 4, 5]
    return [
        {
            "name": HOTEL_NAMES[i].format(city=city),
            "city": city,
            "stars": stars[i],
            "price_per_night_gbp": round(base * multipliers[i] + (s % 20), 2),
            "nights": nights,
            "total_cost_gbp": round((base * multipliers[i] + (s % 20)) * nights, 2),
        }
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

holiday_agent = Agent[HolidayDeps, HolidayPlan](
    "gemini-3-flash-preview",
    deps_type=HolidayDeps,
    output_type=HolidayPlan,
    system_prompt="""You are an expert holiday planning assistant.
When a user describes their ideal holiday, follow this exact process:

1. THINK about which region(s) suit the user's request: Europe, Asia, South America.
   Consider all three if the user is open-minded.

2. Call find_cities() for each region you want to explore to get city options.

3. Pick the most suitable city per region and call find_flights_and_hotels()
   to retrieve available flights and hotels.

4. Call prioritize_options(city, region) for each city — it fetches data internally
   and returns packages ranked by total cost. Do NOT pass flights or hotels to it.

5. Return a HolidayPlan with up to 3 packages (cheapest first), a clear top_pick,
   your reasoning, and 3-5 practical travel tips.

Always respect the user's budget and trip duration. Cheapest total cost is the
primary ranking criterion; star rating breaks ties.""",
)


# ---------------------------------------------------------------------------
# Tools — mapping directly to flowchart nodes
# ---------------------------------------------------------------------------

@holiday_agent.tool
async def find_cities(
    ctx: RunContext[HolidayDeps],
    region: Literal["Europe", "Asia", "South America"],
) -> list[str]:
    """Return top holiday cities for the given region (flowchart node: Find cities)."""
    return REGION_CITIES.get(region, [])


@holiday_agent.tool
async def find_flights_and_hotels(
    ctx: RunContext[HolidayDeps],
    city: str,
    region: Literal["Europe", "Asia", "South America"],
) -> dict[str, list[dict]]:
    """Find available flights and hotels for a city (flowchart: find flights and hotels)."""
    flights = _mock_flights(city, region)
    hotels = _mock_hotels(city, region, ctx.deps.duration_nights)
    return {"flights": flights, "hotels": hotels}


@holiday_agent.tool
async def prioritize_options(
    ctx: RunContext[HolidayDeps],
    city: str,
    region: Literal["Europe", "Asia", "South America"],
) -> list[dict]:
    """Rank flight+hotel packages by total cost, cheapest first (flowchart node: G).
    Regenerates flight and hotel data internally — no need to pass them back."""
    flights = _mock_flights(city, region)
    hotels = _mock_hotels(city, region, ctx.deps.duration_nights)

    packages = []
    for flight in flights:
        for hotel in hotels:
            total = round(flight["price_gbp"] + hotel["total_cost_gbp"], 2)
            if total <= ctx.deps.budget_gbp:
                packages.append(
                    {
                        "region": region,
                        "city": city,
                        "flight": flight,
                        "hotel": hotel,
                        "total_package_cost_gbp": total,
                    }
                )

    packages.sort(key=lambda p: p["total_package_cost_gbp"])
    return packages[:3]


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------

def _print_holiday_plan(plan: HolidayPlan) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  YOUR HOLIDAY PLAN")
    print(sep)
    print(f"\nTOP PICK: {plan.top_pick.city}, {plan.top_pick.region}")
    print(f"Total Cost:  £{plan.top_pick.total_package_cost_gbp:.2f}")
    print(f"  Flight:  {plan.top_pick.flight.airline} ({plan.top_pick.flight.cabin_class})"
          f"  —  £{plan.top_pick.flight.price_gbp:.2f} / {plan.top_pick.flight.duration_hours}h")
    print(f"  Hotel:   {plan.top_pick.hotel.name} ({plan.top_pick.hotel.stars}★)"
          f"  —  £{plan.top_pick.hotel.price_per_night_gbp:.2f}/night × {plan.top_pick.hotel.nights} nights"
          f"  =  £{plan.top_pick.hotel.total_cost_gbp:.2f}")

    print(f"\nReasoning: {plan.reasoning}")

    print(f"\nAll packages (cheapest first):")
    for i, pkg in enumerate(plan.recommended_packages, 1):
        print(f"  {i}. {pkg.city:20s} £{pkg.total_package_cost_gbp:.2f}"
              f"  (flight £{pkg.flight.price_gbp:.2f} + hotel £{pkg.hotel.total_cost_gbp:.2f})")

    print(f"\nTravel tips:")
    for tip in plan.travel_tips:
        print(f"  • {tip}")
    print()


# ---------------------------------------------------------------------------
# Main — streaming iter loop
# ---------------------------------------------------------------------------

async def main() -> None:
    deps = HolidayDeps(
        budget_gbp=2000.0,
        duration_nights=7,
        travel_month="June",
        origin_city="London",
    )

    user_prompt = (
        "I want a 7-night holiday in June with a budget of £2000. "
        "I enjoy warm weather, great food, and cultural experiences. "
        "I'm open to Europe, Asia, or South America — find me the best value."
    )

    print(f"\n{'='*60}")
    print("  HOLIDAY PLANNER — Starting agent run")
    print(f"{'='*60}")
    print(f"\n[User] {user_prompt}\n")

    async with holiday_agent.iter(user_prompt, deps=deps) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                pass  # already printed above

            elif Agent.is_model_request_node(node):
                print("[Agent] Thinking...")
                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            part = event.part
                            if hasattr(part, "tool_name"):
                                print(f"  -> Preparing tool call: {part.tool_name}")
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, ThinkingPartDelta):
                                print(f"  [thinking] {event.delta.content_delta}", end="", flush=True)
                            elif isinstance(event.delta, ToolCallPartDelta):
                                pass  # suppress verbose arg streaming
                        elif isinstance(event, FinalResultEvent):
                            print("\n[Agent] Producing final holiday plan...", flush=True)
                            final_result_found = True
                            break

                    if final_result_found:
                        async for _ in request_stream.stream_output():
                            pass  # consume structured output stream

            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            print(f"\n[Tool] '{event.part.tool_name}' called")
                            print(f"       args: {event.part.args}")
                        elif isinstance(event, FunctionToolResultEvent):
                            print(f"[Tool] result received")

            elif Agent.is_end_node(node):
                assert run.result is not None
                _print_holiday_plan(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
