"""Golden reference responses and success criteria extracted from scope.md.

Each GoldenResponse contains the expected behavior for one intent bucket,
used by the LLM judge to evaluate actual system responses.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class GoldenResponse(BaseModel):
    """A reference response with its scoring rubric."""

    bucket: str = Field(description="Intent bucket (e.g., 'real_time')")
    query: str = Field(description="The user query")
    reference_response: str = Field(description="Expected response from scope.md")
    success_criteria: List[str] = Field(description="Bullet-point criteria from scope.md")
    scoring_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension weight overrides (dimension_name -> weight)",
    )
    locations: List[str] = Field(
        default_factory=list,
        description="Known location names in query and response (used by code assertions)",
    )


GOLDEN_RESPONSES: List[GoldenResponse] = [
    # Bucket 1: Real-Time Data
    GoldenResponse(
        bucket="real_time",
        query="What is the wind speed and gust spread in San Francisco right now?",
        reference_response=(
            "Current winds in San Francisco are sustained at 18 mph with gusts "
            "reaching 31 mph. That gives a gust spread of 13 mph, which indicates "
            "moderate turbulence — typical for the coastal gap flow through the "
            "Golden Gate. Conditions may feel notably gustier than the sustained "
            "speed suggests, especially near the waterfront."
        ),
        success_criteria=[
            "Must use get_current_weather (or equivalent MCP tool) to fetch live wind data. Do not hallucinate values.",
            "Must return both sustained wind speed and gust speed as distinct values.",
            "Must compute the gust spread (gust minus sustained) and present it explicitly.",
            "Should provide brief contextual interpretation of what the spread means for real-world conditions.",
        ],
        scoring_weights={"factual_grounding": 1.5, "completeness": 1.0, "formatting": 0.5},
        locations=["San Francisco"],
    ),
    # Bucket 2: Comparative Logic
    GoldenResponse(
        bucket="comparative",
        query="How does the current winter storm severity in Buffalo compare to Boston's accumulation?",
        reference_response=(
            "Buffalo is experiencing significantly more severe winter storm conditions. "
            "Current snowfall accumulation in Buffalo is 14.2 inches with sustained winds "
            "of 35 mph and near-whiteout visibility at 0.1 miles. Boston has received 4.8 "
            "inches of accumulation with winds at 22 mph and visibility holding at 0.8 miles. "
            "Buffalo's accumulation is nearly 3x Boston's, and the combination of high winds "
            "and low visibility puts Buffalo in blizzard-condition territory while Boston is "
            "seeing a moderate snow event."
        ),
        success_criteria=[
            "Must call the API for each location separately.",
            "Must identify and compare multiple relevant variables for 'storm severity' — not just one metric. At minimum: accumulation, wind speed, and visibility.",
            "Must explicitly state the compared values and the magnitude of difference.",
            "Should provide an overall severity assessment that synthesizes the individual comparisons into a coherent picture.",
        ],
        scoring_weights={"factual_grounding": 1.5, "completeness": 1.5, "formatting": 1.0},
        locations=["Buffalo", "Boston"],
    ),
    # Bucket 3: Lifestyle & Reasoning
    GoldenResponse(
        bucket="lifestyle",
        query=(
            "I'm planning a ski trip to Colorado in February. Based on current "
            "snowpack and wind forecasts, which resort town should I target?"
        ),
        reference_response=(
            "Based on current conditions, I'd recommend Steamboat Springs. It has the "
            "deepest recent snowfall at 18 inches in the last 48 hours with a total base "
            "depth of 67 inches. Winds are moderate at 12 mph — well below the 30+ mph "
            "threshold that typically causes lift holds. Breckenridge has a solid base "
            "(72 inches) but received less fresh snow (6 inches) and is seeing gusty winds "
            "at 28 mph that could affect upper-mountain lifts. Vail sits in between with "
            "11 inches of fresh snow and calmer winds. For the best powder day with minimal "
            "wind disruption, Steamboat is your pick."
        ),
        success_criteria=[
            "Must dynamically select weather variables based on the activity context — for skiing: snow depth/accumulation, recent snowfall, wind speed, temperature, and optionally visibility.",
            "Must compare at least 2-3 candidate locations rather than answering for a single location.",
            "Must synthesize multiple variables into a cohesive recommendation, not just list raw data.",
            "Tone should be advisory and helpful. Must highlight potential risks (e.g., wind holds, icy conditions) and justify the recommendation.",
        ],
        scoring_weights={"completeness": 1.5, "actionability": 1.5, "formatting": 0.5},
        locations=["Steamboat Springs", "Breckenridge", "Vail"],
    ),
    # Bucket 4: Safety & Alerts
    GoldenResponse(
        bucket="safety",
        query=(
            "My logistics fleet is traveling from Dallas to Denver between Feb 7th "
            "and 9th. Are there any severe weather advisories or icing risks along this route?"
        ),
        reference_response=(
            "**Route Safety Briefing: Dallas → Denver (Feb 7-9)**\n\n"
            "**Amarillo, TX:** A Winter Storm Warning is in effect Feb 7-8. Expected ice "
            "accumulation of 0.25-0.5 inches with freezing rain transitioning to snow. "
            "Wind gusts to 45 mph will create hazardous driving on I-40.\n\n"
            "**Trinidad, CO / Raton Pass:** Winter Weather Advisory through Feb 8. Snow "
            "accumulation of 4-8 inches on I-25 over the pass. Chain requirements likely.\n\n"
            "**Colorado Springs → Denver:** Moderate snow expected Feb 8 (2-4 inches) but "
            "roads should remain passable with reduced speeds.\n\n"
            "**Recommendation:** Delay departure until Feb 9 if possible. The worst icing "
            "risk is the Amarillo segment on Feb 7. If the fleet must depart on schedule, "
            "route via I-35 N to Wichita then I-70 W to avoid the worst of the Texas "
            "Panhandle ice."
        ),
        success_criteria=[
            "Must query alerts for multiple waypoints along the route, not just origin and destination.",
            "Alert types assessed should include: winter storm warnings, ice storm warnings, wind advisories, freezing rain, road condition alerts.",
            "Urgency formatting must scale with severity — immediate hazards get bold/urgent framing; advisories get cautionary tone.",
            "Must include actionable safety guidance specific to the scenario — route alternatives, timing recommendations, or preparation advice.",
            "Must address the specific time window mentioned in the query.",
        ],
        scoring_weights={"completeness": 1.5, "actionability": 2.0, "formatting": 1.0, "factual_grounding": 1.5},
        locations=["Dallas", "Denver", "Amarillo", "Trinidad", "Colorado Springs"],
    ),
    # Bucket 5: Explainability (RAG)
    GoldenResponse(
        bucket="explanation",
        query=(
            "Why did the 2023 Canadian wildfire smoke travel so far south into the US, "
            "and what atmospheric blocking patterns caused it?"
        ),
        reference_response=(
            "The unprecedented southward transport of Canadian wildfire smoke in June 2023 "
            "was driven by a persistent **upper-level ridge** that anchored over eastern Canada, "
            "creating a **Rex Block** pattern. This blocking configuration deflected the jet "
            "stream southward, creating a direct atmospheric highway from Quebec into the US "
            "Northeast and Midwest.\n\n"
            "The smoke plume was injected into the upper troposphere (above 5 km altitude) by "
            "the intense heat of the fires — a process called **pyroconvection** — which allowed "
            "it to travel thousands of kilometers without being washed out by precipitation. "
            "Satellite imagery from GOES-16 showed the plume extending from Hudson Bay to "
            "Washington, D.C. in a continuous arc.\n\n"
            "The blocking pattern persisted for nearly two weeks because the jet stream was "
            "anomalously weak at those latitudes, a condition increasingly linked to **Arctic "
            "amplification** — the faster warming of polar regions reducing the equator-to-pole "
            "temperature gradient that drives jet stream strength. *[Based on analysis from the "
            "NOAA Climate Risk Assessment CPO-003 and NASA/AMS Great Lakes atmospheric transport "
            "study, which document the relationship between blocking patterns and long-range "
            "aerosol transport.]*"
        ),
        success_criteria=[
            "Must use scientific terms appropriately (pyroconvection, Rex Block, Arctic amplification, upper-level ridge, jet stream).",
            "Must combine RAG-retrieved knowledge with live or contextual Xweather data in a single coherent response.",
            "When visual data (charts, graphs, satellite imagery) was ingested for the topic, the response must include citations or descriptions derived from that visual data.",
            "Retrieved source documents must be traceable via doc_id metadata.",
            "The explanation must address both parts of the question: the transport mechanism AND the blocking pattern cause.",
        ],
        scoring_weights={
            "scientific_accuracy": 2.0,
            "citation_quality": 1.5,
            "completeness": 1.5,
            "faithfulness": 1.5,
            "context_precision": 1.0,
            "context_recall": 1.0,
        },
        locations=[],
    ),
    # Bucket 6: Data Operations & Math
    GoldenResponse(
        bucket="math",
        query="Calculate the difference between Chicago's current wind chill and Miami's heat index in degrees Celsius.",
        reference_response=(
            "Current conditions:\n"
            "- **Chicago wind chill:** -8°F → Converted: -22.2°C\n"
            "- **Miami heat index:** 97°F → Converted: 36.1°C\n\n"
            "The difference between Miami's heat index and Chicago's wind chill is **58.3°C** "
            "(36.1 - (-22.2) = 58.3). That's a staggering thermal spread — while Miami feels "
            "like a sweltering tropical afternoon, Chicago's exposed-skin conditions approach "
            "frostbite territory."
        ),
        success_criteria=[
            "Must accurately fetch live values for each location — never hallucinate or estimate data.",
            "Must fetch the correct derived variables: wind chill (not just temperature) for Chicago, and heat index (not just temperature) for Miami.",
            "Must show intermediate values (raw fetched data in original units) before presenting the final computed result.",
            "All operands must be in consistent units (Celsius) before the final calculation. Must explicitly show the unit conversion steps.",
            "The mathematical operation is determined by the query — here it's subtraction after conversion.",
        ],
        scoring_weights={"factual_grounding": 2.0, "completeness": 1.5, "formatting": 1.0},
        locations=["Chicago", "Miami"],
    ),
]
