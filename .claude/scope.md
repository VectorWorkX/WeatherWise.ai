# WeatherWise.ai - Scope & Behavior Guide

## Purpose
This document defines the "Ground Truth" for how the agent should categorize user queries and the quality of response expected.

## Bucket 1: Real-Time Data (Direct API)
**User Query:** "What is the temp in San Jose?"
**Intent:** Fetch current conditions.
**Expected Response:** "It is currently 72°F in San Jose with clear skies. Winds are mild at 5 mph."
**Success Criteria:** Must use `get_current_weather` tool. Do not hallucinate data.

## Bucket 2: Comparative Logic (Multi-Step)
**Intent:** Identify the weather variable(s) being compared from the query → Fetch data for each location → Compare the relevant values → Summarize with explicit differences.

**Example A — Temperature:**
**User Query:** "Is it hotter in Phoenix or Miami right now?"
**Expected Response:** "Phoenix is currently hotter. It is 102°F in Phoenix compared to 88°F in Miami, a difference of 14 degrees."

**Example B — Air Quality:**
**User Query:** "Is the air quality better in Denver or LA right now?"
**Expected Response:** "Denver has significantly better air quality. The AQI in Denver is 42 (Good) compared to 138 (Unhealthy for Sensitive Groups) in LA. If you have respiratory concerns, Denver is the safer choice today."

**Success Criteria:**
- Must call the API for each location separately.
- Must explicitly state the compared values and the difference or ranking.
- The compared variable is determined by the query, not hardcoded — comparisons can involve temperature, AQI, wind speed, snow accumulation, humidity, UV index, or any other available metric.

## Bucket 3: Lifestyle & Reasoning (Synthesizer)
**Intent:** Identify the activity or concern from the query → Determine which weather variables are contextually relevant (e.g., precip/temp/wind for outdoor events, AQI/humidity for exercise, snow/visibility for travel, UV index for beach plans) → Fetch those conditions → Apply contextual "Comfort & Safety" logic.

**Example A — Outdoor Event:**
**User Query:** "Is today good for a wedding in Napa?"
**Expected Response:** "It looks promising. Temperatures are a pleasant 75°F, but there is a 20% chance of light rain around 5 PM. I recommend having a tent or indoor option on standby just in case."

**Example B — Health & Air Quality:**
**User Query:** "Is it safe to go for a run in Delhi today?"
**Expected Response:** "I'd advise against outdoor running today. The AQI in Delhi is currently 285 (Very Unhealthy), with PM2.5 levels at 180 µg/m³. Combined with 34°C heat and 70% humidity, this poses serious respiratory and heat stress risks. Consider an indoor workout or wait until early morning when AQI typically drops."

**Success Criteria:**
- The agent must dynamically select weather variables based on the activity and location context, not rely on a fixed set (temp, wind, precip).
- Relevant variables include but are not limited to: temperature, precipitation, wind, AQI, humidity, UV index, snow accumulation, visibility, dew point, and pollen count.
- Tone should be advisory and helpful. Must highlight potential risks and offer actionable alternatives when conditions are unfavorable.

## Bucket 4: Safety & Alerts (Priority)
**Intent:** Detect that the query involves safety or active hazards → Query the severe weather alerts endpoint for the relevant location → Return the alert with urgency-appropriate formatting and actionable guidance.

**Example A — Tornado:**
**User Query:** "Are there any active tornado warnings in my county?"
**Expected Response:** "⚠️ **URGENT:** There is an active Tornado Watch until 9:00 PM. Please monitor local news and stay near a shelter."

**Example B — Heat Advisory:**
**User Query:** "Any weather warnings for Phoenix this week?"
**Expected Response:** "🔶 **ADVISORY:** An Excessive Heat Warning is in effect through Friday. Temperatures are expected to reach 115°F. Limit outdoor activity to early morning hours, stay hydrated, and never leave children or pets in vehicles."

**Success Criteria:**
- Must query the alerts endpoint, not infer hazards from forecast data alone.
- Alert types include but are not limited to: tornado, hurricane, flood, heat advisory, winter storm, air quality, wildfire smoke, and thunderstorm warnings.
- Urgency formatting must scale with severity — immediate life-threat alerts (tornado, hurricane) use bold urgent framing; advisories (heat, AQI) use a cautionary but informative tone.
- Must include actionable safety guidance specific to the alert type.

## Bucket 5: Explainability (Deep Atmospheric RAG)
**User Query:** "Why is there a heat dome over the Midwest?"
**Intent:** Query ChromaDB (Multi-Vector Retrieval) for "heat dome mechanism" + Fetch current pressure data from Xweather MCP.
**Expected Response:** "This heat wave is being driven by a persistent **Omega Block** pattern. A high-pressure ridge has stalled over the Great Plains, causing air to sink (subsidence). As this air sinks, it compresses and warms up. Current data shows 500mb height anomalies are significantly above normal. *[Based on analysis of the 500mb geopotential height chart from NOAA Technical Memorandum OAR-GSD-68, which shows the characteristic ridge axis over the central US consistent with the current blocking pattern.]*"
**Success Criteria:**
- Must use scientific terms (subsidence, blocking pattern, omega block).
- Must combine RAG-retrieved knowledge with live Xweather data in a single coherent response.
- When visual data (charts, graphs, satellite imagery) was ingested for the topic, the response must include citations or descriptions derived from that visual data (e.g., referencing chart trends, axis values, or spatial patterns).
- Retrieved source documents must be traceable via `doc_id` metadata.

## Bucket 6: Data Operations & Math (Tool Interoperability)
**Intent:** Identify the weather variables and mathematical operation from the query → Fetch the required data points → Perform the calculation (aggregation, conversion, averaging, difference, etc.) → Present both the raw values and the computed result.

**Example A — Summation & Unit Conversion:**
**User Query:** "Add the temperature of Boston and New York and give the answer in Kelvin."
**Expected Response:** "Currently, Boston is 10°C and New York is 12°C. The sum of these temperatures is 22°C. Converted to Kelvin, that is **295.15 K**."

**Example B — Aggregation & Averaging:**
**User Query:** "What's the average wind speed across Chicago, Detroit, and Milwaukee?"
**Expected Response:** "Current wind speeds: Chicago 18 mph, Detroit 12 mph, Milwaukee 22 mph. The average wind speed across these three cities is **17.3 mph**."

**Success Criteria:**
- Must accurately fetch live values for each location — never hallucinate or estimate data.
- The mathematical operation is determined by the query and can include: addition, subtraction, averaging, min/max, unit conversion (F↔C↔K, mph↔km/h, in↔cm), percentage differences, or any combination.
- Must show intermediate values (the raw fetched data) before presenting the final computed result, so the user can verify the math.
- All operands must be in consistent units before calculation. If the API returns mixed units, normalize first and state the conversion.