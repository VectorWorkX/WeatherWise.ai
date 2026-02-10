# WeatherWise.ai - Scope & Behavior Guide

## Purpose
This document defines the "Ground Truth" for how the agent should categorize user queries and the quality of response expected. These examples represent the complexity level we target — multi-variable lookups, multi-location reasoning, domain-specific advisory logic, and scientific depth.

## Bucket 1: Real-Time Data (Direct API)
**User Query:** "What is the wind speed and gust spread in San Francisco right now?"
**Intent:** Fetch current wind conditions for a single location, including both sustained speed and gust data, then compute the spread (difference between gust and sustained wind).
**Expected Response:** "Current winds in San Francisco are sustained at 18 mph with gusts reaching 31 mph. That gives a gust spread of 13 mph, which indicates moderate turbulence — typical for the coastal gap flow through the Golden Gate. Conditions may feel notably gustier than the sustained speed suggests, especially near the waterfront."
**Success Criteria:**
- Must use `get_current_weather` (or equivalent MCP tool) to fetch live wind data. Do not hallucinate values.
- Must return both sustained wind speed and gust speed as distinct values.
- Must compute the gust spread (gust minus sustained) and present it explicitly.
- Should provide brief contextual interpretation of what the spread means for real-world conditions.

## Bucket 2: Comparative Logic (Multi-Step)
**User Query:** "How does the current winter storm severity in Buffalo compare to Boston's accumulation?"
**Intent:** Identify the weather variables implied by "winter storm severity" (snowfall accumulation, wind speed, temperature, visibility) → Fetch current conditions and storm data for both Buffalo and Boston → Compare the relevant severity indicators → Summarize with explicit differences and a severity assessment.
**Expected Response:** "Buffalo is experiencing significantly more severe winter storm conditions. Current snowfall accumulation in Buffalo is 14.2 inches with sustained winds of 35 mph and near-whiteout visibility at 0.1 miles. Boston has received 4.8 inches of accumulation with winds at 22 mph and visibility holding at 0.8 miles. Buffalo's accumulation is nearly 3x Boston's, and the combination of high winds and low visibility puts Buffalo in blizzard-condition territory while Boston is seeing a moderate snow event."
**Success Criteria:**
- Must call the API for each location separately.
- Must identify and compare multiple relevant variables for "storm severity" — not just one metric. At minimum: accumulation, wind speed, and visibility.
- Must explicitly state the compared values and the magnitude of difference.
- Should provide an overall severity assessment that synthesizes the individual comparisons into a coherent picture.

## Bucket 3: Lifestyle & Reasoning (Synthesizer)
**User Query:** "I'm planning a ski trip to Colorado in February. Based on current snowpack and wind forecasts, which resort town should I target?"
**Intent:** Identify the activity (skiing) and relevant weather variables (snowpack/snow depth, recent snowfall, wind speed, temperature, visibility) → Determine candidate resort locations in Colorado (e.g., Vail, Breckenridge, Aspen, Steamboat Springs) → Fetch conditions for each → Apply contextual reasoning about ski quality (fresh powder, wind hold risks, lift operation conditions) → Recommend with justification.
**Expected Response:** "Based on current conditions, I'd recommend Steamboat Springs. It has the deepest recent snowfall at 18 inches in the last 48 hours with a total base depth of 67 inches. Winds are moderate at 12 mph — well below the 30+ mph threshold that typically causes lift holds. Breckenridge has a solid base (72 inches) but received less fresh snow (6 inches) and is seeing gusty winds at 28 mph that could affect upper-mountain lifts. Vail sits in between with 11 inches of fresh snow and calmer winds. For the best powder day with minimal wind disruption, Steamboat is your pick."
**Success Criteria:**
- Must dynamically select weather variables based on the activity context — for skiing: snow depth/accumulation, recent snowfall, wind speed, temperature, and optionally visibility.
- Must compare at least 2-3 candidate locations rather than answering for a single location.
- Must synthesize multiple variables into a cohesive recommendation, not just list raw data.
- Tone should be advisory and helpful. Must highlight potential risks (e.g., wind holds, icy conditions) and justify the recommendation.

## Bucket 4: Safety & Alerts (Priority)
**User Query:** "My logistics fleet is traveling from Dallas to Denver between Feb 7th and 9th. Are there any severe weather advisories or icing risks along this route?"
**Intent:** Detect that the query involves safety, route-based hazards, and a time window → Identify key waypoints along the Dallas-to-Denver corridor (e.g., Dallas TX, Amarillo TX, Trinidad CO, Colorado Springs CO, Denver CO) → Query severe weather alerts and winter conditions for each waypoint → Assess icing risks (freezing rain, ice accumulation, road surface temps) → Return a route-aware safety briefing with urgency-appropriate formatting.
**Expected Response:** "**Route Safety Briefing: Dallas → Denver (Feb 7-9)**\n\n**Amarillo, TX:** A Winter Storm Warning is in effect Feb 7-8. Expected ice accumulation of 0.25-0.5 inches with freezing rain transitioning to snow. Wind gusts to 45 mph will create hazardous driving on I-40.\n\n**Trinidad, CO / Raton Pass:** Winter Weather Advisory through Feb 8. Snow accumulation of 4-8 inches on I-25 over the pass. Chain requirements likely.\n\n**Colorado Springs → Denver:** Moderate snow expected Feb 8 (2-4 inches) but roads should remain passable with reduced speeds.\n\n**Recommendation:** Delay departure until Feb 9 if possible. The worst icing risk is the Amarillo segment on Feb 7. If the fleet must depart on schedule, route via I-35 N to Wichita then I-70 W to avoid the worst of the Texas Panhandle ice."
**Success Criteria:**
- Must query alerts for multiple waypoints along the route, not just origin and destination.
- Alert types assessed should include: winter storm warnings, ice storm warnings, wind advisories, freezing rain, road condition alerts.
- Urgency formatting must scale with severity — immediate hazards (ice storms, blizzard warnings) get bold/urgent framing; advisories get cautionary tone.
- Must include actionable safety guidance specific to the scenario — route alternatives, timing recommendations, or preparation advice.
- Must address the specific time window mentioned in the query.

## Bucket 5: Explainability (Deep Atmospheric RAG)
**User Query:** "Why did the 2023 Canadian wildfire smoke travel so far south into the US, and what atmospheric blocking patterns caused it?"
**Intent:** Query ChromaDB (Multi-Vector Retrieval) for "Canadian wildfire smoke transport," "atmospheric blocking patterns," and "2023 wildfire season" → Fetch any current or historical atmospheric data from Xweather MCP for context → Synthesize RAG-retrieved scientific knowledge with data into a coherent explanation citing sources.
**Expected Response:** "The unprecedented southward transport of Canadian wildfire smoke in June 2023 was driven by a persistent **upper-level ridge** that anchored over eastern Canada, creating a **Rex Block** pattern. This blocking configuration deflected the jet stream southward, creating a direct atmospheric highway from Quebec into the US Northeast and Midwest.\n\nThe smoke plume was injected into the upper troposphere (above 5 km altitude) by the intense heat of the fires — a process called **pyroconvection** — which allowed it to travel thousands of kilometers without being washed out by precipitation. Satellite imagery from GOES-16 showed the plume extending from Hudson Bay to Washington, D.C. in a continuous arc.\n\nThe blocking pattern persisted for nearly two weeks because the jet stream was anomalously weak at those latitudes, a condition increasingly linked to **Arctic amplification** — the faster warming of polar regions reducing the equator-to-pole temperature gradient that drives jet stream strength. *[Based on analysis from the NOAA Climate Risk Assessment CPO-003 and NASA/AMS Great Lakes atmospheric transport study, which document the relationship between blocking patterns and long-range aerosol transport.]*"
**Success Criteria:**
- Must use scientific terms appropriately (pyroconvection, Rex Block, Arctic amplification, upper-level ridge, jet stream).
- Must combine RAG-retrieved knowledge with live or contextual Xweather data in a single coherent response.
- When visual data (charts, graphs, satellite imagery) was ingested for the topic, the response must include citations or descriptions derived from that visual data.
- Retrieved source documents must be traceable via `doc_id` metadata.
- The explanation must address both parts of the question: the transport mechanism AND the blocking pattern cause.

## Bucket 6: Data Operations & Math (Tool Interoperability)
**User Query:** "Calculate the difference between Chicago's current wind chill and Miami's heat index in degrees Celsius."
**Intent:** Identify the weather variables (wind chill for Chicago, heat index for Miami) and mathematical operation (difference, with unit conversion to Celsius) → Fetch wind chill data for Chicago and heat index data for Miami → Convert both to Celsius if returned in Fahrenheit → Compute the difference → Present raw values, conversions, and final result.
**Expected Response:** "Current conditions:\n- **Chicago wind chill:** -8°F → Converted: -22.2°C\n- **Miami heat index:** 97°F → Converted: 36.1°C\n\nThe difference between Miami's heat index and Chicago's wind chill is **58.3°C** (36.1 - (-22.2) = 58.3). That's a staggering thermal spread — while Miami feels like a sweltering tropical afternoon, Chicago's exposed-skin conditions approach frostbite territory."
**Success Criteria:**
- Must accurately fetch live values for each location — never hallucinate or estimate data.
- Must fetch the correct derived variables: wind chill (not just temperature) for Chicago, and heat index (not just temperature) for Miami.
- Must show intermediate values (raw fetched data in original units) before presenting the final computed result.
- All operands must be in consistent units (Celsius) before the final calculation. Must explicitly show the unit conversion steps.
- The mathematical operation is determined by the query — here it's subtraction after conversion.
