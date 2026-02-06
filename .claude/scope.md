# WeatherWise.ai - Scope & Behavior Guide

## Purpose
This document defines the "Ground Truth" for how the agent should categorize user queries and the quality of response expected.

## Bucket 1: Real-Time Data (Direct API)
**User Query:** "What is the temp in San Jose?"
**Intent:** Fetch current conditions.
**Expected Response:** "It is currently 72°F in San Jose with clear skies. Winds are mild at 5 mph."
**Success Criteria:** Must use `get_current_weather` tool. Do not hallucinate data.

## Bucket 2: Comparative Logic (Multi-Step)
**User Query:** "Is it hotter in Phoenix or Miami right now?"
**Intent:** Fetch data for two locations -> Compare values -> Summarize.
**Expected Response:** "Phoenix is currently hotter. It is 102°F in Phoenix compared to 88°F in Miami, a difference of 14 degrees."
**Success Criteria:** Must call API twice. Must explicitly state the difference.

## Bucket 3: Lifestyle & Reasoning (Synthesizer)
**User Query:** "Is today good for a wedding in Napa?"
**Intent:** Check precip, temp, wind -> Apply "Comfort" logic.
**Expected Response:** "It looks promising. Temperatures are a pleasant 75°F, but there is a 20% chance of light rain around 5 PM. I recommend having a tent or indoor option on standby just in case."
**Success Criteria:** Tone should be advisory and helpful. Highlights potential risks.

## Bucket 4: Safety & Alerts (Priority)
**User Query:** "Are there any active tornado warnings in my county?"
**Intent:** Check severe weather alerts endpoint.
**Expected Response:** "⚠️ **URGENT:** There is an active Tornado Watch until 9:00 PM. Please monitor local news and stay near a shelter."
**Success Criteria:** High priority tone. Clear, bold warning signs.

## Bucket 5: Explainability (Deep Atmospheric RAG)
**User Query:** "Why is there a heat dome over the Midwest?"
**Intent:** Query Vector DB for "heat dome mechanism" + Fetch current pressure data.
**Expected Response:** "This heat wave is being driven by a persistent **Omega Block** pattern. A high-pressure ridge has stalled over the Great Plains, causing air to sink (subsidence). As this air sinks, it compresses and warms up. Current data shows 500mb height anomalies are significantly above normal."
**Success Criteria:** Must use scientific terms (subsidence, blocking pattern).

## Bucket 6: Data Operations & Math (Tool Interoperability)
**User Query:** "Add the temperature of Boston and New York and give the answer in Kelvin."
**Intent:** Fetch Temp(Boston) + Fetch Temp(New York) → Perform Summation → Apply Kelvin Conversion Formula ($T_c + 273.15$).
**Expected Response:** "Currently, Boston is 10°C and New York is 12°C. The sum of these temperatures is 22°C. Converted to Kelvin, that is **295.15 K**."
**Success Criteria:** Must accurately fetch values and calculate the result without hallucinating.